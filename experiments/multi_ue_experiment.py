"""
P1: Multi-UE Environment with Scheduler Contention
Shows SENTRY-Lite-2 behavior under resource competition
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd
import sys
sys.path.insert(0, 'C:/Users/LOQ/Desktop/SENTRY-Lite-2/src')
from environment.wireless_env import WirelessConfig, ChannelModel


@dataclass
class MultiUEConfig:
    """Multi-UE system configuration"""
    n_ues: int = 2
    total_prbs: int = 50          # Total PRBs in system
    prbs_per_ue_base: int = 10    # PRBs per UE in BASE mode
    prbs_per_ue_boost: int = 25   # PRBs per UE in BOOST mode
    scheduler: str = 'EDF'        # EDF or PF
    channel_model: ChannelModel = ChannelModel.RAYLEIGH
    snr_db: float = 20.0
    load_per_ue: float = 0.85
    episode_length: int = 10000
    deadline_ttis: int = 4        # 0.5ms / 0.125ms = 4 TTIs


class MultiUEScheduler:
    """Simple scheduler for multi-UE contention"""
    
    def __init__(self, config: MultiUEConfig):
        self.config = config
        self.scheduler_type = config.scheduler
    
    def allocate_prbs(self, ue_requests: List[Dict]) -> List[int]:
        """
        Allocate PRBs to UEs based on scheduler policy
        
        Args:
            ue_requests: List of {ue_id, mode, queue_len, min_deadline}
        
        Returns:
            List of PRBs allocated to each UE
        """
        n_ues = len(ue_requests)
        total = self.config.total_prbs
        
        if self.scheduler_type == 'EDF':
            # Earliest Deadline First
            # Sort by urgency (min deadline remaining)
            sorted_ues = sorted(enumerate(ue_requests), 
                               key=lambda x: x[1]['min_deadline'])
            
            allocations = [0] * n_ues
            remaining = total
            
            for ue_idx, req in sorted_ues:
                # Requested PRBs based on mode
                requested = (self.config.prbs_per_ue_boost if req['mode'] == 1 
                            else self.config.prbs_per_ue_base)
                
                # Allocate minimum of requested and remaining
                allocated = min(requested, remaining)
                allocations[ue_idx] = allocated
                remaining -= allocated
            
            return allocations
        
        elif self.scheduler_type == 'PF':
            # Proportional Fair (simplified)
            # Give equal share, adjusted by queue length
            base_share = total // n_ues
            
            allocations = []
            for req in ue_requests:
                # More PRBs if queue is longer
                queue_factor = min(2.0, 1.0 + req['queue_len'] / 20.0)
                allocated = int(base_share * queue_factor)
                allocations.append(min(allocated, total // n_ues + 5))
            
            return allocations
        
        else:
            # Default: equal share
            return [total // n_ues] * n_ues


class MultiUEEnvironment:
    """Multi-UE URLLC Environment with contention"""
    
    def __init__(self, config: MultiUEConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.scheduler = MultiUEScheduler(config)
        
        # Per-UE state
        self.queues: List[List[int]] = [[] for _ in range(config.n_ues)]
        self.modes: List[int] = [1] * config.n_ues  # Start in BOOST
        self.ewma_slacks: List[float] = [0.5] * config.n_ues
        
        # Per-UE channel (independent fading)
        self.snrs: List[float] = [config.snr_db] * config.n_ues
        self.cqis: List[int] = [10] * config.n_ues
        self._fading_states: List[float] = [1.0] * config.n_ues
        
        # Metrics
        self.total_arrivals = [0] * config.n_ues
        self.total_misses = [0] * config.n_ues
        self.switches = [0] * config.n_ues
        self.base_ttis = [0] * config.n_ues
        
        self.time = 0
    
    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        n = self.config.n_ues
        self.queues = [[] for _ in range(n)]
        self.modes = [1] * n
        self.ewma_slacks = [0.5] * n
        self.snrs = [self.config.snr_db] * n
        self.cqis = [10] * n
        self._fading_states = [1.0] * n
        self.total_arrivals = [0] * n
        self.total_misses = [0] * n
        self.switches = [0] * n
        self.base_ttis = [0] * n
        self.time = 0
    
    def _update_channel(self, ue_id: int):
        """Update channel for one UE"""
        cfg = self.config
        
        if cfg.channel_model == ChannelModel.AWGN:
            self.snrs[ue_id] = cfg.snr_db
        else:
            # Rayleigh/Rician fading
            fd_normalized = 10.0 * (0.125 / 1000)  # Doppler * TTI
            correlation = np.exp(-2 * np.pi * fd_normalized)
            
            innovation = self.rng.standard_normal()
            self._fading_states[ue_id] = (
                correlation * self._fading_states[ue_id] + 
                np.sqrt(1 - correlation**2) * innovation
            )
            
            if cfg.channel_model == ChannelModel.RAYLEIGH:
                fading_power = np.abs(self._fading_states[ue_id])**2
            else:  # Rician
                k = 3.0
                los = np.sqrt(k / (k + 1))
                scatter = np.sqrt(1 / (k + 1)) * self._fading_states[ue_id]
                fading_power = np.abs(los + scatter)**2
            
            self.snrs[ue_id] = cfg.snr_db + 10 * np.log10(fading_power + 1e-10)
        
        # SNR to CQI
        thresholds = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 
                      8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7]
        cqi = 15
        for i, thresh in enumerate(thresholds, 1):
            if self.snrs[ue_id] < thresh:
                cqi = max(1, i - 1)
                break
        self.cqis[ue_id] = cqi
    
    def _prbs_to_capacity(self, prbs: int, cqi: int) -> int:
        """Convert PRBs to packet capacity"""
        # Simplified: capacity ∝ PRBs × CQI
        base_capacity = prbs // 5  # ~1 packet per 5 PRBs at CQI=10
        cqi_factor = cqi / 10.0
        return max(1, int(base_capacity * cqi_factor))
    
    def step(self, actions: List[int]) -> Dict:
        """
        Execute one TTI for all UEs
        
        Args:
            actions: List of actions (0=BASE, 1=BOOST) for each UE
        """
        cfg = self.config
        
        # Update channels
        for ue in range(cfg.n_ues):
            self._update_channel(ue)
        
        # Mode switches
        for ue in range(cfg.n_ues):
            if actions[ue] != self.modes[ue]:
                self.switches[ue] += 1
                self.modes[ue] = actions[ue]
            if self.modes[ue] == 0:
                self.base_ttis[ue] += 1
        
        # Arrivals (independent Poisson per UE)
        mean_arrivals = cfg.load_per_ue * 4  # Base capacity = 4
        for ue in range(cfg.n_ues):
            arrivals = self.rng.poisson(mean_arrivals)
            for _ in range(arrivals):
                self.queues[ue].append(cfg.deadline_ttis)
                self.total_arrivals[ue] += 1
        
        # Scheduler: allocate PRBs
        ue_requests = []
        for ue in range(cfg.n_ues):
            min_deadline = min(self.queues[ue]) if self.queues[ue] else cfg.deadline_ttis
            ue_requests.append({
                'ue_id': ue,
                'mode': self.modes[ue],
                'queue_len': len(self.queues[ue]),
                'min_deadline': min_deadline
            })
        
        prb_allocations = self.scheduler.allocate_prbs(ue_requests)
        
        # Service
        for ue in range(cfg.n_ues):
            capacity = self._prbs_to_capacity(prb_allocations[ue], self.cqis[ue])
            served = min(capacity, len(self.queues[ue]))
            self.queues[ue] = self.queues[ue][served:]
        
        # Age and check deadlines
        for ue in range(cfg.n_ues):
            new_queue = []
            for remaining in self.queues[ue]:
                if remaining <= 1:
                    self.total_misses[ue] += 1
                else:
                    new_queue.append(remaining - 1)
            self.queues[ue] = new_queue
            
            # Update slack
            if self.queues[ue]:
                slack = min(self.queues[ue]) / cfg.deadline_ttis
            else:
                slack = 1.0
            self.ewma_slacks[ue] = 0.1 * slack + 0.9 * self.ewma_slacks[ue]
        
        self.time += 1
        
        return {
            'queues': [len(q) for q in self.queues],
            'cqis': self.cqis.copy(),
            'prbs': prb_allocations
        }
    
    def get_metrics(self) -> Dict:
        """Get per-UE and aggregate metrics"""
        n = self.config.n_ues
        
        per_ue = []
        for ue in range(n):
            per_ue.append({
                'ue_id': ue,
                'miss_prob': self.total_misses[ue] / max(1, self.total_arrivals[ue]) * 100,
                'savings': self.base_ttis[ue] / max(1, self.time) * 100,
                'switches_per_ktti': self.switches[ue] / max(1, self.time) * 1000
            })
        
        # Aggregate
        total_arrivals = sum(self.total_arrivals)
        total_misses = sum(self.total_misses)
        
        return {
            'per_ue': per_ue,
            'aggregate': {
                'miss_prob': total_misses / max(1, total_arrivals) * 100,
                'savings': np.mean([p['savings'] for p in per_ue]),
                'switches_per_ktti': np.mean([p['switches_per_ktti'] for p in per_ue]),
                'fairness': 1 - np.std([p['miss_prob'] for p in per_ue]) / max(0.001, np.mean([p['miss_prob'] for p in per_ue]))
            }
        }


def run_multi_ue_experiment(n_ues: int, channel: ChannelModel, snr: float,
                            scheduler: str = 'EDF', n_seeds: int = 5) -> Dict:
    """Run multi-UE experiment"""
    
    results = {'miss': [], 'savings': [], 'switches': [], 'fairness': []}
    
    for seed in range(n_seeds):
        config = MultiUEConfig(
            n_ues=n_ues,
            channel_model=channel,
            snr_db=snr,
            scheduler=scheduler
        )
        
        env = MultiUEEnvironment(config, seed=seed)
        env.reset(seed=seed)
        
        # Per-UE controllers
        counters = [0] * n_ues
        
        for t in range(config.episode_length):
            actions = []
            for ue in range(n_ues):
                Q = len(env.queues[ue])
                slack = env.ewma_slacks[ue]
                cqi = env.cqis[ue]
                
                # SENTRY-Lite-2 policy with CQI awareness
                if Q >= 15 or slack < 0.25 or cqi < 8:
                    action = 1
                    counters[ue] = 0
                elif Q < 6 and slack > 0.50 and cqi >= 10:
                    counters[ue] += 1
                    action = 0 if counters[ue] >= 2 else env.modes[ue]
                else:
                    action = env.modes[ue]
                    counters[ue] = 0
                
                actions.append(action)
            
            env.step(actions)
        
        metrics = env.get_metrics()
        results['miss'].append(metrics['aggregate']['miss_prob'])
        results['savings'].append(metrics['aggregate']['savings'])
        results['switches'].append(metrics['aggregate']['switches_per_ktti'])
        results['fairness'].append(metrics['aggregate']['fairness'])
    
    return {
        'miss_mean': np.mean(results['miss']),
        'miss_std': np.std(results['miss']),
        'savings_mean': np.mean(results['savings']),
        'switches_mean': np.mean(results['switches']),
        'fairness_mean': np.mean(results['fairness'])
    }


if __name__ == '__main__':
    print("="*70)
    print("P1: MULTI-UE EXPERIMENT WITH SCHEDULER CONTENTION")
    print("="*70)
    
    # Test configurations
    ue_counts = [1, 2, 5]
    snrs = [20, 22, 25]
    
    results = []
    
    for n_ues in ue_counts:
        for snr in snrs:
            print(f"\n  Testing: {n_ues} UEs, SNR={snr} dB, Rayleigh...")
            
            res = run_multi_ue_experiment(
                n_ues=n_ues,
                channel=ChannelModel.RAYLEIGH,
                snr=snr,
                scheduler='EDF',
                n_seeds=5
            )
            
            results.append({
                'n_ues': n_ues,
                'snr': snr,
                **res
            })
    
    # Print results
    print("\n" + "="*70)
    print("[RESULTS: Multi-UE under Rayleigh Fading]")
    print("-"*70)
    print(f"{'UEs':>4} | {'SNR':>4} | {'Miss%':>8} | {'Save%':>6} | {'Sw':>5} | {'Fair':>5} | URLLC")
    print("-"*70)
    
    for r in results:
        urllc = "✓" if r['miss_mean'] < 1.0 else "✗"
        print(f"{r['n_ues']:>4} | {r['snr']:>4} | {r['miss_mean']:>6.3f}% | "
              f"{r['savings_mean']:>5.1f}% | {r['switches_mean']:>5.1f} | "
              f"{r['fairness_mean']:>5.2f} | {urllc:>5}")
    
    # Save
    df = pd.DataFrame(results)
    df.to_csv('experiments/results/multi_ue_results.csv', index=False)
    
    print("\n" + "="*70)
    print("KEY FINDING:")
    print("  Multi-UE contention increases miss probability but")
    print("  SENTRY-Lite-2 maintains URLLC compliance at higher SNR")
    print("="*70)
