"""
FIXED Multi-UE Environment - Aligned with Single-UE capacity model
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import sys
sys.path.insert(0, 'C:/Users/LOQ/Desktop/SENTRY-Lite-2/src')
from environment.wireless_env import WirelessConfig, ChannelModel


@dataclass
class MultiUEConfig:
    """Multi-UE system configuration"""
    n_ues: int = 2
    total_capacity_base: int = 8      # Total BASE capacity (2 UEs × 4)
    total_capacity_boost: int = 20    # Total BOOST capacity (2 UEs × 10)
    scheduler: str = 'EDF'
    channel_model: ChannelModel = ChannelModel.RAYLEIGH
    snr_db: float = 20.0
    load_per_ue: float = 0.85
    episode_length: int = 10000
    deadline_ttis: int = 4
    
    # Per-UE capacities (for fair comparison)
    capacity_base_per_ue: int = 4
    capacity_boost_per_ue: int = 10


class MultiUEEnvironment:
    """Multi-UE URLLC Environment - Fixed capacity model"""
    
    def __init__(self, config: MultiUEConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        n = config.n_ues
        self.queues: List[List[int]] = [[] for _ in range(n)]
        self.modes: List[int] = [1] * n
        self.ewma_slacks: List[float] = [0.5] * n
        
        # Per-UE channel state
        self.snrs: List[float] = [config.snr_db] * n
        self.cqis: List[int] = [10] * n
        self._fading_states: List[float] = [1.0] * n
        
        # Metrics
        self.total_arrivals = [0] * n
        self.total_misses = [0] * n
        self.switches = [0] * n
        self.base_ttis = [0] * n
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
            fd_normalized = 10.0 * (0.125 / 1000)
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
    
    def _get_capacity(self, ue_id: int) -> int:
        """Get capacity for UE based on mode and CQI"""
        cfg = self.config
        mode = self.modes[ue_id]
        cqi = self.cqis[ue_id]
        
        # Base capacity same as single-UE model
        base_cap = cfg.capacity_base_per_ue if mode == 0 else cfg.capacity_boost_per_ue
        
        # CQI scaling (same as original)
        cqi_factor = cqi / 10.0
        effective_cap = int(base_cap * cqi_factor)
        
        return max(1, effective_cap)
    
    def _schedule_resources(self) -> List[int]:
        """
        Scheduler: allocate capacity to UEs
        Returns capacity allocation per UE
        """
        cfg = self.config
        n = cfg.n_ues
        
        # Calculate total requested capacity
        requests = []
        for ue in range(n):
            cap = self._get_capacity(ue)
            min_deadline = min(self.queues[ue]) if self.queues[ue] else cfg.deadline_ttis
            requests.append({
                'ue': ue,
                'capacity': cap,
                'queue_len': len(self.queues[ue]),
                'min_deadline': min_deadline,
                'urgency': len(self.queues[ue]) / max(1, min_deadline)
            })
        
        # Calculate total system capacity (contention)
        # Assume total capacity is limited when multiple UEs in BOOST
        boost_count = sum(1 for ue in range(n) if self.modes[ue] == 1)
        base_count = n - boost_count
        
        # Contention model: total capacity scales sub-linearly with UEs
        # This reflects resource contention (PRBs, scheduler overhead)
        if n == 1:
            contention_factor = 1.0
        elif n == 2:
            contention_factor = 0.95  # Slight contention
        else:
            contention_factor = 0.85  # More contention with 5+ UEs
        
        # EDF Scheduler: prioritize by urgency
        if cfg.scheduler == 'EDF':
            sorted_ues = sorted(requests, key=lambda x: -x['urgency'])
            
            allocations = [0] * n
            
            for req in sorted_ues:
                ue = req['ue']
                # Capacity with contention factor
                cap = int(req['capacity'] * contention_factor)
                allocations[ue] = max(1, cap)
            
            return allocations
        
        else:  # Round Robin / Fair
            return [int(self._get_capacity(ue) * contention_factor) for ue in range(n)]
    
    def step(self, actions: List[int]) -> Dict:
        """Execute one TTI"""
        cfg = self.config
        n = cfg.n_ues
        
        # Update channels
        for ue in range(n):
            self._update_channel(ue)
        
        # Mode switches
        for ue in range(n):
            if actions[ue] != self.modes[ue]:
                self.switches[ue] += 1
                self.modes[ue] = actions[ue]
            if self.modes[ue] == 0:
                self.base_ttis[ue] += 1
        
        # Arrivals
        mean_arrivals = cfg.load_per_ue * cfg.capacity_base_per_ue
        for ue in range(n):
            arrivals = self.rng.poisson(mean_arrivals)
            for _ in range(arrivals):
                self.queues[ue].append(cfg.deadline_ttis)
                self.total_arrivals[ue] += 1
        
        # Schedule and serve
        capacities = self._schedule_resources()
        
        for ue in range(n):
            served = min(capacities[ue], len(self.queues[ue]))
            self.queues[ue] = self.queues[ue][served:]
        
        # Age and check deadlines
        for ue in range(n):
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
        return {'cqis': self.cqis.copy(), 'capacities': capacities}
    
    def get_metrics(self) -> Dict:
        """Get metrics"""
        n = self.config.n_ues
        
        per_ue = []
        for ue in range(n):
            per_ue.append({
                'ue_id': ue,
                'miss_prob': self.total_misses[ue] / max(1, self.total_arrivals[ue]) * 100,
                'savings': self.base_ttis[ue] / max(1, self.time) * 100,
                'switches_per_ktti': self.switches[ue] / max(1, self.time) * 1000
            })
        
        total_arrivals = sum(self.total_arrivals)
        total_misses = sum(self.total_misses)
        
        miss_probs = [p['miss_prob'] for p in per_ue]
        
        return {
            'per_ue': per_ue,
            'aggregate': {
                'miss_prob': total_misses / max(1, total_arrivals) * 100,
                'savings': np.mean([p['savings'] for p in per_ue]),
                'switches_per_ktti': np.mean([p['switches_per_ktti'] for p in per_ue]),
                'fairness': 1 - np.std(miss_probs) / max(0.001, np.mean(miss_probs)) if np.mean(miss_probs) > 0 else 1.0
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
        
        counters = [0] * n_ues
        
        for t in range(config.episode_length):
            actions = []
            for ue in range(n_ues):
                Q = len(env.queues[ue])
                slack = env.ewma_slacks[ue]
                cqi = env.cqis[ue]
                
                # SENTRY-Lite-2 with CQI awareness
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
    print("P1 FIXED: MULTI-UE EXPERIMENT WITH CORRECT CAPACITY MODEL")
    print("="*70)
    
    ue_counts = [1, 2, 5]
    snrs = [20, 22, 25]
    
    results = []
    
    for n_ues in ue_counts:
        for snr in snrs:
            print(f"  Testing: {n_ues} UEs, SNR={snr} dB, Rayleigh...")
            
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
    
    print("\n" + "="*70)
    print("[RESULTS: Multi-UE under Rayleigh Fading - FIXED]")
    print("-"*70)
    print(f"{'UEs':>4} | {'SNR':>4} | {'Miss%':>8} | {'Save%':>6} | {'Sw':>5} | {'Fair':>5} | URLLC")
    print("-"*70)
    
    for r in results:
        urllc = "✓" if r['miss_mean'] < 1.0 else "✗"
        print(f"{r['n_ues']:>4} | {r['snr']:>4} | {r['miss_mean']:>6.3f}% | "
              f"{r['savings_mean']:>5.1f}% | {r['switches_mean']:>5.1f} | "
              f"{r['fairness_mean']:>5.2f} | {urllc:>5}")
    
    # Find operational boundary
    print("\n[OPERATIONAL BOUNDARY]")
    for n_ues in ue_counts:
        ue_results = [r for r in results if r['n_ues'] == n_ues]
        compliant = [r for r in ue_results if r['miss_mean'] < 1.0]
        if compliant:
            min_snr = min(r['snr'] for r in compliant)
            print(f"  {n_ues} UEs: URLLC compliance at SNR ≥ {min_snr} dB")
        else:
            print(f"  {n_ues} UEs: No URLLC compliance in tested range")
    
    df = pd.DataFrame(results)
    df.to_csv('experiments/results/multi_ue_results_fixed.csv', index=False)
    
    print("\n" + "="*70)
