"""
CQI-Adaptive SENTRY v2: Aggressive Tuning for Fading Channels
Target: Achieve URLLC compliance (< 1% miss) under Rayleigh/Rician
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import sys
sys.path.insert(0, 'C:/Users/LOQ/Desktop/SENTRY-Lite-2/src')
from environment.wireless_env import WirelessURLLCEnv, WirelessConfig, ChannelModel


@dataclass
class AggressiveConfig:
    """Aggressive parameters for fading channels"""
    # Queue thresholds (more conservative)
    q_low_base: int = 4       # Lower (was 6)
    q_crit_base: int = 10     # Lower (was 15)
    theta_up: float = 0.60    # Higher (was 0.50)
    theta_down: float = 0.35  # Higher (was 0.25)
    counter_thresh: int = 3   # Higher (was 2)
    
    # Channel-adaptive (more aggressive)
    cqi_good: int = 12        # Higher (was 10) - harder to enter BASE
    cqi_bad: int = 9          # Higher (was 7) - earlier BOOST trigger
    cqi_critical: int = 6     # NEW: stay in BOOST no matter what
    risk_alpha: float = 8.0   # Higher (was 5.0)
    slack_beta: float = 0.2   # Higher (was 0.1)
    mu_ref: float = 6.0
    
    # EWMA
    cqi_ewma_alpha: float = 0.3  # Faster response (was 0.2)
    mu_ewma_alpha: float = 0.3


class AggressiveCQISENTRY:
    """More aggressive CQI-Adaptive controller"""
    
    def __init__(self, config: AggressiveConfig = None):
        self.config = config or AggressiveConfig()
        self.reset()
    
    def reset(self):
        self.counter = 0
        self.current_mode = 1
        self.cqi_ewma = 10.0
        self.mu_ewma = self.config.mu_ref
        self.risk = 0.0
        self.consecutive_bad_cqi = 0
    
    def _update_estimates(self, cqi: int, served: int):
        cfg = self.config
        self.cqi_ewma = cfg.cqi_ewma_alpha * cqi + (1 - cfg.cqi_ewma_alpha) * self.cqi_ewma
        self.mu_ewma = cfg.mu_ewma_alpha * served + (1 - cfg.mu_ewma_alpha) * self.mu_ewma
        
        cqi_risk = max(0, cfg.cqi_good - self.cqi_ewma) / cfg.cqi_good
        mu_risk = max(0, cfg.mu_ref - self.mu_ewma) / cfg.mu_ref
        self.risk = 0.6 * cqi_risk + 0.4 * mu_risk  # Weight CQI more
        
        # Track consecutive bad CQI
        if cqi < cfg.cqi_bad:
            self.consecutive_bad_cqi += 1
        else:
            self.consecutive_bad_cqi = 0
    
    def _get_dynamic_q_crit(self) -> int:
        cfg = self.config
        q_crit = cfg.q_crit_base - cfg.risk_alpha * self.risk
        return max(3, int(q_crit))  # Can go as low as 3
    
    def _get_adjusted_slack(self, slack: float) -> float:
        cfg = self.config
        mu_penalty = cfg.slack_beta * max(0, cfg.mu_ref - self.mu_ewma)
        return max(0, slack - mu_penalty)
    
    def decide(self, Q: int, slack: float, cqi: int, served: int = 0) -> int:
        cfg = self.config
        self._update_estimates(cqi, served)
        
        q_crit = self._get_dynamic_q_crit()
        slack_adj = self._get_adjusted_slack(slack)
        
        # Priority 1: Critical channel condition → BOOST always
        if cqi < cfg.cqi_critical or self.consecutive_bad_cqi >= 3:
            self.counter = 0
            self.current_mode = 1
            return 1
        
        # Priority 2: Queue emergency
        if Q >= q_crit:
            self.counter = 0
            self.current_mode = 1
            return 1
        
        # Priority 3: Bad channel
        if cqi < cfg.cqi_bad:
            self.counter = 0
            self.current_mode = 1
            return 1
        
        # Priority 4: Low slack
        if slack_adj < cfg.theta_down:
            self.counter = 0
            self.current_mode = 1
            return 1
        
        # Priority 5: High risk
        if self.risk > 0.5:
            self.counter = 0
            self.current_mode = 1
            return 1
        
        # Safe to consider BASE
        if Q < cfg.q_low_base and slack_adj > cfg.theta_up and cqi >= cfg.cqi_good:
            self.counter += 1
            if self.counter >= cfg.counter_thresh:
                self.current_mode = 0
                return 0
            return self.current_mode
        
        self.counter = 0
        return self.current_mode


def parameter_sweep(channel_model: ChannelModel, n_seeds: int = 5) -> Dict:
    """Sweep key parameters to find optimal"""
    
    config = WirelessConfig(channel_model=channel_model, snr_db=15.0, doppler_hz=10.0)
    
    # Parameter combinations to try
    param_sets = [
        {'name': 'Original', 'cqi_bad': 7, 'q_crit': 15, 'cqi_critical': 4},
        {'name': 'Moderate', 'cqi_bad': 8, 'q_crit': 12, 'cqi_critical': 5},
        {'name': 'Aggressive', 'cqi_bad': 9, 'q_crit': 10, 'cqi_critical': 6},
        {'name': 'Very Aggressive', 'cqi_bad': 10, 'q_crit': 8, 'cqi_critical': 7},
        {'name': 'Ultra', 'cqi_bad': 11, 'q_crit': 6, 'cqi_critical': 8},
    ]
    
    results = []
    
    for params in param_sets:
        miss_list = []
        save_list = []
        sw_list = []
        
        for seed in range(n_seeds):
            env = WirelessURLLCEnv(config, seed=seed)
            env.reset(seed=seed)
            
            ctrl_config = AggressiveConfig()
            ctrl_config.cqi_bad = params['cqi_bad']
            ctrl_config.q_crit_base = params['q_crit']
            ctrl_config.cqi_critical = params['cqi_critical']
            
            controller = AggressiveCQISENTRY(ctrl_config)
            controller.reset()
            
            for t in range(20000):
                Q = len(env.queue)
                slack = env.ewma_slack
                cqi = env.channel.current_cqi
                action = controller.decide(Q, slack, cqi, 0)
                env.step(action)
            
            metrics = env.get_metrics()
            miss_list.append(metrics['miss_probability'] * 100)
            save_list.append(metrics['energy_savings'])
            sw_list.append(metrics['switches_per_ktti'])
        
        results.append({
            'name': params['name'],
            'cqi_bad': params['cqi_bad'],
            'q_crit': params['q_crit'],
            'miss': np.mean(miss_list),
            'miss_std': np.std(miss_list),
            'savings': np.mean(save_list),
            'switches': np.mean(sw_list),
            'urllc': np.mean(miss_list) < 1.0,
            'deploy': np.mean(sw_list) < 50
        })
    
    return results


if __name__ == '__main__':
    print("="*70)
    print("PARAMETER SWEEP FOR URLLC COMPLIANCE UNDER FADING")
    print("="*70)
    
    for channel in [ChannelModel.RAYLEIGH, ChannelModel.RICIAN]:
        print(f"\n[Channel: {channel.value.upper()}]")
        print("-"*70)
        
        results = parameter_sweep(channel, n_seeds=10)
        
        print(f"{'Config':<18} | {'CQI_bad':>7} | {'Q_crit':>6} | {'Miss%':>8} | {'Save%':>6} | {'Sw':>5} | URLLC | Deploy")
        print("-"*70)
        
        for r in results:
            urllc = "✓" if r['urllc'] else "✗"
            deploy = "✓" if r['deploy'] else "✗"
            print(f"{r['name']:<18} | {r['cqi_bad']:>7} | {r['q_crit']:>6} | {r['miss']:>6.3f}% | {r['savings']:>5.1f}% | {r['switches']:>5.1f} | {urllc:>5} | {deploy:>6}")
        
        # Find best config
        urllc_compliant = [r for r in results if r['urllc']]
        if urllc_compliant:
            best = max(urllc_compliant, key=lambda x: x['savings'])
            print(f"\n  ✓ BEST CONFIG: {best['name']} - Miss={best['miss']:.3f}%, Savings={best['savings']:.1f}%")
        else:
            best = min(results, key=lambda x: x['miss'])
            print(f"\n  ✗ No URLLC compliance achieved. Closest: {best['name']} - Miss={best['miss']:.3f}%")
    
    print("\n" + "="*70)
