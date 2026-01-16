"""
CQI-Adaptive SENTRY Controller
Addresses W1: Robust operation under fading channels

Key innovations:
1. Dynamic q_crit based on channel risk
2. Channel-aware slack adjustment
3. CQI-triggered early BOOST
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
import sys
sys.path.insert(0, 'C:/Users/LOQ/Desktop/SENTRY-Lite-2/src')
from environment.wireless_env import WirelessURLLCEnv, WirelessConfig, ChannelModel


@dataclass
class CQIAdaptiveConfig:
    """CQI-Adaptive SENTRY parameters"""
    # Original SENTRY-Lite-2 parameters
    q_low_base: int = 6
    q_crit_base: int = 15
    theta_up: float = 0.50
    theta_down: float = 0.25
    counter_thresh: int = 2
    
    # NEW: Channel-adaptive parameters
    cqi_good: int = 10      # CQI threshold for "good channel"
    cqi_bad: int = 7        # CQI threshold for "bad channel"
    risk_alpha: float = 5.0  # q_crit reduction per risk unit
    slack_beta: float = 0.1  # Slack penalty for channel degradation
    mu_ref: float = 6.0      # Reference service rate
    
    # EWMA smoothing for channel state
    cqi_ewma_alpha: float = 0.2
    mu_ewma_alpha: float = 0.2


class CQIAdaptiveSENTRY:
    """
    CQI-Adaptive SENTRY Controller
    
    Extends SENTRY-Lite-2 with channel awareness:
    1. q_crit(t) = q_crit_base - α × risk(t)
    2. slack'(t) = slack(t) - β × max(0, μ_ref - μ̂(t))
    3. Early BOOST trigger when CQI < cqi_bad
    """
    
    def __init__(self, config: CQIAdaptiveConfig = None):
        self.config = config or CQIAdaptiveConfig()
        self.reset()
    
    def reset(self):
        """Reset controller state"""
        self.counter = 0
        self.current_mode = 1  # Start in BOOST
        
        # Channel state estimates
        self.cqi_ewma = 10.0  # Start optimistic
        self.mu_ewma = self.config.mu_ref
        self.risk = 0.0
    
    def _update_channel_estimates(self, cqi: int, served: int):
        """Update channel state estimates"""
        cfg = self.config
        
        # Update CQI EWMA
        self.cqi_ewma = cfg.cqi_ewma_alpha * cqi + (1 - cfg.cqi_ewma_alpha) * self.cqi_ewma
        
        # Update service rate estimate
        self.mu_ewma = cfg.mu_ewma_alpha * served + (1 - cfg.mu_ewma_alpha) * self.mu_ewma
        
        # Compute risk indicator (higher = worse channel)
        # Risk based on CQI deviation from "good"
        cqi_risk = max(0, cfg.cqi_good - self.cqi_ewma) / cfg.cqi_good
        
        # Risk based on service rate degradation
        mu_risk = max(0, cfg.mu_ref - self.mu_ewma) / cfg.mu_ref
        
        # Combined risk (0 = perfect, 1 = very bad)
        self.risk = 0.5 * cqi_risk + 0.5 * mu_risk
    
    def _get_dynamic_q_crit(self) -> int:
        """Compute dynamic q_crit based on channel risk"""
        cfg = self.config
        
        # Lower q_crit when channel is bad (enter BOOST earlier)
        q_crit = cfg.q_crit_base - cfg.risk_alpha * self.risk
        
        # Clamp to reasonable range
        return max(5, int(q_crit))
    
    def _get_adjusted_slack(self, slack: float) -> float:
        """Adjust slack based on channel state"""
        cfg = self.config
        
        # Reduce slack when service rate is degraded
        mu_penalty = cfg.slack_beta * max(0, cfg.mu_ref - self.mu_ewma)
        
        return max(0, slack - mu_penalty)
    
    def decide(self, Q: int, slack: float, cqi: int, served: int = 0) -> int:
        """
        Make BASE/BOOST decision
        
        Args:
            Q: Current queue length
            slack: EWMA slack (from environment)
            cqi: Current CQI (1-15)
            served: Packets served in last TTI
        
        Returns:
            action: 0 (BASE) or 1 (BOOST)
        """
        cfg = self.config
        
        # Update channel estimates
        self._update_channel_estimates(cqi, served)
        
        # Get dynamic thresholds
        q_crit = self._get_dynamic_q_crit()
        slack_adj = self._get_adjusted_slack(slack)
        
        # Decision logic (CQI-Adaptive SENTRY)
        
        # Priority 1: Emergency conditions → BOOST immediately
        if Q >= q_crit:
            self.counter = 0
            self.current_mode = 1
            return 1
        
        # Priority 2: Bad channel → BOOST (NEW)
        if cqi < cfg.cqi_bad:
            self.counter = 0
            self.current_mode = 1
            return 1
        
        # Priority 3: Low adjusted slack → BOOST
        if slack_adj < cfg.theta_down:
            self.counter = 0
            self.current_mode = 1
            return 1
        
        # Priority 4: Safe conditions → transition to BASE
        if Q < cfg.q_low_base and slack_adj > cfg.theta_up and cqi >= cfg.cqi_good:
            self.counter += 1
            if self.counter >= cfg.counter_thresh:
                self.current_mode = 0
                return 0
            else:
                return self.current_mode
        
        # Default: hold current mode
        self.counter = 0
        return self.current_mode
    
    def get_state(self) -> Dict:
        """Get controller internal state for debugging"""
        return {
            'mode': self.current_mode,
            'counter': self.counter,
            'cqi_ewma': self.cqi_ewma,
            'mu_ewma': self.mu_ewma,
            'risk': self.risk,
            'q_crit_dynamic': self._get_dynamic_q_crit()
        }


def run_comparison_experiment(channel_model: ChannelModel, n_seeds: int = 10,
                               episode_length: int = 20000) -> Dict:
    """Compare Original vs CQI-Adaptive SENTRY"""
    
    config = WirelessConfig(
        channel_model=channel_model,
        snr_db=15.0,
        doppler_hz=10.0
    )
    
    results = {
        'original': {'miss': [], 'savings': [], 'switches': []},
        'adaptive': {'miss': [], 'savings': [], 'switches': []}
    }
    
    for seed in range(n_seeds):
        # === Original SENTRY-Lite-2 ===
        env = WirelessURLLCEnv(config, seed=seed)
        env.reset(seed=seed)
        
        counter = 0
        for t in range(episode_length):
            Q = len(env.queue)
            slack = env.ewma_slack
            
            # Original policy (no CQI awareness)
            if Q >= 15 or slack < 0.25:
                action = 1
                counter = 0
            elif Q < 6 and slack > 0.50:
                counter += 1
                action = 0 if counter >= 2 else env.current_mode
            else:
                action = env.current_mode
                counter = 0
            
            env.step(action)
        
        metrics = env.get_metrics()
        results['original']['miss'].append(metrics['miss_probability'] * 100)
        results['original']['savings'].append(metrics['energy_savings'])
        results['original']['switches'].append(metrics['switches_per_ktti'])
        
        # === CQI-Adaptive SENTRY ===
        env = WirelessURLLCEnv(config, seed=seed)
        env.reset(seed=seed)
        controller = CQIAdaptiveSENTRY()
        controller.reset()
        
        for t in range(episode_length):
            Q = len(env.queue)
            slack = env.ewma_slack
            cqi = env.channel.current_cqi
            
            # Get last served count (approximate)
            served = env.total_departures - (env.total_departures if t == 0 else 0)
            
            action = controller.decide(Q, slack, cqi, served)
            env.step(action)
        
        metrics = env.get_metrics()
        results['adaptive']['miss'].append(metrics['miss_probability'] * 100)
        results['adaptive']['savings'].append(metrics['energy_savings'])
        results['adaptive']['switches'].append(metrics['switches_per_ktti'])
    
    return results


#=============================================================================
# MAIN TEST
#=============================================================================

if __name__ == '__main__':
    print("="*70)
    print("CQI-ADAPTIVE SENTRY: COMPARISON WITH ORIGINAL")
    print("="*70)
    
    for channel in [ChannelModel.AWGN, ChannelModel.RAYLEIGH, ChannelModel.RICIAN]:
        print(f"\n[Channel: {channel.value}]")
        print("-"*50)
        
        results = run_comparison_experiment(channel, n_seeds=10, episode_length=20000)
        
        print(f"  {'Controller':<15} | {'Miss Prob':>12} | {'Savings':>10} | {'Sw/kTTI':>10}")
        print(f"  {'-'*15} | {'-'*12} | {'-'*10} | {'-'*10}")
        
        for name in ['original', 'adaptive']:
            miss = np.mean(results[name]['miss'])
            miss_std = np.std(results[name]['miss'])
            save = np.mean(results[name]['savings'])
            sw = np.mean(results[name]['switches'])
            
            urllc = "✓" if miss < 1.0 else "✗"
            deploy = "✓" if sw < 50 else "✗"
            
            print(f"  {name:<15} | {miss:>6.3f}% ±{miss_std:>4.2f} | {save:>8.1f}% | {sw:>8.1f} | URLLC:{urllc} Deploy:{deploy}")
    
    print("\n" + "="*70)
    print("KEY IMPROVEMENTS:")
    print("  - Dynamic q_crit based on channel risk")
    print("  - CQI-triggered early BOOST")
    print("  - Channel-aware slack adjustment")
    print("="*70)
