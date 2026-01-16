"""
PAPER VALIDATION v2 - Exact Paper Implementation
"""

import numpy as np
from collections import deque

print("="*70)
print("PAPER VALIDATION v2: SENTRY-Lite-2 (IEEE TWC)")
print("="*70)

# Paper Table I parameters
CONFIG = {
    'tti_ms': 0.125,
    's_base': 4,
    's_boost': 10,
    'deadline_ttis': 4,  # 0.5ms / 0.125ms
    'load': 0.85,
    
    # Controller
    'q_low': 6,
    'q_crit': 15,
    'counter': 2,
    'theta_up': 0.50,
    'theta_down': 0.25,
    'emergency_duration': 5,
    'cooldown': 1,
    'ewma_alpha': 0.20,
    'miss_threshold': 0.02,
    
    # Traffic
    'burst_factor': 3.0,
    'p_good_bad': 0.05,
    'p_bad_good': 0.30,
}


class SentryLite2_v2:
    """Exact Algorithm 1 from paper"""
    
    def __init__(self, cfg):
        self.c = cfg
        self.reset()
    
    def reset(self):
        self.mode = 1  # BOOST
        self.slack_ewma = 0.5
        self.emergency_timer = 0
        self.cooldown_timer = 0
        self.queue_safe_counter = 0
        self.switches = 0
    
    def compute_slack(self, queue_len, avg_delay, deadline):
        """Slack = (D - avg_delay) / D"""
        if queue_len == 0:
            return 1.0
        slack = max(0.0, min(1.0, (deadline - avg_delay) / deadline))
        return slack
    
    def decide(self, queue_len, avg_delay, deadline, had_miss):
        # Compute instantaneous slack
        s_t = self.compute_slack(queue_len, avg_delay, deadline)
        
        # EWMA update
        self.slack_ewma = self.c['ewma_alpha'] * s_t + (1 - self.c['ewma_alpha']) * self.slack_ewma
        
        # Layer 1: Emergency latch
        if self.emergency_timer > 0:
            self.emergency_timer -= 1
            return 1
        
        # Layer 2: Cooldown
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return self.mode
        
        # Emergency trigger
        if self.slack_ewma < self.c['miss_threshold'] or queue_len >= self.c['q_crit'] or had_miss:
            self.emergency_timer = self.c['emergency_duration']
            if self.mode == 0:
                self.switches += 1
                self.mode = 1
            return 1
        
        # Layer 3: Normal
        if self.mode == 0:  # BASE
            if self.slack_ewma < self.c['theta_down']:
                self.cooldown_timer = self.c['cooldown']
                self.switches += 1
                self.mode = 1
                return 1
            return 0
        else:  # BOOST
            if self.slack_ewma > self.c['theta_up'] and queue_len < self.c['q_low']:
                self.queue_safe_counter += 1
                if self.queue_safe_counter >= self.c['counter']:
                    self.cooldown_timer = self.c['cooldown']
                    self.queue_safe_counter = 0
                    self.switches += 1
                    self.mode = 0
                    return 0
            else:
                self.queue_safe_counter = 0
            return 1


def run_experiment(cfg, n_ttis, seed):
    rng = np.random.default_rng(seed)
    
    # Traffic generator
    traffic_state = 'good'
    base_rate = cfg['load'] * cfg['s_base']
    good_rate = base_rate * 0.8
    bad_rate = base_rate * cfg['burst_factor']
    
    # Controller
    ctrl = SentryLite2_v2(cfg)
    
    # Queue: list of (arrival_time, deadline_time)
    queue = deque()
    
    # Metrics
    total_arrivals = 0
    total_misses = 0
    boost_ttis = 0
    deadline = cfg['deadline_ttis']
    
    for t in range(n_ttis):
        # Traffic transition
        if traffic_state == 'good':
            if rng.random() < cfg['p_good_bad']:
                traffic_state = 'bad'
        else:
            if rng.random() < cfg['p_bad_good']:
                traffic_state = 'good'
        
        # Arrivals
        rate = good_rate if traffic_state == 'good' else bad_rate
        arrivals = rng.poisson(rate)
        total_arrivals += arrivals
        for _ in range(arrivals):
            queue.append((t, t + deadline))
        
        # Check deadline violations
        had_miss = False
        while queue and queue[0][1] <= t:
            queue.popleft()
            total_misses += 1
            had_miss = True
        
        # Compute average delay
        if queue:
            avg_delay = sum(t - pkt[0] for pkt in queue) / len(queue)
        else:
            avg_delay = 0
        
        # Controller decision
        action = ctrl.decide(len(queue), avg_delay, deadline, had_miss)
        
        # Service
        capacity = cfg['s_boost'] if action == 1 else cfg['s_base']
        served = min(len(queue), capacity)
        for _ in range(served):
            queue.popleft()
        
        if action == 1:
            boost_ttis += 1
    
    # Drain remaining
    while queue:
        if queue[0][1] <= n_ttis:
            total_misses += 1
        queue.popleft()
    
    return {
        'savings': (1 - boost_ttis / n_ttis) * 100,
        'pmiss': total_misses / max(1, total_arrivals) * 100,
        'switches': ctrl.switches / n_ttis * 1000,
        'arrivals': total_arrivals,
        'misses': total_misses
    }


if __name__ == '__main__':
    N_SEEDS = 20
    N_TTIS = 100000
    
    print(f"\n[Config: {N_SEEDS} seeds x {N_TTIS:,} TTIs]")
    
    # Reference test
    print("\n" + "-"*70)
    print("[1] SENTRY-Lite-2 @ rho=0.85, D=0.5ms")
    print("-"*70)
    
    results = []
    for seed in range(N_SEEDS):
        r = run_experiment(CONFIG, N_TTIS, seed)
        results.append(r)
        print(f"  Seed {seed:2d}: Savings={r['savings']:.1f}%, P_miss={r['pmiss']:.4f}%, Sw={r['switches']:.1f}")
    
    avg = {
        'savings': np.mean([r['savings'] for r in results]),
        'pmiss': np.mean([r['pmiss'] for r in results]),
        'switches': np.mean([r['switches'] for r in results])
    }
    std_savings = np.std([r['savings'] for r in results])
    total_arr = sum(r['arrivals'] for r in results)
    total_miss = sum(r['misses'] for r in results)
    
    print(f"\n  MEAN: Savings={avg['savings']:.1f}% (+/-{std_savings:.1f}), P_miss={avg['pmiss']:.4f}%, Sw={avg['switches']:.1f}")
    print(f"  PAPER: Savings=84.0%, P_miss=0.00%, Sw=44.0")
    print(f"  Total: {total_arr:,} packets, {total_miss} misses")
    
    # Clopper-Pearson CI
    from scipy import stats
    if total_miss == 0:
        ci_upper = 3.0 / total_arr * 100  # Rule of Three
    else:
        ci_upper = stats.beta.ppf(0.95, total_miss + 1, total_arr - total_miss) * 100
    print(f"  95% CI upper bound: {ci_upper:.6f}%")
    
    # Stress test
    print("\n" + "-"*70)
    print("[2] Stress @ rho=0.90")
    print("-"*70)
    
    cfg_stress = CONFIG.copy()
    cfg_stress['load'] = 0.90
    
    results_s = []
    for seed in range(N_SEEDS):
        r = run_experiment(cfg_stress, N_TTIS, seed)
        results_s.append(r)
    
    avg_s = {
        'savings': np.mean([r['savings'] for r in results_s]),
        'pmiss': np.mean([r['pmiss'] for r in results_s]),
        'switches': np.mean([r['switches'] for r in results_s])
    }
    print(f"  MEAN: Savings={avg_s['savings']:.1f}%, P_miss={avg_s['pmiss']:.4f}%, Sw={avg_s['switches']:.1f}")
    print(f"  PAPER: Savings=79.2%, P_miss=0.00%, Sw=57.0")
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    checks = []
    
    # 1. Savings range
    if 75 <= avg['savings'] <= 90:
        checks.append(("[OK]", "Energy savings in valid range", f"{avg['savings']:.1f}%"))
    else:
        checks.append(("[XX]", "Energy savings", f"{avg['savings']:.1f}%"))
    
    # 2. URLLC
    if avg['pmiss'] < 1.0:
        checks.append(("[OK]", "URLLC compliance (<1%)", f"{avg['pmiss']:.4f}%"))
    else:
        checks.append(("[XX]", "URLLC compliance", f"{avg['pmiss']:.4f}%"))
    
    # 3. Deployability
    if avg['switches'] < 55:
        checks.append(("[OK]", "Deployability (sw/kTTI)", f"{avg['switches']:.1f}"))
    else:
        checks.append(("[XX]", "Deployability", f"{avg['switches']:.1f}"))
    
    # 4. Stress maintains URLLC
    if avg_s['pmiss'] < 1.0:
        checks.append(("[OK]", "Stress URLLC compliance", f"{avg_s['pmiss']:.4f}%"))
    else:
        checks.append(("[XX]", "Stress URLLC", f"{avg_s['pmiss']:.4f}%"))
    
    print("\n[Key Claims]")
    for status, claim, value in checks:
        print(f"  {status} {claim}: {value}")
    
    all_ok = all("[OK]" in c[0] for c in checks)
    
    print("\n" + "="*70)
    if all_ok:
        print("[PASS] Algorithm implementation verified - Ready for GitHub")
    else:
        print("[WARN] Some metrics differ from paper")
    print("="*70)
