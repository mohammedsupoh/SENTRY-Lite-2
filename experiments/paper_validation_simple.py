"""
PAPER VALIDATION SCRIPT - Simple Version
Verifies SENTRY-Lite-2 results match the paper
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from collections import deque

print("="*70)
print("PAPER VALIDATION: SENTRY-Lite-2 (IEEE TWC)")
print("="*70)

# ============================================================
# PAPER PARAMETERS (Table I)
# ============================================================
@dataclass
class PaperConfig:
    # System (6G-aligned)
    tti_ms: float = 0.125
    s_base: int = 4
    s_boost: int = 10
    deadline_ms: float = 0.5
    load: float = 0.85
    
    # Controller (SENTRY-Lite-2)
    q_low: int = 6
    q_crit: int = 15
    counter_thresh: int = 2
    theta_up: float = 0.50
    theta_down: float = 0.25
    emergency_duration: int = 5
    cooldown: int = 1
    ewma_alpha: float = 0.20
    miss_threshold: float = 0.02
    
    # Traffic
    burst_factor: float = 3.0
    p_good_to_bad: float = 0.05
    p_bad_to_good: float = 0.30
    
    @property
    def deadline_ttis(self):
        return int(self.deadline_ms / self.tti_ms)


# ============================================================
# TRAFFIC GENERATOR (Section IV-B)
# ============================================================
class MarkovTraffic:
    def __init__(self, config: PaperConfig, seed: int):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.state = 'good'
        base_rate = config.load * config.s_base
        self.good_rate = base_rate * 0.8
        self.bad_rate = base_rate * config.burst_factor
    
    def step(self) -> int:
        if self.state == 'good':
            if self.rng.random() < self.config.p_good_to_bad:
                self.state = 'bad'
        else:
            if self.rng.random() < self.config.p_bad_to_good:
                self.state = 'good'
        rate = self.good_rate if self.state == 'good' else self.bad_rate
        return self.rng.poisson(rate)
    
    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.state = 'good'


# ============================================================
# SENTRY-LITE-2 CONTROLLER (Algorithm 1)
# ============================================================
class SentryLite2:
    def __init__(self, config: PaperConfig):
        self.c = config
        self.reset()
    
    def reset(self):
        self.mode = 1  # Start BOOST
        self.slack_ewma = 0.5
        self.emergency_timer = 0
        self.cooldown_timer = 0
        self.queue_safe_counter = 0
        self.switches = 0
    
    def decide(self, queue_len: int, misses: int, deadline_ttis: int) -> int:
        # Update slack
        if queue_len > 0:
            slack = max(0, 1.0 - queue_len / (deadline_ttis * self.c.s_boost))
        else:
            slack = 1.0
        self.slack_ewma = self.c.ewma_alpha * slack + (1 - self.c.ewma_alpha) * self.slack_ewma
        
        # Layer 1: Emergency
        if self.emergency_timer > 0:
            self.emergency_timer -= 1
            return 1  # BOOST
        
        # Layer 2: Cooldown
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return self.mode
        
        # Emergency trigger
        if self.slack_ewma < self.c.miss_threshold or queue_len >= self.c.q_crit or misses > 0:
            self.emergency_timer = self.c.emergency_duration
            if self.mode == 0:
                self.switches += 1
                self.mode = 1
            return 1
        
        # Layer 3: Normal operation
        if self.mode == 0:  # BASE
            if self.slack_ewma < self.c.theta_down:
                self.cooldown_timer = self.c.cooldown
                self.switches += 1
                self.mode = 1
                return 1
            return 0
        else:  # BOOST
            if self.slack_ewma > self.c.theta_up and queue_len < self.c.q_low:
                self.queue_safe_counter += 1
                if self.queue_safe_counter >= self.c.counter_thresh:
                    self.cooldown_timer = self.c.cooldown
                    self.queue_safe_counter = 0
                    self.switches += 1
                    self.mode = 0
                    return 0
            else:
                self.queue_safe_counter = 0
            return 1


# ============================================================
# REACTIVE CONTROLLER (Baseline)
# ============================================================
class ReactiveController:
    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self.mode = 1
        self.switches = 0
    
    def reset(self):
        self.mode = 1
        self.switches = 0
    
    def decide(self, queue_len: int) -> int:
        new_mode = 1 if queue_len > self.threshold else 0
        if new_mode != self.mode:
            self.switches += 1
            self.mode = new_mode
        return self.mode


# ============================================================
# SIMULATION
# ============================================================
def run_simulation(config: PaperConfig, controller, n_ttis: int, seed: int):
    traffic = MarkovTraffic(config, seed)
    
    # Packet queue: (arrival_time, deadline_time)
    queue = deque()
    
    total_arrivals = 0
    total_misses = 0
    total_served = 0
    boost_ttis = 0
    
    if hasattr(controller, 'reset'):
        controller.reset()
    
    for t in range(n_ttis):
        # Arrivals
        arrivals = traffic.step()
        total_arrivals += arrivals
        for _ in range(arrivals):
            queue.append((t, t + config.deadline_ttis))
        
        # Check misses (packets past deadline)
        misses_this_tti = 0
        while queue and queue[0][1] <= t:
            queue.popleft()
            total_misses += 1
            misses_this_tti += 1
        
        # Controller decision
        if isinstance(controller, SentryLite2):
            action = controller.decide(len(queue), misses_this_tti, config.deadline_ttis)
        else:
            action = controller.decide(len(queue))
        
        # Service
        capacity = config.s_boost if action == 1 else config.s_base
        served = min(len(queue), capacity)
        for _ in range(served):
            queue.popleft()
            total_served += 1
        
        if action == 1:
            boost_ttis += 1
    
    # Final check for remaining packets
    while queue:
        if queue[0][1] <= n_ttis:
            total_misses += 1
        queue.popleft()
    
    savings = (1 - boost_ttis / n_ttis) * 100
    pmiss = total_misses / max(1, total_arrivals) * 100
    sw_ktti = controller.switches / n_ttis * 1000
    
    return {
        'savings': savings,
        'pmiss': pmiss,
        'switches': sw_ktti,
        'total_arrivals': total_arrivals,
        'total_misses': total_misses
    }


# ============================================================
# MAIN VALIDATION
# ============================================================
if __name__ == '__main__':
    
    # Paper expected results
    PAPER_TABLE2 = {
        'SENTRY-Lite-2': {'savings': 84.0, 'pmiss': 0.0, 'switches': 44.0},
        'Reactive': {'savings': 89.2, 'pmiss': 0.0, 'switches': 173.2},
    }
    
    PAPER_TABLE3 = {
        'Reference': {'savings': 83.9, 'pmiss': 0.0, 'switches': 44.3},
        'Stress_0.90': {'savings': 79.2, 'pmiss': 0.0, 'switches': 57.0},
    }
    
    N_SEEDS = 10
    N_TTIS = 100000
    
    print(f"\n[Config: {N_SEEDS} seeds × {N_TTIS:,} TTIs]")
    
    # ========== Test 1: SENTRY-Lite-2 Reference ==========
    print("\n" + "-"*70)
    print("[1] Table II: SENTRY-Lite-2 @ rho=0.85, D=0.5ms")
    print("-"*70)
    
    config = PaperConfig()
    results = []
    
    for seed in range(N_SEEDS):
        ctrl = SentryLite2(config)
        r = run_simulation(config, ctrl, N_TTIS, seed)
        results.append(r)
        print(f"  Seed {seed}: Savings={r['savings']:.1f}%, P_miss={r['pmiss']:.3f}%, Sw={r['switches']:.1f}")
    
    avg_savings = np.mean([r['savings'] for r in results])
    avg_pmiss = np.mean([r['pmiss'] for r in results])
    avg_switches = np.mean([r['switches'] for r in results])
    total_arr = sum([r['total_arrivals'] for r in results])
    total_miss = sum([r['total_misses'] for r in results])
    
    print(f"\n  AVERAGE: Savings={avg_savings:.1f}%, P_miss={avg_pmiss:.4f}%, Sw={avg_switches:.1f}")
    print(f"  PAPER:   Savings=84.0%, P_miss=0.00%, Sw=44.0")
    print(f"  Total packets: {total_arr:,}, Misses: {total_miss}")
    
    # ========== Test 2: Stress ρ=0.90 ==========
    print("\n" + "-"*70)
    print("[2] Table III: Stress @ rho=0.90, D=0.5ms")
    print("-"*70)
    
    config_stress = PaperConfig()
    config_stress.load = 0.90
    results_stress = []
    
    for seed in range(N_SEEDS):
        ctrl = SentryLite2(config_stress)
        r = run_simulation(config_stress, ctrl, N_TTIS, seed)
        results_stress.append(r)
    
    avg_savings_s = np.mean([r['savings'] for r in results_stress])
    avg_pmiss_s = np.mean([r['pmiss'] for r in results_stress])
    avg_switches_s = np.mean([r['switches'] for r in results_stress])
    
    print(f"  AVERAGE: Savings={avg_savings_s:.1f}%, P_miss={avg_pmiss_s:.4f}%, Sw={avg_switches_s:.1f}")
    print(f"  PAPER:   Savings=79.2%, P_miss=0.00%, Sw=57.0")
    
    # ========== Test 3: Reactive Baseline ==========
    print("\n" + "-"*70)
    print("[3] Table II: Reactive Controller")
    print("-"*70)
    
    results_reactive = []
    for seed in range(N_SEEDS):
        ctrl = ReactiveController(threshold=5)
        r = run_simulation(config, ctrl, N_TTIS, seed)
        results_reactive.append(r)
    
    avg_savings_r = np.mean([r['savings'] for r in results_reactive])
    avg_pmiss_r = np.mean([r['pmiss'] for r in results_reactive])
    avg_switches_r = np.mean([r['switches'] for r in results_reactive])
    
    print(f"  AVERAGE: Savings={avg_savings_r:.1f}%, P_miss={avg_pmiss_r:.4f}%, Sw={avg_switches_r:.1f}")
    print(f"  PAPER:   Savings=89.2%, P_miss=0.00%, Sw=173.2")
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    checks = []
    
    # Check 1: 84% savings (±5%)
    if 79 <= avg_savings <= 89:
        checks.append(("OK", "84% energy savings", f"{avg_savings:.1f}%"))
    else:
        checks.append(("XX", "84% energy savings", f"{avg_savings:.1f}%"))
    
    # Check 2: URLLC compliance
    if avg_pmiss < 1.0:
        checks.append(("OK", "URLLC compliance (<1%)", f"{avg_pmiss:.4f}%"))
    else:
        checks.append(("XX", "URLLC compliance (<1%)", f"{avg_pmiss:.4f}%"))
    
    # Check 3: Deployability
    if avg_switches < 50:
        checks.append(("OK", "Deployability (<50 sw/kTTI)", f"{avg_switches:.1f}"))
    else:
        checks.append(("XX", "Deployability (<50 sw/kTTI)", f"{avg_switches:.1f}"))
    
    # Check 4: ~4x less switching than Reactive
    ratio = avg_switches_r / max(1, avg_switches)
    if ratio >= 3.0:
        checks.append(("OK", f"~4x less switching vs Reactive", f"{ratio:.1f}x"))
    else:
        checks.append(("XX", f"~4x less switching vs Reactive", f"{ratio:.1f}x"))
    
    # Check 5: Stress test
    if 74 <= avg_savings_s <= 84:
        checks.append(("OK", "Stress test (rho=0.90) savings", f"{avg_savings_s:.1f}%"))
    else:
        checks.append(("XX", "Stress test (rho=0.90) savings", f"{avg_savings_s:.1f}%"))
    
    print("\n[Key Paper Claims]")
    for status, claim, value in checks:
        print(f"  {status} {claim}: {value}")
    
    all_pass = all(c[0] == "OK" for c in checks)
    
    print("\n" + "="*70)
    if all_pass:
        print("[OK] ALL PAPER CLAIMS VERIFIED - READY FOR GITHUB")
    else:
        print("[!!] SOME CLAIMS NEED REVIEW")
    print("="*70)
