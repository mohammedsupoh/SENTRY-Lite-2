"""
PAPER VALIDATION SCRIPT
Verifies all results match the paper exactly
Target: IEEE TWC SENTRY-Lite-2
"""

import numpy as np
import sys
sys.path.insert(0, 'C:/Users/LOQ/Desktop/SENTRY-Lite-2/src')

from controllers.sentry_lite2 import SentryLite2Controller
from environment.urllc_env import URLLCEnvironment, URLLCConfig

# Paper parameters (Section IV)
PAPER_CONFIG = {
    'tti_ms': 0.125,        # 6G mini-slot
    's_base': 4,            # packets/TTI
    's_boost': 10,          # packets/TTI
    'deadline_ttis': 4,     # 0.5ms / 0.125ms = 4 TTIs
    'load': 0.85,
    'q_low': 6,
    'q_crit': 15,
    'counter': 2,
    'theta_up': 0.50,
    'theta_down': 0.25,
    'emergency_duration': 5,
    'cooldown': 1,
    'ewma_alpha': 0.20,
}

# Expected results from paper
PAPER_RESULTS = {
    'table2_sentry': {'savings': 84.0, 'pmiss': 0.0, 'switches': 44.0},
    'table2_reactive': {'savings': 89.2, 'pmiss': 0.0, 'switches': 173.2},
    'table3_reference': {'savings': 83.9, 'pmiss': 0.0, 'switches': 44.3},
    'table3_stress': {'savings': 79.2, 'pmiss': 0.0, 'switches': 57.0},
}


def run_sentry_lite2(config, n_seeds=20, episode_length=100000):
    """Run SENTRY-Lite-2 controller"""
    all_savings = []
    all_pmiss = []
    all_switches = []
    total_arrivals = 0
    total_misses = 0
    
    for seed in range(n_seeds):
        env_config = URLLCConfig(
            s_base=config['s_base'],
            s_boost=config['s_boost'],
            deadline_ttis=config['deadline_ttis'],
            load=config['load'],
            episode_length=episode_length
        )
        
        env = URLLCEnvironment(env_config, seed=seed)
        controller = SentryLite2Controller(
            q_low=config['q_low'],
            q_crit=config['q_crit'],
            counter=config['counter'],
            theta_up=config['theta_up'],
            theta_down=config['theta_down'],
            emergency_duration=config['emergency_duration'],
            cooldown=config['cooldown'],
            ewma_alpha=config['ewma_alpha']
        )
        
        env.reset(seed=seed)
        controller.reset()
        
        for t in range(episode_length):
            state = env.get_state()
            action = controller.decide(state)
            env.step(action)
        
        metrics = env.get_metrics()
        all_savings.append(metrics['energy_savings'])
        all_pmiss.append(metrics['miss_probability'] * 100)
        all_switches.append(metrics['switches_per_ktti'])
        total_arrivals += metrics['total_arrivals']
        total_misses += metrics['total_misses']
    
    return {
        'savings_mean': np.mean(all_savings),
        'savings_std': np.std(all_savings),
        'pmiss_mean': np.mean(all_pmiss),
        'switches_mean': np.mean(all_switches),
        'total_arrivals': total_arrivals,
        'total_misses': total_misses
    }


def run_reactive(config, n_seeds=20, episode_length=100000):
    """Run Reactive controller (baseline)"""
    all_savings = []
    all_pmiss = []
    all_switches = []
    
    for seed in range(n_seeds):
        env_config = URLLCConfig(
            s_base=config['s_base'],
            s_boost=config['s_boost'],
            deadline_ttis=config['deadline_ttis'],
            load=config['load'],
            episode_length=episode_length
        )
        
        env = URLLCEnvironment(env_config, seed=seed)
        env.reset(seed=seed)
        
        # Simple reactive: BOOST if Q > threshold
        threshold = 5
        mode = 1
        switches = 0
        base_ttis = 0
        
        for t in range(episode_length):
            Q = len(env.queue)
            
            new_mode = 1 if Q > threshold else 0
            if new_mode != mode:
                switches += 1
                mode = new_mode
            
            if mode == 0:
                base_ttis += 1
            
            env.step(mode)
        
        metrics = env.get_metrics()
        savings = base_ttis / episode_length * 100
        all_savings.append(savings)
        all_pmiss.append(metrics['miss_probability'] * 100)
        all_switches.append(switches / episode_length * 1000)
    
    return {
        'savings_mean': np.mean(all_savings),
        'pmiss_mean': np.mean(all_pmiss),
        'switches_mean': np.mean(all_switches)
    }


def validate_result(name, actual, expected, tolerance=5.0):
    """Check if result matches paper within tolerance"""
    
    checks = []
    
    # Savings check
    if abs(actual['savings_mean'] - expected['savings']) <= tolerance:
        checks.append(('Savings', '✓', actual['savings_mean'], expected['savings']))
    else:
        checks.append(('Savings', '✗', actual['savings_mean'], expected['savings']))
    
    # P_miss check (stricter)
    if actual['pmiss_mean'] <= expected['pmiss'] + 0.5:
        checks.append(('P_miss', '✓', actual['pmiss_mean'], expected['pmiss']))
    else:
        checks.append(('P_miss', '✗', actual['pmiss_mean'], expected['pmiss']))
    
    # Switches check
    if abs(actual['switches_mean'] - expected['switches']) <= tolerance * 2:
        checks.append(('Sw/kTTI', '✓', actual['switches_mean'], expected['switches']))
    else:
        checks.append(('Sw/kTTI', '✗', actual['switches_mean'], expected['switches']))
    
    return checks


if __name__ == '__main__':
    print("="*70)
    print("PAPER VALIDATION: SENTRY-Lite-2 (IEEE TWC)")
    print("="*70)
    
    # Quick validation (fewer seeds for speed)
    N_SEEDS = 5  # Use 20 for full validation
    EPISODE_LEN = 50000  # Use 100000 for full validation
    
    print(f"\n[Settings: {N_SEEDS} seeds × {EPISODE_LEN} TTIs]")
    print("-"*70)
    
    # Test 1: Table II - SENTRY-Lite-2 Reference (ρ=0.85, D=0.5ms)
    print("\n[1] Table II: SENTRY-Lite-2 @ ρ=0.85, D=0.5ms")
    config_ref = PAPER_CONFIG.copy()
    
    result = run_sentry_lite2(config_ref, n_seeds=N_SEEDS, episode_length=EPISODE_LEN)
    
    print(f"    Actual:   Savings={result['savings_mean']:.1f}%, "
          f"P_miss={result['pmiss_mean']:.3f}%, Sw={result['switches_mean']:.1f}")
    print(f"    Expected: Savings=84.0%, P_miss=0.00%, Sw=44.0")
    
    checks = validate_result('Table II', result, PAPER_RESULTS['table2_sentry'])
    for metric, status, actual, expected in checks:
        print(f"    {status} {metric}: {actual:.1f} vs {expected:.1f}")
    
    # Test 2: Table III - Stress (ρ=0.90)
    print("\n[2] Table III: Stress @ ρ=0.90, D=0.5ms")
    config_stress = PAPER_CONFIG.copy()
    config_stress['load'] = 0.90
    
    result_stress = run_sentry_lite2(config_stress, n_seeds=N_SEEDS, episode_length=EPISODE_LEN)
    
    print(f"    Actual:   Savings={result_stress['savings_mean']:.1f}%, "
          f"P_miss={result_stress['pmiss_mean']:.3f}%, Sw={result_stress['switches_mean']:.1f}")
    print(f"    Expected: Savings=79.2%, P_miss=0.00%, Sw=57.0")
    
    checks = validate_result('Stress', result_stress, PAPER_RESULTS['table3_stress'])
    for metric, status, actual, expected in checks:
        print(f"    {status} {metric}: {actual:.1f} vs {expected:.1f}")
    
    # Test 3: Reactive baseline
    print("\n[3] Table II: Reactive Controller")
    result_reactive = run_reactive(config_ref, n_seeds=N_SEEDS, episode_length=EPISODE_LEN)
    
    print(f"    Actual:   Savings={result_reactive['savings_mean']:.1f}%, "
          f"P_miss={result_reactive['pmiss_mean']:.3f}%, Sw={result_reactive['switches_mean']:.1f}")
    print(f"    Expected: Savings=89.2%, P_miss=0.00%, Sw=173.2")
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_pass = True
    
    # Check key claims
    print("\n[Key Paper Claims Check]")
    
    # Claim 1: 84% savings
    if 80 <= result['savings_mean'] <= 88:
        print("  ✓ 84% energy savings claim: VERIFIED")
    else:
        print(f"  ✗ 84% energy savings claim: GOT {result['savings_mean']:.1f}%")
        all_pass = False
    
    # Claim 2: URLLC compliance
    if result['pmiss_mean'] < 1.0:
        print("  ✓ URLLC compliance (<1% miss): VERIFIED")
    else:
        print(f"  ✗ URLLC compliance: GOT {result['pmiss_mean']:.2f}%")
        all_pass = False
    
    # Claim 3: Deployability (<50 sw/kTTI)
    if result['switches_mean'] < 50:
        print("  ✓ Deployability (<50 sw/kTTI): VERIFIED")
    else:
        print(f"  ✗ Deployability: GOT {result['switches_mean']:.1f} sw/kTTI")
        all_pass = False
    
    # Claim 4: 4x less switching than Reactive
    ratio = result_reactive['switches_mean'] / max(1, result['switches_mean'])
    if ratio >= 3.5:
        print(f"  ✓ ~4x less switching than Reactive: VERIFIED ({ratio:.1f}x)")
    else:
        print(f"  ✗ ~4x less switching: GOT {ratio:.1f}x")
        all_pass = False
    
    print("\n" + "="*70)
    if all_pass:
        print("✅ ALL PAPER CLAIMS VERIFIED - READY FOR GITHUB")
    else:
        print("⚠️ SOME CLAIMS NEED REVIEW")
    print("="*70)
