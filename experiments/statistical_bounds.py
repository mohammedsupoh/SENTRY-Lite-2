"""
W2 SOLUTION: Statistical Bounds for Zero-Miss Claims
Provides rigorous confidence intervals using Rule of Three and Bootstrap

Key outputs:
1. Upper bound on miss probability (even when observed = 0)
2. Bootstrap confidence intervals
3. Clopper-Pearson exact binomial bounds
"""

import numpy as np
from scipy import stats
from typing import Tuple, List, Dict
import pandas as pd

def rule_of_three_bound(n_trials: int, n_events: int = 0, 
                        confidence: float = 0.95) -> float:
    """
    Rule of Three for rare events
    
    When observing 0 events in n trials:
    Upper bound ≈ 3/n at 95% confidence
    
    This is a conservative approximation of the exact binomial bound.
    """
    if n_events > 0:
        # Use exact binomial for non-zero events
        return clopper_pearson_upper(n_events, n_trials, confidence)
    
    # Rule of three approximation
    alpha = 1 - confidence
    return -np.log(alpha) / n_trials


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """
    Clopper-Pearson exact binomial upper bound
    
    This is the gold standard for binomial proportion confidence intervals.
    """
    alpha = 1 - confidence
    if k == n:
        return 1.0
    return stats.beta.ppf(1 - alpha/2, k + 1, n - k)


def clopper_pearson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval"""
    alpha = 1 - confidence
    
    if k == 0:
        lower = 0.0
    else:
        lower = stats.beta.ppf(alpha/2, k, n - k + 1)
    
    if k == n:
        upper = 1.0
    else:
        upper = stats.beta.ppf(1 - alpha/2, k + 1, n - k)
    
    return lower, upper


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval (better for small proportions)"""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = k / n
    
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denominator
    
    return max(0, center - margin), min(1, center + margin)


def bootstrap_miss_probability(miss_counts: List[int], total_counts: List[int],
                                n_bootstrap: int = 10000, 
                                confidence: float = 0.95) -> Dict:
    """
    Bootstrap confidence interval for miss probability
    
    Args:
        miss_counts: List of miss counts from each seed
        total_counts: List of total arrivals from each seed
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        Dict with mean, lower, upper bounds
    """
    n_seeds = len(miss_counts)
    rng = np.random.default_rng(42)
    
    # Original estimate
    total_misses = sum(miss_counts)
    total_arrivals = sum(total_counts)
    point_estimate = total_misses / total_arrivals
    
    # Bootstrap
    boot_estimates = []
    for _ in range(n_bootstrap):
        # Resample seeds with replacement
        indices = rng.choice(n_seeds, size=n_seeds, replace=True)
        boot_misses = sum(miss_counts[i] for i in indices)
        boot_arrivals = sum(total_counts[i] for i in indices)
        boot_estimates.append(boot_misses / boot_arrivals)
    
    boot_estimates = np.array(boot_estimates)
    
    # Percentile method
    alpha = (1 - confidence) / 2
    lower = np.percentile(boot_estimates, alpha * 100)
    upper = np.percentile(boot_estimates, (1 - alpha) * 100)
    
    # BCa method (bias-corrected and accelerated) for better accuracy
    # Simplified version
    
    return {
        'point_estimate': point_estimate,
        'mean': np.mean(boot_estimates),
        'std': np.std(boot_estimates),
        'lower': lower,
        'upper': upper,
        'confidence': confidence
    }


def compute_required_samples(target_bound: float, confidence: float = 0.95) -> int:
    """
    Compute required sample size to achieve target upper bound
    
    Using Rule of Three: n ≈ 3 / target_bound at 95% confidence
    """
    alpha = 1 - confidence
    return int(np.ceil(-np.log(alpha) / target_bound))


#=============================================================================
# ANALYSIS FOR SENTRY-LITE-2
#=============================================================================

def analyze_miss_probability_bounds(experimental_results: Dict) -> pd.DataFrame:
    """
    Comprehensive analysis of miss probability bounds
    
    Args:
        experimental_results: Dict with 'miss_counts' and 'total_arrivals' lists
    """
    miss_counts = experimental_results['miss_counts']
    total_counts = experimental_results['total_arrivals']
    
    total_misses = sum(miss_counts)
    total_arrivals = sum(total_counts)
    
    results = []
    
    # 1. Point estimate
    point_est = total_misses / total_arrivals
    results.append({
        'Method': 'Point Estimate',
        'Lower': point_est,
        'Upper': point_est,
        'Notes': f'{total_misses}/{total_arrivals} misses'
    })
    
    # 2. Rule of Three (if zero events)
    if total_misses == 0:
        r3_bound = rule_of_three_bound(total_arrivals)
        results.append({
            'Method': 'Rule of Three (95%)',
            'Lower': 0,
            'Upper': r3_bound,
            'Notes': f'Upper bound when 0 events observed'
        })
    
    # 3. Clopper-Pearson exact
    cp_lower, cp_upper = clopper_pearson_interval(total_misses, total_arrivals)
    results.append({
        'Method': 'Clopper-Pearson (95%)',
        'Lower': cp_lower,
        'Upper': cp_upper,
        'Notes': 'Exact binomial CI'
    })
    
    # 4. Wilson score
    w_lower, w_upper = wilson_interval(total_misses, total_arrivals)
    results.append({
        'Method': 'Wilson Score (95%)',
        'Lower': w_lower,
        'Upper': w_upper,
        'Notes': 'Better for small proportions'
    })
    
    # 5. Bootstrap
    boot = bootstrap_miss_probability(miss_counts, total_counts)
    results.append({
        'Method': 'Bootstrap (95%)',
        'Lower': boot['lower'],
        'Upper': boot['upper'],
        'Notes': f'10000 resamples, std={boot["std"]:.6f}'
    })
    
    return pd.DataFrame(results)


#=============================================================================
# MAIN TEST
#=============================================================================

if __name__ == '__main__':
    print("="*70)
    print("W2 SOLUTION: STATISTICAL BOUNDS FOR MISS PROBABILITY")
    print("="*70)
    
    # Simulate experimental results (20 seeds × 50k TTIs)
    # Using actual SENTRY-Lite-2 results
    
    import sys
    sys.path.insert(0, 'C:/Users/LOQ/Desktop/SENTRY-Lite-2/src')
    
    print("\n[1. Running Experiments to Collect Data]")
    print("-"*50)
    
    from environment.urllc_env import URLLCEnv, URRLCConfig
    
    config = URRLCConfig(load=0.85, deadline=0.5, episode_length=50000)
    
    miss_counts = []
    total_arrivals = []
    
    for seed in range(20):
        env = URLLCEnv(config, seed=seed)
        env.reset(seed=seed)
        
        counter = 0
        for t in range(config.episode_length):
            Q = len(env.queue)
            slack = env.ewma_slack
            
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
        miss_counts.append(metrics['total_misses'])
        total_arrivals.append(metrics['total_arrivals'])
        
        if seed < 5:
            print(f"  Seed {seed}: {metrics['total_misses']} misses / {metrics['total_arrivals']} arrivals")
    
    print(f"  ...")
    print(f"  Total: {sum(miss_counts)} misses / {sum(total_arrivals)} arrivals")
    
    # Analyze bounds
    print("\n[2. Statistical Bounds Analysis]")
    print("-"*50)
    
    exp_results = {
        'miss_counts': miss_counts,
        'total_arrivals': total_arrivals
    }
    
    bounds_df = analyze_miss_probability_bounds(exp_results)
    
    for _, row in bounds_df.iterrows():
        print(f"\n{row['Method']}:")
        print(f"  Interval: [{row['Lower']*100:.6f}%, {row['Upper']*100:.6f}%]")
        print(f"  Notes: {row['Notes']}")
    
    # URLLC compliance check
    print("\n[3. URLLC Compliance Check]")
    print("-"*50)
    
    urllc_threshold = 0.01  # 1% = 10^-2
    strict_threshold = 0.001  # 0.1% = 10^-3
    
    cp_upper = bounds_df[bounds_df['Method'] == 'Clopper-Pearson (95%)']['Upper'].values[0]
    
    print(f"Clopper-Pearson 95% Upper Bound: {cp_upper*100:.4f}%")
    print(f"URLLC Threshold (1%):            {urllc_threshold*100:.2f}%")
    print(f"Strict Threshold (0.1%):         {strict_threshold*100:.2f}%")
    print()
    
    if cp_upper < strict_threshold:
        print("✓ SENTRY-Lite-2 satisfies STRICT URLLC requirement (< 0.1%)")
    elif cp_upper < urllc_threshold:
        print("✓ SENTRY-Lite-2 satisfies URLLC requirement (< 1%)")
    else:
        print("✗ URLLC requirement not satisfied")
    
    # Required samples for tighter bounds
    print("\n[4. Sample Size Requirements]")
    print("-"*50)
    
    for target in [0.01, 0.001, 0.0001]:
        n_required = compute_required_samples(target)
        print(f"To claim P_miss < {target*100:.2f}% with 95% confidence: need {n_required:,} arrivals")
    
    print(f"\nWe have {sum(total_arrivals):,} arrivals → can claim upper bound of {3/sum(total_arrivals)*100:.4f}%")
    
    # Save results
    bounds_df.to_csv('experiments/results/miss_probability_bounds.csv', index=False)
    
    # LaTeX output
    print("\n" + "="*70)
    print("LATEX TABLE FOR PAPER")
    print("="*70)
    
    print(r"""
\begin{table}[t]
\centering
\caption{Statistical Bounds for Miss Probability (20 seeds $\times$ 50k TTIs)}
\label{tab:miss_bounds}
\small
\begin{tabular}{lcc}
\toprule
\textbf{Method} & \textbf{Lower} & \textbf{Upper} \\
\midrule""")
    
    for _, row in bounds_df.iterrows():
        print(f"{row['Method']} & {row['Lower']*100:.4f}\\% & {row['Upper']*100:.4f}\\% \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")
    
    print("\n" + "="*70)
    print("W2 ADDRESSED: Statistical bounds provided:")
    print("  ✓ Rule of Three upper bound (when 0 events)")
    print("  ✓ Clopper-Pearson exact binomial CI")
    print("  ✓ Wilson score interval")
    print("  ✓ Bootstrap confidence interval")
    print("  ✓ Required sample size calculations")
    print("="*70)
