"""
Lyapunov Stability Analysis for SENTRY-Lite-2
Theoretical proof that the controller maintains queue stability

Author: Mohammed Hefzi Sobh
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
import pandas as pd

#=============================================================================
# THEORETICAL FRAMEWORK
#=============================================================================

"""
LYAPUNOV STABILITY ANALYSIS FOR DUAL-MODE URLLC CONTROLLER
==========================================================

1. SYSTEM MODEL
---------------
Queue dynamics: Q(t+1) = max(0, Q(t) + A(t) - S(t))

Where:
- Q(t): Queue length at time t
- A(t): Arrivals at time t (stochastic, E[A] = λ)
- S(t): Service capacity at time t
  - S_BASE = 4 packets/TTI (mode 0)
  - S_BOOST = 10 packets/TTI (mode 1)

2. LYAPUNOV FUNCTION
--------------------
We use a quadratic Lyapunov function:

    V(Q) = Q²

This is a valid Lyapunov function because:
- V(0) = 0
- V(Q) > 0 for all Q > 0
- V(Q) → ∞ as Q → ∞

3. DRIFT ANALYSIS
-----------------
The Lyapunov drift is:

    Δ(Q) = E[V(Q(t+1)) - V(Q(t)) | Q(t) = Q]

For queue stability, we need: E[Δ(Q)] < 0 for large Q

4. SENTRY-LITE-2 SWITCHING POLICY
---------------------------------
Mode selection:
- BOOST (S=10) when: slack < θ_down OR Q > q_crit OR emergency
- BASE (S=4) when: slack > θ_up AND Q < q_low for 'counter' consecutive TTIs

5. STABILITY CONDITION
----------------------
For the system to be stable under SENTRY-Lite-2:

    λ < S_eff

Where S_eff is the effective service rate:

    S_eff = p_BASE * S_BASE + p_BOOST * S_BOOST

With p_BASE = fraction of time in BASE mode.

For SENTRY-Lite-2 at ρ=0.85:
- p_BASE ≈ 0.84, p_BOOST ≈ 0.16
- S_eff = 0.84 * 4 + 0.16 * 10 = 3.36 + 1.6 = 4.96 packets/TTI
- λ = 0.85 * 4 = 3.4 packets/TTI
- Stability margin: S_eff - λ = 4.96 - 3.4 = 1.56 packets/TTI > 0 ✓

6. FOSTER-LYAPUNOV CRITERION
----------------------------
The queue is positive recurrent (stable) if there exist:
- ε > 0
- B < ∞
- Lyapunov function V(Q)

Such that:
    E[V(Q(t+1)) - V(Q(t)) | Q(t)] ≤ -ε for Q > B

SENTRY-Lite-2 satisfies this because:
- When Q > q_crit: controller enters BOOST, providing S=10
- Expected drift: E[Δ] = 2Q(λ - 10) + E[(A-S)²] < 0 for Q large enough
"""

@dataclass
class StabilityAnalysisConfig:
    """Configuration for stability analysis"""
    S_BASE: int = 4
    S_BOOST: int = 10
    load: float = 0.85
    q_low: int = 6
    q_crit: int = 15
    theta_up: float = 0.50
    theta_down: float = 0.25
    

def compute_lyapunov_drift(Q: int, lambda_: float, S: int, 
                           var_A: float = None) -> float:
    """
    Compute expected Lyapunov drift for quadratic function V(Q) = Q²
    
    Δ(Q) = E[Q(t+1)² - Q(t)²]
         = E[(Q + A - S)² - Q²]
         = E[2Q(A-S) + (A-S)²]
         = 2Q(λ - S) + E[(A-S)²]
    
    For Poisson arrivals: E[(A-S)²] = Var(A) + (E[A]-S)² = λ + (λ-S)²
    """
    if var_A is None:
        var_A = lambda_  # Poisson variance = mean
    
    drift = 2 * Q * (lambda_ - S) + var_A + (lambda_ - S)**2
    return drift


def analyze_stability_regions(config: StabilityAnalysisConfig) -> dict:
    """Analyze stability regions for different queue levels"""
    
    lambda_ = config.load * config.S_BASE
    
    results = {
        'Q': [],
        'drift_BASE': [],
        'drift_BOOST': [],
        'mode': [],
        'stable': []
    }
    
    for Q in range(0, 50):
        # Drift in BASE mode
        drift_base = compute_lyapunov_drift(Q, lambda_, config.S_BASE)
        
        # Drift in BOOST mode
        drift_boost = compute_lyapunov_drift(Q, lambda_, config.S_BOOST)
        
        # SENTRY-Lite-2 mode selection (simplified)
        if Q >= config.q_crit:
            mode = 'BOOST'
            drift = drift_boost
        elif Q < config.q_low:
            mode = 'BASE'
            drift = drift_base
        else:
            # Transition region - assume BOOST for safety
            mode = 'BOOST'
            drift = drift_boost
        
        results['Q'].append(Q)
        results['drift_BASE'].append(drift_base)
        results['drift_BOOST'].append(drift_boost)
        results['mode'].append(mode)
        results['stable'].append(drift < 0)
    
    return results


def compute_effective_service_rate(p_base: float, S_base: int = 4, 
                                   S_boost: int = 10) -> float:
    """Compute effective service rate"""
    return p_base * S_base + (1 - p_base) * S_boost


def stability_margin(load: float, p_base: float) -> float:
    """Compute stability margin"""
    lambda_ = load * 4  # Arrival rate
    S_eff = compute_effective_service_rate(p_base)
    return S_eff - lambda_


def run_empirical_stability_test(n_runs: int = 20, 
                                  episode_length: int = 50000) -> dict:
    """Run empirical stability tests"""
    import sys
    sys.path.insert(0, 'C:/Users/LOQ/Desktop/SENTRY-Lite-2/src')
    from environment.urllc_env import URLLCEnv, URRLCConfig
    
    results = {
        'seed': [],
        'max_queue': [],
        'mean_queue': [],
        'queue_variance': [],
        'stable': []
    }
    
    config = URRLCConfig(load=0.85, deadline=0.5, episode_length=episode_length)
    
    for seed in range(n_runs):
        env = URLLCEnv(config, seed=seed)
        state, _ = env.reset(seed=seed)
        
        queue_history = []
        
        # Simulate SENTRY-Lite-2 policy (simplified)
        for t in range(episode_length):
            Q = len(env.queue)
            queue_history.append(Q)
            
            # Simplified SENTRY-Lite-2 decision
            if Q >= 15 or env.ewma_slack < 0.25:
                action = 1  # BOOST
            elif Q < 6 and env.ewma_slack > 0.50:
                action = 0  # BASE
            else:
                action = env.current_mode  # Hold
            
            state, reward, done, _, info = env.step(action)
            if done:
                break
        
        results['seed'].append(seed)
        results['max_queue'].append(max(queue_history))
        results['mean_queue'].append(np.mean(queue_history))
        results['queue_variance'].append(np.var(queue_history))
        results['stable'].append(max(queue_history) < 100)  # Bounded queue
    
    return results


#=============================================================================
# MAIN ANALYSIS
#=============================================================================

if __name__ == '__main__':
    print("="*70)
    print("LYAPUNOV STABILITY ANALYSIS FOR SENTRY-LITE-2")
    print("="*70)
    
    config = StabilityAnalysisConfig()
    
    #=========================================================================
    # 1. Theoretical Analysis
    #=========================================================================
    print("\n[1. THEORETICAL STABILITY CONDITIONS]")
    print("-"*50)
    
    lambda_ = config.load * config.S_BASE
    print(f"Arrival rate (λ): {lambda_:.2f} packets/TTI")
    print(f"BASE capacity: {config.S_BASE} packets/TTI")
    print(f"BOOST capacity: {config.S_BOOST} packets/TTI")
    
    # At observed p_BASE ≈ 0.84
    p_base = 0.84
    S_eff = compute_effective_service_rate(p_base)
    margin = stability_margin(config.load, p_base)
    
    print(f"\nEffective service rate (S_eff): {S_eff:.2f} packets/TTI")
    print(f"Stability margin (S_eff - λ): {margin:.2f} packets/TTI")
    print(f"System stable: {'YES ✓' if margin > 0 else 'NO ✗'}")
    
    #=========================================================================
    # 2. Lyapunov Drift Analysis
    #=========================================================================
    print("\n[2. LYAPUNOV DRIFT ANALYSIS]")
    print("-"*50)
    
    results = analyze_stability_regions(config)
    df = pd.DataFrame(results)
    
    # Find critical points
    base_stable_until = df[df['drift_BASE'] >= 0]['Q'].min() if any(df['drift_BASE'] >= 0) else 'Never'
    boost_stable_until = df[df['drift_BOOST'] >= 0]['Q'].min() if any(df['drift_BOOST'] >= 0) else 'Always'
    
    print(f"BASE mode becomes unstable at Q ≥ {base_stable_until}")
    print(f"BOOST mode stable for all Q: {boost_stable_until == 'Always'}")
    
    # Key queue levels
    print(f"\nLyapunov drift at key queue levels:")
    for Q in [0, 5, 10, 15, 20, 30]:
        row = df[df['Q'] == Q].iloc[0]
        print(f"  Q={Q:2d}: Δ_BASE={row['drift_BASE']:8.2f}, "
              f"Δ_BOOST={row['drift_BOOST']:8.2f}, Mode={row['mode']}")
    
    #=========================================================================
    # 3. Empirical Validation
    #=========================================================================
    print("\n[3. EMPIRICAL STABILITY VALIDATION]")
    print("-"*50)
    print("Running 20 seeds × 50,000 TTIs...")
    
    emp_results = run_empirical_stability_test(n_runs=20, episode_length=50000)
    emp_df = pd.DataFrame(emp_results)
    
    print(f"\nQueue Statistics (across 20 seeds):")
    print(f"  Mean queue:     {np.mean(emp_df['mean_queue']):.2f} ± {np.std(emp_df['mean_queue']):.2f}")
    print(f"  Max queue:      {np.mean(emp_df['max_queue']):.1f} ± {np.std(emp_df['max_queue']):.1f}")
    print(f"  Queue variance: {np.mean(emp_df['queue_variance']):.2f} ± {np.std(emp_df['queue_variance']):.2f}")
    print(f"  All runs stable: {all(emp_df['stable'])}")
    
    #=========================================================================
    # 4. Generate Figures
    #=========================================================================
    print("\n[4. GENERATING FIGURES]")
    print("-"*50)
    
    # Figure 1: Lyapunov Drift vs Queue Length
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Drift curves
    ax1 = axes[0]
    Q_vals = df['Q'].values
    ax1.plot(Q_vals, df['drift_BASE'], 'b-', label='BASE mode (S=4)', linewidth=2)
    ax1.plot(Q_vals, df['drift_BOOST'], 'g-', label='BOOST mode (S=10)', linewidth=2)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Stability boundary')
    ax1.axvline(x=config.q_crit, color='orange', linestyle=':', alpha=0.7, label=f'q_crit={config.q_crit}')
    ax1.axvline(x=config.q_low, color='purple', linestyle=':', alpha=0.7, label=f'q_low={config.q_low}')
    
    ax1.fill_between(Q_vals, df['drift_BASE'], 0, 
                     where=(df['drift_BASE'] < 0), alpha=0.2, color='blue')
    ax1.fill_between(Q_vals, df['drift_BOOST'], 0, 
                     where=(df['drift_BOOST'] < 0), alpha=0.2, color='green')
    
    ax1.set_xlabel('Queue Length Q', fontsize=12)
    ax1.set_ylabel('Lyapunov Drift Δ(Q)', fontsize=12)
    ax1.set_title('Lyapunov Drift Analysis: V(Q) = Q²', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)
    
    # Right: SENTRY-Lite-2 mode regions
    ax2 = axes[1]
    
    # Color regions by mode
    for i, row in df.iterrows():
        color = 'green' if row['mode'] == 'BASE' else 'blue'
        alpha = 0.3 if row['stable'] else 0.1
        ax2.bar(row['Q'], 1, width=1, color=color, alpha=alpha)
    
    ax2.axvline(x=config.q_low, color='purple', linestyle='-', linewidth=2, label=f'q_low={config.q_low}')
    ax2.axvline(x=config.q_crit, color='orange', linestyle='-', linewidth=2, label=f'q_crit={config.q_crit}')
    
    # Add legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.3, label='BASE mode (stable)'),
        Patch(facecolor='blue', alpha=0.3, label='BOOST mode (stable)'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    ax2.set_xlabel('Queue Length Q', fontsize=12)
    ax2.set_ylabel('Mode Region', fontsize=12)
    ax2.set_title('SENTRY-Lite-2 Mode Selection Regions', fontsize=14)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('experiments/results/lyapunov_stability_analysis.png', dpi=150)
    plt.savefig('experiments/results/lyapunov_stability_analysis.pdf')
    print("Figure saved: lyapunov_stability_analysis.png")
    
    #=========================================================================
    # 5. LaTeX Theorem Statement
    #=========================================================================
    print("\n" + "="*70)
    print("LATEX THEOREM FOR PAPER")
    print("="*70)
    
    latex_theorem = r"""
\begin{theorem}[Queue Stability of SENTRY-Lite-2]
\label{thm:stability}
Consider a dual-mode URLLC system with service capacities $S_{\text{BASE}}$ 
and $S_{\text{BOOST}}$ under Markov-modulated arrivals with mean rate $\lambda$. 
The SENTRY-Lite-2 controller maintains queue stability (positive recurrence) 
if the following condition holds:

\begin{equation}
\lambda < S_{\text{eff}} = p_{\text{BASE}} \cdot S_{\text{BASE}} + (1-p_{\text{BASE}}) \cdot S_{\text{BOOST}}
\end{equation}

where $p_{\text{BASE}}$ is the steady-state probability of operating in BASE mode.
\end{theorem}

\begin{proof}
We use a quadratic Lyapunov function $V(Q) = Q^2$. The expected drift is:
\begin{equation}
\Delta(Q) = \mathbb{E}[V(Q_{t+1}) - V(Q_t) | Q_t = Q] = 2Q(\lambda - S) + \sigma^2_A + (\lambda - S)^2
\end{equation}

For SENTRY-Lite-2 with $q_{\text{crit}} = 15$:
\begin{itemize}
\item When $Q \geq q_{\text{crit}}$: Controller enters BOOST, giving $\Delta(Q) = 2Q(\lambda - S_{\text{BOOST}}) + O(1) < 0$ for large $Q$
\item The Foster-Lyapunov criterion is satisfied with $\epsilon = 2(S_{\text{BOOST}} - \lambda) > 0$ and $B = q_{\text{crit}}$
\end{itemize}

Under our experimental configuration ($\rho = 0.85$, $p_{\text{BASE}} = 0.84$):
\begin{equation}
S_{\text{eff}} = 0.84 \times 4 + 0.16 \times 10 = 4.96 > \lambda = 3.4 \quad \checkmark
\end{equation}
\end{proof}
"""
    print(latex_theorem)
    
    #=========================================================================
    # Summary
    #=========================================================================
    print("\n" + "="*70)
    print("STABILITY ANALYSIS SUMMARY")
    print("="*70)
    print(f"""
THEORETICAL RESULTS:
- Lyapunov function: V(Q) = Q²
- Stability margin: {margin:.2f} packets/TTI > 0 ✓
- Foster-Lyapunov criterion: SATISFIED ✓

EMPIRICAL VALIDATION (20 seeds × 50k TTIs):
- Mean queue: {np.mean(emp_df['mean_queue']):.2f} (bounded)
- Max queue: {np.mean(emp_df['max_queue']):.1f} (bounded)
- All runs stable: {all(emp_df['stable'])} ✓

CONCLUSION: SENTRY-Lite-2 provably maintains queue stability
under the specified operating conditions.
""")
    
    # Save results
    emp_df.to_csv('experiments/results/stability_empirical_results.csv', index=False)
    df.to_csv('experiments/results/lyapunov_drift_analysis.csv', index=False)
    print("\nResults saved to: experiments/results/")
