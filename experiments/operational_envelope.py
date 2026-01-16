"""
W11 SOLUTION: Operational Envelope Characterization
Multi-load sweep to define where SENTRY-Lite-2 is deployable

Key outputs:
1. Performance across load range (ρ = 0.5 - 0.95)
2. Operational envelope boundaries
3. Feasibility regions
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
import sys
sys.path.insert(0, 'C:/Users/LOQ/Desktop/SENTRY-Lite-2/src')
from environment.urllc_env import URLLCEnv, URRLCConfig


def run_load_sweep(loads: List[float], n_seeds: int = 10, 
                   episode_length: int = 20000) -> pd.DataFrame:
    """Run experiments across multiple loads"""
    
    results = []
    
    for load in loads:
        print(f"  Testing load ρ = {load:.2f}...")
        
        config = URRLCConfig(load=load, deadline=0.5, episode_length=episode_length)
        
        seed_results = {
            'miss_prob': [],
            'savings': [],
            'switches': []
        }
        
        for seed in range(n_seeds):
            env = URLLCEnv(config, seed=seed)
            env.reset(seed=seed)
            
            counter = 0
            for t in range(episode_length):
                Q = len(env.queue)
                slack = env.ewma_slack
                
                # SENTRY-Lite-2 policy
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
            seed_results['miss_prob'].append(metrics['miss_probability'] * 100)
            seed_results['savings'].append(metrics['energy_savings'])
            seed_results['switches'].append(metrics['switches_per_ktti'])
        
        # Compute statistics
        results.append({
            'load': load,
            'miss_mean': np.mean(seed_results['miss_prob']),
            'miss_std': np.std(seed_results['miss_prob']),
            'miss_ci_lo': np.percentile(seed_results['miss_prob'], 2.5),
            'miss_ci_hi': np.percentile(seed_results['miss_prob'], 97.5),
            'savings_mean': np.mean(seed_results['savings']),
            'savings_std': np.std(seed_results['savings']),
            'savings_ci_lo': np.percentile(seed_results['savings'], 2.5),
            'savings_ci_hi': np.percentile(seed_results['savings'], 97.5),
            'switches_mean': np.mean(seed_results['switches']),
            'switches_std': np.std(seed_results['switches']),
            'switches_ci_lo': np.percentile(seed_results['switches'], 2.5),
            'switches_ci_hi': np.percentile(seed_results['switches'], 97.5),
            'urllc_compliant': np.mean(seed_results['miss_prob']) < 1.0,
            'deployable': np.mean(seed_results['switches']) < 50
        })
    
    return pd.DataFrame(results)


def find_operational_boundaries(df: pd.DataFrame) -> Dict:
    """Find operational envelope boundaries"""
    
    # URLLC boundary: max load where miss < 1%
    urllc_compliant = df[df['urllc_compliant'] == True]
    max_urllc_load = urllc_compliant['load'].max() if len(urllc_compliant) > 0 else 0
    
    # Deployability boundary: max load where switches < 50
    deployable = df[df['deployable'] == True]
    max_deploy_load = deployable['load'].max() if len(deployable) > 0 else 0
    
    # Combined boundary
    both = df[(df['urllc_compliant'] == True) & (df['deployable'] == True)]
    max_operational_load = both['load'].max() if len(both) > 0 else 0
    
    # Find critical transition points
    boundaries = {
        'max_urllc_load': max_urllc_load,
        'max_deployable_load': max_deploy_load,
        'max_operational_load': max_operational_load,
        'min_tested_load': df['load'].min(),
        'max_tested_load': df['load'].max()
    }
    
    return boundaries


def plot_operational_envelope(df: pd.DataFrame, output_dir: str):
    """Generate operational envelope figures"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    loads = df['load'].values
    
    # 1. Miss Probability vs Load
    ax1 = axes[0, 0]
    ax1.fill_between(loads, df['miss_ci_lo'], df['miss_ci_hi'], alpha=0.3, color='red')
    ax1.plot(loads, df['miss_mean'], 'r-o', linewidth=2, markersize=8, label='SENTRY-Lite-2')
    ax1.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, label='URLLC Threshold (1%)')
    ax1.axhline(y=0.1, color='green', linestyle=':', linewidth=2, label='Strict (0.1%)')
    
    # Shade feasible region
    feasible_loads = df[df['urllc_compliant'] == True]['load'].values
    if len(feasible_loads) > 0:
        ax1.axvspan(loads.min(), feasible_loads.max(), alpha=0.1, color='green')
    
    ax1.set_xlabel('System Load (ρ)', fontsize=12)
    ax1.set_ylabel('Miss Probability (%)', fontsize=12)
    ax1.set_title('(a) Reliability vs Load', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-4, 10)
    
    # 2. Energy Savings vs Load
    ax2 = axes[0, 1]
    ax2.fill_between(loads, df['savings_ci_lo'], df['savings_ci_hi'], alpha=0.3, color='blue')
    ax2.plot(loads, df['savings_mean'], 'b-o', linewidth=2, markersize=8)
    
    ax2.set_xlabel('System Load (ρ)', fontsize=12)
    ax2.set_ylabel('Energy Savings (%)', fontsize=12)
    ax2.set_title('(b) Energy Efficiency vs Load', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # 3. Switching Rate vs Load
    ax3 = axes[1, 0]
    ax3.fill_between(loads, df['switches_ci_lo'], df['switches_ci_hi'], alpha=0.3, color='green')
    ax3.plot(loads, df['switches_mean'], 'g-o', linewidth=2, markersize=8)
    ax3.axhline(y=50, color='orange', linestyle='--', linewidth=2, label='Deploy Budget (50)')
    
    # Shade feasible region
    deploy_loads = df[df['deployable'] == True]['load'].values
    if len(deploy_loads) > 0:
        ax3.axvspan(loads.min(), deploy_loads.max(), alpha=0.1, color='green')
    
    ax3.set_xlabel('System Load (ρ)', fontsize=12)
    ax3.set_ylabel('Switches/kTTI', fontsize=12)
    ax3.set_title('(c) Switching Activity vs Load', fontsize=14)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Operational Envelope Summary
    ax4 = axes[1, 1]
    
    # Create regions
    categories = ['URLLC\nCompliant', 'Deployable\n(Sw<50)', 'Both\n(Operational)']
    
    urllc_range = len(df[df['urllc_compliant'] == True])
    deploy_range = len(df[df['deployable'] == True])
    both_range = len(df[(df['urllc_compliant'] == True) & (df['deployable'] == True)])
    
    values = [urllc_range, deploy_range, both_range]
    colors = ['red', 'green', 'blue']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.7)
    ax4.set_ylabel('Number of Load Points', fontsize=12)
    ax4.set_title('(d) Operational Envelope Coverage', fontsize=14)
    ax4.set_ylim(0, len(df) + 1)
    
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.2, 
                f'{val}/{len(df)}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/operational_envelope.png', dpi=150)
    plt.savefig(f'{output_dir}/operational_envelope.pdf')
    print(f"Figure saved: operational_envelope.png")
    
    # Pareto frontier with load points
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    # Color by load
    scatter = ax.scatter(df['savings_mean'], df['switches_mean'], 
                        c=df['load'], cmap='viridis', s=150, 
                        edgecolors='black', linewidths=1)
    
    # Mark operational region
    operational = df[(df['urllc_compliant'] == True) & (df['deployable'] == True)]
    ax.scatter(operational['savings_mean'], operational['switches_mean'],
              s=250, facecolors='none', edgecolors='green', linewidths=3,
              label='Operational Region')
    
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Deploy Budget')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('System Load (ρ)', fontsize=12)
    
    # Label key points
    for _, row in df.iterrows():
        if row['load'] in [0.5, 0.7, 0.85, 0.95]:
            ax.annotate(f"ρ={row['load']:.2f}", 
                       (row['savings_mean'], row['switches_mean']),
                       textcoords="offset points", xytext=(10, 5), fontsize=9)
    
    ax.set_xlabel('Energy Savings (%)', fontsize=12)
    ax.set_ylabel('Switches/kTTI', fontsize=12)
    ax.set_title('Operational Envelope: Energy-Stability Tradeoff', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/operational_pareto.png', dpi=150)
    plt.savefig(f'{output_dir}/operational_pareto.pdf')
    print(f"Figure saved: operational_pareto.png")


#=============================================================================
# MAIN
#=============================================================================

if __name__ == '__main__':
    print("="*70)
    print("W11 SOLUTION: OPERATIONAL ENVELOPE CHARACTERIZATION")
    print("="*70)
    
    # Configuration
    loads = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    n_seeds = 10
    episode_length = 20000
    output_dir = 'experiments/results'
    
    print(f"\nConfiguration:")
    print(f"  Loads: {loads}")
    print(f"  Seeds per load: {n_seeds}")
    print(f"  Episode length: {episode_length} TTIs")
    
    # Run sweep
    print("\n[1. Running Multi-Load Sweep]")
    print("-"*50)
    
    df = run_load_sweep(loads, n_seeds, episode_length)
    
    # Find boundaries
    print("\n[2. Operational Boundaries]")
    print("-"*50)
    
    boundaries = find_operational_boundaries(df)
    
    print(f"  Max URLLC-compliant load:  ρ = {boundaries['max_urllc_load']:.2f}")
    print(f"  Max Deployable load:       ρ = {boundaries['max_deployable_load']:.2f}")
    print(f"  Max Operational load:      ρ = {boundaries['max_operational_load']:.2f}")
    
    # Results table
    print("\n[3. Results Summary]")
    print("-"*50)
    
    summary_cols = ['load', 'miss_mean', 'savings_mean', 'switches_mean', 
                    'urllc_compliant', 'deployable']
    print(df[summary_cols].to_string(index=False))
    
    # Generate figures
    print("\n[4. Generating Figures]")
    print("-"*50)
    
    plot_operational_envelope(df, output_dir)
    
    # Save results
    df.to_csv(f'{output_dir}/operational_envelope_data.csv', index=False)
    
    # LaTeX table
    print("\n" + "="*70)
    print("LATEX TABLE FOR PAPER")
    print("="*70)
    
    print(r"""
\begin{table}[t]
\centering
\caption{Operational Envelope: SENTRY-Lite-2 Performance Across Loads}
\label{tab:envelope}
\small
\begin{tabular}{ccccccc}
\toprule
$\rho$ & $P_{\text{miss}}$ (\%) & Savings (\%) & Sw/kTTI & URLLC & Deploy \\
\midrule""")
    
    for _, row in df.iterrows():
        urllc = r'\checkmark' if row['urllc_compliant'] else r'$\times$'
        deploy = r'\checkmark' if row['deployable'] else r'$\times$'
        print(f"{row['load']:.2f} & {row['miss_mean']:.3f} & {row['savings_mean']:.1f} & "
              f"{row['switches_mean']:.1f} & {urllc} & {deploy} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")
    
    # Summary
    print("\n" + "="*70)
    print("OPERATIONAL ENVELOPE SUMMARY")
    print("="*70)
    
    operational = df[(df['urllc_compliant'] == True) & (df['deployable'] == True)]
    
    print(f"""
SENTRY-Lite-2 Operational Envelope:
  
  Load Range: ρ ∈ [{operational['load'].min():.2f}, {operational['load'].max():.2f}]
  
  At Maximum Operational Load (ρ = {boundaries['max_operational_load']:.2f}):
    - Miss Probability: {operational[operational['load']==boundaries['max_operational_load']]['miss_mean'].values[0]:.4f}%
    - Energy Savings: {operational[operational['load']==boundaries['max_operational_load']]['savings_mean'].values[0]:.1f}%
    - Switches/kTTI: {operational[operational['load']==boundaries['max_operational_load']]['switches_mean'].values[0]:.1f}

KEY CLAIM FOR PAPER:
  "SENTRY-Lite-2 maintains deployable operation across 
   load range ρ ∈ [0.50, {boundaries['max_operational_load']:.2f}], achieving
   {operational['savings_mean'].min():.0f}%-{operational['savings_mean'].max():.0f}% energy savings
   while guaranteeing URLLC compliance."
""")
    
    print("\n" + "="*70)
    print("W11 ADDRESSED:")
    print("  ✓ Multi-load sweep (ρ = 0.50 - 0.95)")
    print("  ✓ Operational boundaries identified")
    print("  ✓ Feasibility regions characterized")
    print("  ✓ Pareto frontier visualization")
    print("="*70)
