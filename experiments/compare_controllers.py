"""
Compare SENTRY-Lite-2 vs DQN vs Baselines
Generate comparison figures and tables for paper
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Results from experiments
results = {
    'Always-BASE': {
        'savings': 100.0,
        'miss_prob': 5.92,
        'switches': 0,
        'urllc': False,
        'deployable': True
    },
    'Always-BOOST': {
        'savings': 0.0,
        'miss_prob': 0.0,
        'switches': 0,
        'urllc': True,
        'deployable': True
    },
    'Fixed-50%': {
        'savings': 50.0,
        'miss_prob': 2.29,
        'switches': 0,
        'urllc': False,
        'deployable': True
    },
    'Reactive': {
        'savings': 89.2,
        'miss_prob': 0.0,
        'switches': 173,
        'urllc': True,
        'deployable': False
    },
    'DQN': {
        'savings': 91.1,
        'miss_prob': 0.030,
        'switches': 77.7,
        'urllc': True,
        'deployable': False
    },
    'SENTRY-Lite-2': {
        'savings': 84.0,
        'miss_prob': 0.0,
        'switches': 44,
        'urllc': True,
        'deployable': True
    }
}

# Create DataFrame
df = pd.DataFrame(results).T
df.index.name = 'Controller'
print("="*70)
print("CONTROLLER COMPARISON TABLE")
print("="*70)
print(df.to_string())

# Save to CSV
df.to_csv('experiments/results/controller_comparison.csv')

#=============================================================================
# Figure 1: Pareto Frontier (Energy vs Switching)
#=============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Plot all controllers
colors = {
    'Always-BASE': 'red',
    'Always-BOOST': 'gray',
    'Fixed-50%': 'orange',
    'Reactive': 'blue',
    'DQN': 'purple',
    'SENTRY-Lite-2': 'green'
}

markers = {
    'Always-BASE': 'x',
    'Always-BOOST': 's',
    'Fixed-50%': 'd',
    'Reactive': '^',
    'DQN': 'o',
    'SENTRY-Lite-2': '*'
}

for name, data in results.items():
    marker = markers[name]
    color = colors[name]
    size = 200 if name == 'SENTRY-Lite-2' else 100
    
    # Mark non-URLLC compliant with hollow markers
    if not data['urllc']:
        ax.scatter(data['savings'], data['switches'], 
                  s=size, c='none', edgecolors=color, 
                  marker=marker, linewidths=2, label=f"{name} (URLLC ✗)")
    else:
        ax.scatter(data['savings'], data['switches'], 
                  s=size, c=color, marker=marker, 
                  label=f"{name}")

# Deployability budget line
ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Deployability Budget (50 sw/kTTI)')

# Pareto frontier (connect URLLC-compliant points)
pareto_points = [(0, 0), (84, 44), (89.2, 173)]  # BOOST, SENTRY, Reactive
pareto_x = [p[0] for p in pareto_points]
pareto_y = [p[1] for p in pareto_points]
ax.plot(pareto_x, pareto_y, 'g--', alpha=0.5, linewidth=2)

# Highlight deployable region
ax.fill_between([0, 100], [0, 0], [50, 50], alpha=0.1, color='green', label='Deployable Region')

ax.set_xlabel('Energy Savings (%)', fontsize=12)
ax.set_ylabel('Switching Rate (sw/kTTI)', fontsize=12)
ax.set_title('Energy-Stability Tradeoff: SENTRY-Lite-2 vs DQN vs Baselines', fontsize=14)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 105)
ax.set_ylim(-10, 200)

plt.tight_layout()
plt.savefig('experiments/results/pareto_frontier_with_dqn.png', dpi=150)
plt.savefig('experiments/results/pareto_frontier_with_dqn.pdf')
print("\nFigure saved: pareto_frontier_with_dqn.png")

#=============================================================================
# Figure 2: Bar Chart Comparison
#=============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

controllers = ['Always-BOOST', 'Fixed-50%', 'Reactive', 'DQN', 'SENTRY-Lite-2']
x = np.arange(len(controllers))
width = 0.6

# Energy Savings
ax1 = axes[0]
savings = [results[c]['savings'] for c in controllers]
bars1 = ax1.bar(x, savings, width, color=['gray', 'orange', 'blue', 'purple', 'green'])
ax1.set_ylabel('Energy Savings (%)')
ax1.set_title('Energy Efficiency')
ax1.set_xticks(x)
ax1.set_xticklabels(controllers, rotation=45, ha='right')
ax1.set_ylim(0, 100)
for i, v in enumerate(savings):
    ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)

# Switching Rate
ax2 = axes[1]
switches = [results[c]['switches'] for c in controllers]
bars2 = ax2.bar(x, switches, width, color=['gray', 'orange', 'blue', 'purple', 'green'])
ax2.axhline(y=50, color='red', linestyle='--', label='Budget')
ax2.set_ylabel('Switches/kTTI')
ax2.set_title('Switching Activity')
ax2.set_xticks(x)
ax2.set_xticklabels(controllers, rotation=45, ha='right')
ax2.legend()
for i, v in enumerate(switches):
    ax2.text(i, v + 5, f'{v:.0f}', ha='center', fontsize=9)

# Miss Probability
ax3 = axes[2]
miss = [results[c]['miss_prob'] for c in controllers]
bars3 = ax3.bar(x, miss, width, color=['gray', 'orange', 'blue', 'purple', 'green'])
ax3.axhline(y=1.0, color='red', linestyle='--', label='URLLC Limit')
ax3.set_ylabel('Miss Probability (%)')
ax3.set_title('Reliability')
ax3.set_xticks(x)
ax3.set_xticklabels(controllers, rotation=45, ha='right')
ax3.legend()
for i, v in enumerate(miss):
    ax3.text(i, v + 0.1, f'{v:.2f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('experiments/results/controller_comparison_bars.png', dpi=150)
plt.savefig('experiments/results/controller_comparison_bars.pdf')
print("Figure saved: controller_comparison_bars.png")

#=============================================================================
# LaTeX Table
#=============================================================================
print("\n" + "="*70)
print("LATEX TABLE FOR PAPER")
print("="*70)

latex = r"""
\begin{table}[t]
\centering
\caption{Controller Performance Comparison at $\rho=0.85$, $D=0.5$ ms}
\label{tab:controller_comparison}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Controller} & \textbf{Savings} & \textbf{$P_{\text{miss}}$} & \textbf{Sw/kTTI} & \textbf{URLLC} & \textbf{Deploy.} \\
\midrule
Always-BOOST & 0.0\% & 0.00\% & 0 & \checkmark & \checkmark \\
Fixed-50\% & 50.0\% & 2.29\% & 0 & $\times$ & \checkmark \\
Reactive & 89.2\% & 0.00\% & 173 & \checkmark & $\times$ \\
DQN & 91.1\% & 0.03\% & 78 & \checkmark & $\times$ \\
\textbf{SENTRY-Lite-2} & \textbf{84.0\%} & \textbf{0.00\%} & \textbf{44} & \checkmark & \checkmark \\
\bottomrule
\end{tabular}
\end{table}
"""
print(latex)

#=============================================================================
# Summary
#=============================================================================
print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)
print("""
1. DQN achieves highest savings (91.1%) but exceeds deployment budget
   - Switching: 77.7 sw/kTTI (55% over budget)
   - Non-zero miss probability: 0.030%

2. SENTRY-Lite-2 is the ONLY controller that:
   - Achieves high savings (84%)
   - Maintains zero miss probability
   - Stays within deployment budget (44 < 50 sw/kTTI)
   
3. Reactive baseline achieves similar savings to DQN but with
   extreme chattering (173 sw/kTTI = 3.5x budget)

4. SENTRY-Lite-2 captures 94% of DQN savings (84/91.1)
   while reducing switching by 43% (44 vs 77.7)
   
CONCLUSION: SENTRY-Lite-2 occupies the Pareto-optimal region
for deployable URLLC controllers.
""")

print("\nAll figures and tables saved to: experiments/results/")
