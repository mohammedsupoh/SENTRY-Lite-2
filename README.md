# SENTRY-Lite-2

**A Pareto-Optimal Dual-Mode Controller for Energy-Efficient URLLC Systems**

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TWC-blue)](https://ieeexplore.ieee.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository contains the simulation code and experiment scripts for the paper:

> M. Supoh, "SENTRY-Lite-2: A Pareto-Optimal Dual-Mode Controller for Energy-Efficient URLLC Systems," *IEEE Transactions on Wireless Communications*, under review, 2025.

SENTRY-Lite-2 is a lightweight UE-side dual-mode controller that achieves **84% energy savings** while maintaining URLLC compliance (P_miss < 1%) through a queue-safe exit mechanism.

## Key Results

| Metric | Value |
|--------|-------|
| Energy Savings (Single-UE) | 84% |
| Miss Probability | < 0.0001% (95% CI) |
| Switching Rate | 44 sw/kTTI |
| Safety Margin | 11,000× below URLLC threshold |

## Repository Structure

```
SENTRY-Lite-2/
├── src/
│   ├── controller/
│   │   └── sentry_lite2.py      # Main controller implementation
│   └── environment/
│       ├── urllc_env.py         # Base URLLC environment
│       └── wireless_env.py      # Wireless channel models
├── experiments/
│   ├── single_ue_validation.py  # Table II experiments
│   ├── multi_ue_homogeneous.py  # Multi-UE experiments
│   ├── multi_ue_heterogeneous.py
│   ├── statistical_bounds.py    # Clopper-Pearson CI
│   ├── operational_envelope.py  # Load-deadline sweep
│   ├── sensitivity_analysis.py  # Parameter sensitivity
│   └── pareto_frontier.py       # Pareto analysis
├── results/
│   └── (CSV outputs from experiments)
├── figures/
│   └── (Generated figures)
├── configs/
│   └── default_config.yaml      # Default parameters
└── README.md
```

## Requirements

```
Python >= 3.8
numpy >= 1.21
pandas >= 1.3
scipy >= 1.7
matplotlib >= 3.4
```

## Installation

```bash
git clone https://github.com/USERNAME/SENTRY-Lite-2.git
cd SENTRY-Lite-2
pip install -r requirements.txt
```

## Quick Start

### Run Single-UE Validation (Table II)

```bash
python experiments/single_ue_validation.py
```

### Run Statistical Bounds Analysis

```bash
python experiments/statistical_bounds.py
```

### Run Operational Envelope Sweep

```bash
python experiments/operational_envelope.py
```

## Controller Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `q_low` | 6 packets | Queue safety threshold |
| `counter` | 2 TTIs | Consecutive confirmation |
| `θ_up` | 0.50 | Safe exit threshold |
| `θ_down` | 0.25 | Entry threshold |
| `emergency_duration` | 5 TTIs | Burst absorption |
| `cooldown` | 1 TTI | Anti-chattering |
| `EWMA α` | 0.20 | Smoothing factor |

## System Parameters (6G-aligned)

| Parameter | Value |
|-----------|-------|
| TTI duration | 0.125 ms |
| S_BASE | 4 packets/TTI |
| S_BOOST | 10 packets/TTI |
| Deadline D | 0.5 ms (default) |
| Load ρ | 0.85 (default) |

## Reproducibility

All experiments use **fixed random seeds** (0-19 for 20 seeds) to ensure reproducibility.

To reproduce Table II results:
```bash
python experiments/single_ue_validation.py --seeds 20 --ttis 100000
```

## Citation

If you use this code, please cite:

```bibtex
@article{supoh2025sentry,
  author  = {M. Supoh},
  title   = {{SENTRY-Lite-2}: A {P}areto-Optimal Dual-Mode Controller 
             for Energy-Efficient {URLLC} Systems},
  journal = {IEEE Trans. Wireless Commun.},
  year    = {2025},
  note    = {under review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Mohammed Supoh - mohammed.sopuh7@gmail.com
