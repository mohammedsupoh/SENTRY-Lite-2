# Documentation Index

This folder contains supplementary documentation for the SENTRY-Lite-2 reproducibility package.

## Robustness Extensions

Planned experimental extensions demonstrating robustness across various conditions:

| Document | Description | Key Experiment |
|----------|-------------|----------------|
| [W1: Wireless Channel](robustness-extensions/W1_wireless_channel.md) | Fading channel evaluation | `envelope_2d.py` |
| [W2: Statistical Bounds](robustness-extensions/W2_statistical_bounds.md) | Confidence intervals for rare events | `statistical_bounds.py` |
| [W11: Operational Envelope](robustness-extensions/W11_operational_envelope.md) | Load-deadline sweep analysis | `operational_envelope.py` |

## Methods

Technical methodology documentation:

| Document | Description |
|----------|-------------|
| [Statistical Bounds](methods/statistical_bounds.md) | Clopper-Pearson and Rule-of-Three methods |

## Quick Reference

### Running All Robustness Experiments

```bash
# W1: Wireless channel effects
python experiments/envelope_2d.py

# W2: Statistical bounds
python experiments/statistical_bounds.py

# W11: Operational envelope
python experiments/operational_envelope.py
```

### Output Location

All results are written to `experiments/results/`.

## Note on Terminology

These documents describe **robustness extensions** and **planned validations** for the methodology presented in the paper. All claims are bounded to the tested parameter ranges documented in each file.
