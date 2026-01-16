# SENTRY-Lite-2: Robustness Validation Plan

This document outlines the robustness extensions and validation experiments included in this repository.

> **Note**: These are planned/implemented validations demonstrating robustness across various conditions. All claims are bounded to the tested parameter ranges.

## Validation Summary

| Extension | Description | Script | Status |
|-----------|-------------|--------|--------|
| Wireless Channel | Fading evaluation (AWGN, Rayleigh, Rician) | `experiments/envelope_2d.py` | ✅ Complete |
| Statistical Bounds | Clopper-Pearson CI for rare events | `experiments/statistical_bounds.py` | ✅ Complete |
| Operational Envelope | Load-deadline sweep | `experiments/operational_envelope.py` | ✅ Complete |
| Sensitivity Analysis | Parameter sensitivity | `experiments/sensitivity_analysis.py` | ✅ Complete |
| Multi-UE Scenarios | Shared resource contention | `experiments/multi_ue_experiment.py` | ✅ Complete |

## Quick Reproduction

```bash
# Core validation (Table II reference)
python experiments/paper_validation_v2.py

# Wireless channel effects
python experiments/envelope_2d.py

# Statistical bounds
python experiments/statistical_bounds.py

# Operational envelope
python experiments/operational_envelope.py
```

## Documentation

Detailed methodology for each extension is available in `docs/`:

- `docs/robustness-extensions/W1_wireless_channel.md`
- `docs/robustness-extensions/W2_statistical_bounds.md`
- `docs/robustness-extensions/W11_operational_envelope.md`
- `docs/methods/statistical_bounds.md`

## Parameter Ranges

All results are bounded to the following tested ranges:

| Parameter | Range | Notes |
|-----------|-------|-------|
| Load ρ | 0.50 - 0.95 | Step 0.05 |
| Deadline D | 0.25 - 1.0 ms | 6G URLLC |
| Seeds | 0 - 19 | 20 seeds per configuration |
| TTIs/seed | 100,000 | 10M total per scenario |

## Output Location

All results are written to `experiments/results/`:
- CSV data files
- PNG/PDF figures
- Pre-generated for reproducibility verification

## Limitations

Results are specific to the tested parameter ranges and simulation assumptions. Extrapolation beyond these ranges requires additional validation.
