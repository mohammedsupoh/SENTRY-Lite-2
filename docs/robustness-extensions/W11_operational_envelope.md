# W11: Operational Envelope Analysis

## Extension Summary

We conducted a systematic load-deadline sweep to identify the operational envelope where both URLLC compliance and deployability constraints are simultaneously satisfied.

## Methodology

- **Load sweep**: ρ ∈ [0.50, 0.95] in steps of 0.05
- **Deadline sweep**: D ∈ [0.25, 1.0] ms
- **Seeds per point**: K = 20 (seeds 0-19)
- **TTIs per seed**: 100,000

## Constraint Definitions

| Constraint | Threshold | Rationale |
|------------|-----------|-----------|
| URLLC Compliance | P_miss < 1% | 3GPP reliability target |
| Deployability | Switches < 100 sw/kTTI | Hardware wear limit |

## Envelope Boundaries

The operational envelope represents (ρ, D) combinations where:
- SENTRY-Lite-2 achieves P_miss < 1%
- Switching rate remains < 100 sw/kTTI
- Energy savings exceed Always-Boost baseline

## Reproduction

```bash
# Run operational envelope sweep
python experiments/operational_envelope.py

# Results location
experiments/results/operational_envelope_data.csv
experiments/results/operational_envelope.png
experiments/results/operational_pareto.png
```

## Output Files

| File | Description |
|------|-------------|
| `operational_envelope_data.csv` | Full sweep results (ρ, D, metrics) |
| `operational_envelope.png/pdf` | Fig. 5 - Envelope heatmap |
| `operational_pareto.png/pdf` | Fig. 3 - Pareto frontier |

## Key Finding

The envelope boundaries and all raw outputs are included in this repository release. Results demonstrate that SENTRY-Lite-2 extends the feasible operating range compared to baseline controllers **within the tested parameter space**.

## Configuration

Base parameters from `configs/default_config.yaml`:
- TTI duration: 0.125 ms (6G-aligned)
- S_BASE: 4 packets/TTI
- S_BOOST: 10 packets/TTI

## Limitations

Results are specific to the tested parameter ranges. Extrapolation beyond these ranges requires additional validation.
