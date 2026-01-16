# W1: Wireless Channel Effects

## Extension Summary

We extended the evaluation to include wireless channel effects (AWGN, Rayleigh, and Rician fading) using the same traffic/load settings as the reference experiments.

## Methodology

- **Channel Models**: AWGN, Rayleigh (K=0), Rician (K=3, 6, 10 dB)
- **CQI Adaptation**: CQI-dependent capacity based on 3GPP mapping
- **Error Model**: BLER-driven packet errors with HARQ retransmissions
- **Doppler**: Tested at fd = 10 Hz (pedestrian), 50 Hz (vehicular)

## Tested Parameter Ranges

| Parameter | Range | Reference |
|-----------|-------|-----------|
| Load ρ | 0.50 - 0.95 | Table II |
| Deadline D | 0.25 - 1.0 ms | 6G URLLC |
| SNR | 5 - 25 dB | Typical indoor/outdoor |
| K-factor | 0 - 10 dB | NLOS to moderate LOS |

## Key Finding

Under the tested parameter ranges (documented in configs), SENTRY-Lite-2 preserves URLLC compliance with miss probability remaining below the reported threshold within the feasible operating envelope.

## Reproduction

```bash
# Run fading comparison
python experiments/envelope_2d.py

# Results location
experiments/results/envelope_2d_rayleigh.csv
experiments/results/envelope_2d_rician.csv
```

## Output Files

| File | Description |
|------|-------------|
| `envelope_2d_rayleigh.csv` | Rayleigh fading sweep results |
| `envelope_2d_rician.csv` | Rician fading sweep results |
| `envelope_2d_rayleigh.png` | Visualization |
| `envelope_2d_rician.png` | Visualization |

## Configuration

See `configs/default_config.yaml` for base parameters.
