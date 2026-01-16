# W2: Statistical Bounds for Zero-Miss Observations

## Extension Summary

We address the "zero observed misses" concern by reporting exact statistical guarantees using established methods from reliability engineering.

## Methodology

For N arrivals with zero misses observed, we compute:

1. **Clopper-Pearson Upper Bound (95% CI)**: Exact binomial confidence interval
2. **Rule-of-Three Bound**: Conservative approximation ≈ 3/N

These bounds quantify the worst-case miss probability consistent with the observations.

## Mathematical Foundation

For k=0 successes (misses) out of N trials:

```
Clopper-Pearson upper bound (95%):
  p_upper = 1 - (α/2)^(1/N) where α = 0.05
  
Rule-of-Three approximation:
  p_upper ≈ 3/N (valid for large N, k=0)
```

## Results Format

| Scenario | N (arrivals) | Observed Misses | CP Upper 95% | Rule-of-Three |
|----------|--------------|-----------------|--------------|---------------|
| Reference | 850,000 | 0 | 3.5×10⁻⁶ | 3.5×10⁻⁶ |

## Reproduction

```bash
# Run statistical bounds calculation
python experiments/statistical_bounds.py

# Results location
experiments/results/miss_probability_bounds.csv
```

## Output Files

| File | Description |
|------|-------------|
| `miss_probability_bounds.csv` | Statistical bounds per scenario |
| `stability_empirical_results.csv` | Empirical stability metrics |

## Interpretation

The reported bounds provide **worst-case guarantees** rather than point estimates. A CP upper bound of 3.5×10⁻⁶ means: "With 95% confidence, the true miss probability is at most 3.5×10⁻⁶."

## Configuration

Seeds 0-19 used for reproducibility. See `configs/default_config.yaml`.
