# Statistical Methods: Confidence Bounds for Rare Events

## Overview

This document describes the statistical methods used to quantify uncertainty in miss probability estimates, particularly when zero or few misses are observed.

## The Zero-Miss Problem

When running N trials and observing k=0 failures (misses), the point estimate is p̂ = 0/N = 0. However, this doesn't mean the true probability is exactly zero—it means we need confidence bounds.

## Method 1: Clopper-Pearson Exact Interval

The Clopper-Pearson method provides **exact** binomial confidence intervals.

### Formula (Upper Bound for k=0)

For k=0 successes out of N trials at confidence level (1-α):

```
p_upper = 1 - (α)^(1/N)
```

For 95% CI (α = 0.05):
```
p_upper = 1 - (0.05)^(1/N)
```

### Properties
- Exact (not approximate)
- Conservative (guarantees ≥ 95% coverage)
- Standard in reliability engineering

## Method 2: Rule of Three

A simple approximation for zero-failure scenarios:

```
p_upper ≈ 3/N  (95% confidence)
```

### Derivation
From the Poisson approximation to the binomial:
- P(k=0 | λ) = e^(-λ)
- Setting e^(-λ) = 0.05 gives λ ≈ 3
- Therefore p ≈ 3/N

### When to Use
- Quick sanity check
- Valid for large N and k=0
- Slightly conservative compared to CP

## Implementation

```python
from scipy import stats

def clopper_pearson_upper(k, n, alpha=0.05):
    """Compute upper bound of Clopper-Pearson interval."""
    if k == n:
        return 1.0
    return stats.beta.ppf(1 - alpha, k + 1, n - k)

def rule_of_three(n):
    """Rule-of-three approximation for k=0."""
    return 3.0 / n
```

## Example Calculation

For N = 850,000 arrivals with 0 misses:

| Method | Upper Bound (95%) |
|--------|-------------------|
| Clopper-Pearson | 3.53 × 10⁻⁶ |
| Rule of Three | 3.53 × 10⁻⁶ |

Both methods agree closely for large N.

## Interpretation

A Clopper-Pearson upper bound of 3.5×10⁻⁶ means:

> "With 95% confidence, the true miss probability is **at most** 3.5×10⁻⁶, which is 2,857× below the 1% URLLC threshold."

This is a **worst-case guarantee**, not a point estimate.

## References

1. Clopper, C. J.; Pearson, E. S. (1934). "The use of confidence or fiducial limits illustrated in the case of the binomial"
2. Jovanovic, B. D.; Levy, P. S. (1997). "A Look at the Rule of Three"
3. Brown, L. D.; Cai, T. T.; DasGupta, A. (2001). "Interval Estimation for a Binomial Proportion"

## Code Location

See `experiments/statistical_bounds.py` for implementation.
