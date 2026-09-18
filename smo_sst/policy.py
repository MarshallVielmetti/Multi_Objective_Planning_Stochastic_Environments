"""Predictive policy parameters and the exact discrete-hazard survival table."""
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=8)
def survival_table(lifetime):
    """S[n,k] = P(still active after k transitions | active at count n)."""
    result = np.ones((lifetime+1, lifetime+1), dtype=np.float64)
    for age in range(lifetime+1):
        for k in range(1, lifetime+1):
            count = age+k-1
            # Enforce the analytic endpoint exactly despite floating-point cancellation.
            hazard = (1. if count >= lifetime-1 else
                      np.clip((count-.8*(lifetime-1))/(.2*(lifetime-1)), 0, 1))
            result[age,k] = result[age,k-1]*(1-hazard)
    result.flags.writeable = False
    return result


def sample_gains(rng, cfg):
    # A discrete atom retains the target-only branch with positive probability.
    if rng.random() < cfg.predictive_baseline_probability:
        return (0., 0.)
    return (float(rng.uniform(0, cfg.predictive_radial_max)),
            float(rng.uniform(-cfg.predictive_tangential_max, cfg.predictive_tangential_max)))


def validate_gains(gains, cfg):
    if (gains is None or len(gains) != 2 or not np.isfinite(gains).all()
            or not 0 <= gains[0] <= cfg.predictive_radial_max
            or not -cfg.predictive_tangential_max <= gains[1] <= cfg.predictive_tangential_max):
        raise ValueError("predictive policy requires finite radial/tangential gains within configured bounds")
    return float(gains[0]), float(gains[1])
