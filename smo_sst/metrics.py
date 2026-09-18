import numpy as np
from scipy.stats import bootstrap
import warnings


def pareto_indices(points):
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if not np.isfinite(p).all():
        raise ValueError("objectives must be finite")
    order = np.lexsort((np.arange(len(p)), p[:, 1], p[:, 0]))
    keep, best = [], np.inf
    for i in order:
        if p[i, 1] < best:
            keep.append(int(i))
            best = p[i, 1]
    return np.array(keep, dtype=int)


def normalized_hypervolume(points, cost_reference):
    """2D minimization HV relative to (1,C_ref), divided by C_ref (ideal=(0,0))."""
    if not np.isfinite(cost_reference) or cost_reference <= 0:
        raise ValueError("positive finite cost reference required")
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if not np.isfinite(p).all() or (p < 0).any() or (p[:, 0] > 1).any():
        raise ValueError("invalid risk/cost values")
    p = p[(p[:, 0] < 1) & (p[:, 1] < cost_reference)]
    p = p[pareto_indices(p)]
    total, previous = 0., cost_reference
    for risk, cost in p:
        total += (1-risk)*(previous-cost)
        previous = cost
    return float(total/cost_reference)


def mean_interval(values, confidence=.95, resamples=10000, seed=0):
    """Environment-level BCa CI; explicit small-sample and degenerate behavior."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("nonempty finite scalar observations required")
    result = {"mean": float(x.mean()), "n": len(x), "confidence": confidence}
    if len(x) < 2:
        return result | {"low": None, "high": None, "method": "unavailable: one environment"}
    if np.all(x == x[0]):
        return result | {"low": float(x[0]), "high": float(x[0]), "method": "degenerate empirical bootstrap"}
    kwargs = dict(confidence_level=confidence, n_resamples=resamples,
                  random_state=np.random.default_rng(seed), batch=256)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = bootstrap((x,), np.mean, method="BCa", **kwargs)
    method = "BCa bootstrap over paired environments"
    lo, hi = fit.confidence_interval
    if not np.isfinite([lo, hi]).all():
        fit = bootstrap((x,), np.mean, method="percentile", **kwargs)
        lo, hi = fit.confidence_interval
        method = "percentile bootstrap fallback"
    return result | {"low": float(lo), "high": float(hi), "method": method}
