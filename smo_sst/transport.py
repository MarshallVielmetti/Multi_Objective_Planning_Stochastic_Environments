"""Conservative empirical W1 tests with an append-only logarithmic KD-tree forest."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .kernel import library


@dataclass(slots=True)
class Signature:
    cloud: np.ndarray
    mean_ego: np.ndarray
    mean_env: np.ndarray
    modes: np.ndarray
    risk: float
    orders: np.ndarray
    embedding: np.ndarray


def _validate_cloud(cloud, cfg):
    if (not isinstance(cloud, np.ndarray) or cloud.dtype != np.float32 or cloud.ndim != 2
            or cloud.shape[0] == 0 or cloud.shape[1] != cfg.state_dim
            or not cloud.flags.c_contiguous):
        raise ValueError("cloud must be a nonempty contiguous float32 array with the configured state dimension")


def signature(cloud, cfg):
    _validate_cloud(cloud, cfg)
    a = cfg.adversaries
    z = cloud[:, 4:4+4*a].reshape(len(cloud), a, 4)
    ego = cloud[:, :4].mean(axis=0, dtype=float)
    env = z[:, :, :3].mean(axis=0, dtype=float).ravel()
    modes = np.stack([(z[:, :, 3] == k).mean(axis=0) for k in range(3)], axis=1)
    risk = float(cloud[:, -2].mean())
    # Each row defines a complete permutation, including failed particles.
    projections = (cloud[:, 0], cloud[:, 1], cloud[:, 0]+cloud[:, 1],
                   cloud[:, 0]-cloud[:, 1])
    orders = np.ascontiguousarray([np.lexsort((p, cloud[:, -2])) for p in projections], dtype=np.int64)
    # Retrieval only; this embedding is never used to authorize a merge.
    emb = np.array([ego[0]/cfg.metric_diameter, ego[1]/cfg.metric_diameter, risk])
    return Signature(cloud, ego, env, modes, risk, orders, emb)


def lower_bound(a, b, cfg):
    # Jensen for each Euclidean term, total variation of a labeled-mode marginal,
    # and the exact W1 distance of Bernoulli failure marginals.
    tv = np.max(np.abs(a.modes-b.modes).sum(axis=1)/2, initial=0.)
    return ((np.linalg.norm(a.mean_ego-b.mean_ego)+np.linalg.norm(a.mean_env-b.mean_env)
             + cfg.mode_penalty*tv)/cfg.metric_diameter + abs(a.risk-b.risk))


def upper_bound(a, b, cfg):
    _validate_cloud(a.cloud, cfg)
    _validate_cloud(b.cloud, cfg)
    if a.cloud.shape != b.cloud.shape:
        raise ValueError("transport requires equal-size, uniformly weighted clouds")
    out = np.empty(len(a.orders))
    library().coupling_costs(a.cloud, b.cloud, a.orders, b.orders, len(a.cloud),
        cfg.adversaries, len(a.orders), cfg.metric_diameter, cfg.mode_penalty, out)
    return float(out.min())


def exact_distance(a, b, cfg):
    _validate_cloud(a, cfg)
    _validate_cloud(b, cfg)
    if a.shape != b.shape or a.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError("exact transport requires equal-size float32 clouds")
    cost = np.empty((len(a), len(b)))
    library().cost_matrix(np.ascontiguousarray(a), np.ascontiguousarray(b), len(a),
                          cfg.adversaries, cfg.metric_diameter, cfg.mode_penalty, cost)
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


class WitnessIndex:
    """Binary-merged immutable blocks: O(W log^2 W) total build, no full-tree rebuilds."""
    def __init__(self, block=128):
        self.block = block
        self.pending = []
        self.levels = []
        self.size = 0

    def add(self, embedding):
        self.pending.append((self.size, embedding))
        self.size += 1
        if len(self.pending) < self.block:
            return
        ids = np.array([v[0] for v in self.pending])
        points = np.array([v[1] for v in self.pending])
        self.pending.clear()
        level = 0
        while level < len(self.levels) and self.levels[level] is not None:
            old_ids, old_points, _ = self.levels[level]
            ids = np.concatenate((old_ids, ids))
            points = np.concatenate((old_points, points))
            self.levels[level] = None
            level += 1
        entry = (ids, points, cKDTree(points))
        if level == len(self.levels):
            self.levels.append(entry)
        else:
            self.levels[level] = entry

    def query(self, point, k):
        candidates = []
        for entry in self.levels:
            if entry is not None:
                ids, _, tree = entry
                d, ix = tree.query(point, k=min(k, len(ids)))
                candidates.extend(zip(np.atleast_1d(d), ids[np.atleast_1d(ix)]))
        if self.pending:
            ids, points = zip(*self.pending)
            d = np.linalg.norm(np.asarray(points)-point, axis=1)
            ix = np.argsort(d, kind="stable")[:k]
            candidates.extend((d[j], ids[j]) for j in ix)
        return [int(i) for _, i in sorted(candidates)[:k]]
