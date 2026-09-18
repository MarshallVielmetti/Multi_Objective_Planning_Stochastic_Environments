from dataclasses import dataclass, field
import time
import numpy as np
from .kernel import Simulator
from .metrics import normalized_hypervolume, pareto_indices
from .transport import WitnessIndex, signature, lower_bound, upper_bound, exact_distance
from .config import METHODS, POLICY_FAMILIES
from .policy import sample_gains


@dataclass(slots=True, eq=False)
class Node:
    id: int
    depth: int
    time: int
    risk: float
    cost: float
    cloud: np.ndarray | None
    parent: "Node | None" = None
    waypoint: tuple = (0., 0.)
    steps: int = 0
    children: int = 0
    active: bool = True
    xy: tuple = (0., 0.)
    gains: tuple | None = None

    def policy(self):
        edges, node = [], self
        while node.parent is not None:
            edge = {"waypoint": list(node.waypoint), "steps": node.steps}
            if node.gains is not None:
                edge.update(radial_gain=node.gains[0], tangential_gain=node.gains[1])
            edges.append(edge)
            node = node.parent
        return edges[::-1]


@dataclass(slots=True)
class Witness:
    center: object
    representatives: list = field(default_factory=list)


def sparse_pareto_update(nodes, resolution):
    # Strict Pareto first, then per-risk-cell cost/risk/creation tie breaking.
    # Sorting by creation guarantees incumbent retention on identical objectives.
    ordered = sorted(nodes, key=lambda n: n.id)
    front = [ordered[i] for i in pareto_indices([(n.risk, n.cost) for n in ordered])]
    cells = {}
    for n in front:
        cell = int(np.floor(n.risk/resolution))
        key = (n.cost, n.risk, n.id)
        if cell not in cells or key < (cells[cell].cost, cells[cell].risk, cells[cell].id):
            cells[cell] = n
    return list(cells.values())


class Planner:
    def __init__(self, cfg, launchers, method, seed):
        if method not in METHODS:
            raise ValueError(f"unknown method {method}")
        self.cfg, self.method = cfg, method
        self.launchers = np.asarray(launchers)
        self.sim = Simulator(cfg, family=POLICY_FAMILIES[method])
        # Separate proposal and process streams: simulator RNG consumption cannot alter proposals.
        proposal_seed, process_seed, gain_seed = np.random.SeedSequence(seed).spawn(3)
        self.rng = np.random.default_rng(proposal_seed)
        self.process_rng = np.random.default_rng(process_seed)
        self.gain_rng = np.random.default_rng(gain_seed)
        cloud = self.sim.initial(launchers)
        q, c = self.sim.objective(cloud)
        root = Node(0, 0, 0, q, c, cloud)
        self.nodes = {0: root}
        self.expandable = [root]
        self.locations = {0: 0}
        self.witnesses = {}
        self.indices = {}
        self.pruning = method.endswith("sst")
        self.iterations = 0
        self.stats = dict(accepted=0, rejected=0, pruned=0, deleted=0, bound_checks=0,
                          lower_bound_rejections=0, omitted_witnesses=0,
                          particle_steps=0, rollout_seconds=0., witness_seconds=0.)

    def _remove_expandable(self, node):
        loc = self.locations.pop(node.id, None)
        if loc is not None:
            last = self.expandable.pop()
            if loc < len(self.expandable):
                self.expandable[loc] = last
                self.locations[last.id] = loc

    def _prune(self, node):
        node.active = False
        self._remove_expandable(node)
        # Inactive ancestors retain genealogy only. Witness centers own their cloud references.
        node.cloud = None
        self.stats["pruned"] += 1
        while node.parent is not None and not node.active and node.children == 0:
            del self.nodes[node.id]
            self.stats["deleted"] += 1
            parent = node.parent
            parent.children -= 1
            node = parent

    def _witness(self, child):
        c = self.cfg
        sig = signature(child.cloud, c)
        depth = child.depth
        witnesses = self.witnesses.setdefault(depth, [])
        index = self.indices.setdefault(depth, WitnessIndex(c.index_block))
        ids = list(range(len(witnesses))) if c.distance == "exact" else index.query(sig.embedding, c.shortlist)
        self.stats["omitted_witnesses"] += len(witnesses)-len(ids)
        best, best_distance = None, c.witness_radius
        for j in ids:
            w = witnesses[j]
            if c.distance == "bounded" and lower_bound(sig, w.center, c) > best_distance:
                self.stats["lower_bound_rejections"] += 1
                continue
            self.stats["bound_checks"] += 1
            d = (exact_distance(sig.cloud, w.center.cloud, c) if c.distance == "exact"
                 else upper_bound(sig, w.center, c))
            if d <= best_distance:
                best, best_distance = w, d
        if best is None:
            best = Witness(sig)
            witnesses.append(best)
            index.add(sig.embedding)
        return best

    def step(self):
        c = self.cfg
        self.iterations += 1
        parent = self.expandable[int(self.rng.integers(len(self.expandable)))]
        waypoint = tuple(self.rng.uniform(0, c.extent, 2))
        steps = int(self.rng.integers(1, c.tau_max+1))
        gains = sample_gains(self.gain_rng, c) if self.sim.family == "predictive" else None
        states = self.process_rng.integers(0, np.iinfo(np.uint64).max,
                                           size=c.particles, dtype=np.uint64)
        start = time.perf_counter()
        cloud = self.sim.propagate(parent.cloud, waypoint, steps, states, gains)
        self.stats["rollout_seconds"] += time.perf_counter()-start
        self.stats["particle_steps"] += c.particles*steps
        risk, cost = self.sim.objective(cloud)
        child = Node(self.iterations, parent.depth+1, parent.time+steps, risk, cost,
                     cloud, parent, waypoint, steps, xy=tuple(cloud[:, :2].mean(axis=0)), gains=gains)
        removed = []
        accepted = True
        if self.pruning:
            start = time.perf_counter()
            witness = self._witness(child)
            old = witness.representatives
            witness.representatives = sparse_pareto_update(old+[child], c.risk_resolution)
            accepted = child in witness.representatives
            removed = [n for n in old if n not in witness.representatives]
            self.stats["witness_seconds"] += time.perf_counter()-start
        if accepted:
            self.nodes[child.id] = child
            parent.children += 1
            if child.depth < c.max_depth:
                self.locations[child.id] = len(self.expandable)
                self.expandable.append(child)
            self.stats["accepted"] += 1
        else:
            self.stats["rejected"] += 1
        for node in removed:
            self._prune(node)
        return child, accepted

    def front(self):
        nodes = list(self.nodes.values())
        ix = pareto_indices([(n.risk, n.cost) for n in nodes])
        return [nodes[i] for i in ix]

    def snapshot(self, elapsed):
        points = [(n.risk, n.cost) for n in self.front()]
        return {"seconds": elapsed, "iterations": self.iterations, "nodes": len(self.nodes),
                "nhv": normalized_hypervolume(points, self.cfg.cost_bound)}

    def run(self):
        c = self.cfg
        # Compilation and initialization have completed before this wall-clock starts.
        start = time.perf_counter()
        history = [self.snapshot(0.)]
        next_checkpoint = c.checkpoint_seconds
        while True:
            elapsed = time.perf_counter()-start
            if c.iterations is None and elapsed >= c.seconds:
                break
            if c.iterations is not None and self.iterations >= c.iterations:
                break
            self.step()
            elapsed = time.perf_counter()-start
            if elapsed >= next_checkpoint:
                history.append(self.snapshot(elapsed))
                next_checkpoint = elapsed+c.checkpoint_seconds
        elapsed = time.perf_counter()-start
        history.append(self.snapshot(elapsed))
        front = self.front()
        # Report actual retained clouds, including fixed centers, without double-counting aliases.
        clouds = {id(n.cloud): n.cloud for n in self.nodes.values() if n.cloud is not None}
        for group in self.witnesses.values():
            for w in group:
                clouds[id(w.center.cloud)] = w.center.cloud
        result = {"method": self.method, "policy_family": self.sim.family, "elapsed_seconds": elapsed,
            "iterations": self.iterations, "iterations_per_second": self.iterations/elapsed,
            "nodes": len(self.nodes), "active_nodes": sum(n.active for n in self.nodes.values()),
            "witnesses": sum(map(len, self.witnesses.values())) + int(self.pruning),
            "cloud_memory_bytes": sum(a.nbytes for a in clouds.values()),
            "nhv": history[-1]["nhv"], "history": history, "metrics": self.stats.copy(),
            "front": [{"node": n.id, "risk": n.risk, "cost": n.cost,
                       "depth": n.depth, "time_steps": n.time, "policy": n.policy()} for n in front]}
        return result

    def tree_arrays(self):
        nodes = list(self.nodes.values())
        return dict(ids=np.array([n.id for n in nodes]),
                    parents=np.array([n.parent.id if n.parent else -1 for n in nodes]),
                    xy=np.array([n.xy for n in nodes]),
                    active=np.array([n.active for n in nodes]))
