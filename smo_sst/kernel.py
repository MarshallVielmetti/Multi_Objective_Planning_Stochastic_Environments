"""Small ctypes boundary around the compiled, fused rollout and transport kernels."""
from functools import lru_cache
from pathlib import Path
import ctypes as ct
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import numpy as np
from .config import Config
from .policy import survival_table, validate_gains


@lru_cache(maxsize=1)
def library():
    source = Path(__file__).with_name("native.cpp")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        raise RuntimeError("A C++17 compiler is required: install clang++ or g++ and set CXX")
    flags = ["-O3", "-std=c++17", "-shared", "-fPIC", "-pthread"]
    key = hashlib.sha256(source.read_bytes() + str((compiler, flags, platform.machine(),
                                                  platform.system())).encode()).hexdigest()[:16]
    cache = Path(os.environ.get("SMO_NATIVE_CACHE", source.parent.parent / ".native"))
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"kernel-{key}.so"
    if not target.exists():
        # Compile to a unique temporary file; atomic publication supports concurrent workers.
        fd, tmp = tempfile.mkstemp(prefix="build-", suffix=".so", dir=cache)
        os.close(fd)
        try:
            result = subprocess.run([compiler, *flags, str(source), "-o", tmp],
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"Native kernel build failed:\n{result.stderr}")
            os.replace(tmp, target)
        finally:
            Path(tmp).unlink(missing_ok=True)
    lib = ct.CDLL(str(target))
    lib.set_threads.argtypes = [ct.c_int]
    lib.set_threads.restype = None
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    u64 = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
    lib.propagate.argtypes = [f32, f32, u64, ct.c_int, ct.c_int, ct.c_int,
                             ct.c_double, ct.c_double, f64, f64, ct.c_double, ct.c_double]
    lib.propagate.restype = None
    lib.controls.argtypes = [f32, ct.c_int, ct.c_int, ct.c_double, ct.c_double,
                             f64, f64, ct.c_double, ct.c_double, f64]
    lib.controls.restype = None
    lib.coupling_costs.argtypes = [f32, f32, i64, i64, ct.c_int, ct.c_int, ct.c_int,
                                  ct.c_double, ct.c_double, f64]
    lib.coupling_costs.restype = None
    lib.cost_matrix.argtypes = [f32, f32, ct.c_int, ct.c_int, ct.c_double, ct.c_double, f64]
    lib.cost_matrix.restype = None
    lib.noise_samples.argtypes = [ct.c_uint64, f64, ct.c_int]
    lib.noise_samples.restype = None
    return lib


class Simulator:
    def __init__(self, config: Config, reactive=True, *, family=None):
        self.cfg = c = config
        self.family = family if family is not None else ("reactive" if reactive else "target")
        if self.family not in ("target", "reactive", "predictive"):
            raise ValueError(f"unknown policy family: {self.family}")
        self.lib = library()
        self.lib.set_threads(c.rollout_threads)
        self.survival = survival_table(c.lifetime)
        self.constants = np.array([c.dt, c.vmax, c.amax, c.omega_max, c.adv_speed,
            c.p_fire, c.rmin, c.rmax, c.capture_radius, c.lifetime, c.repulsion_radius,
            c.repulsion_gain if self.family == "reactive" else 0., c.smoothing, c.noise_position,
            c.noise_heading, c.noise_speed_fraction*c.vmax, c.noise_adversary,
            ("target", "reactive", "predictive").index(self.family), c.predictive_horizon,
            c.predictive_guard_radius, c.predictive_velocity_epsilon, c.predictive_urgency_time])

    def initial(self, launchers, particles=None):
        c = self.cfg
        launchers = np.asarray(launchers, dtype=float)
        if launchers.shape != (c.adversaries, 2) or not np.isfinite(launchers).all():
            raise ValueError("launchers must have shape (adversaries, 2) and finite coordinates")
        n = c.particles if particles is None else particles
        if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 1:
            raise ValueError("particles must be a positive integer")
        cloud = np.zeros((n, c.state_dim), dtype=np.float32)
        for j, xy in enumerate(launchers):
            cloud[:, 4+4*j:6+4*j] = xy
        return cloud

    @staticmethod
    def rng_states(seed, count):
        return np.random.SeedSequence(int(seed)).generate_state(count, dtype=np.uint64)

    def _validate_input(self, cloud, waypoint, gains):
        c = self.cfg
        if (cloud.dtype != np.float32 or cloud.ndim != 2 or cloud.shape[1] != c.state_dim
                or not cloud.flags.c_contiguous or cloud.shape[0] < 1):
            raise ValueError("invalid particle cloud")
        if len(waypoint) != 2 or not np.isfinite(waypoint).all():
            raise ValueError("waypoint must have two finite coordinates")
        if self.family == "predictive":
            return validate_gains(gains, c)
        if gains is not None:
            raise ValueError("avoidance gains are only valid for the predictive policy family")
        return 0., 0.

    def controls(self, cloud, waypoint, gains=None):
        """Return [acceleration, turn rate] from the same controller used by rollouts."""
        radial, tangential = self._validate_input(cloud, waypoint, gains)
        out = np.empty((len(cloud), 2))
        self.lib.controls(cloud, len(cloud), self.cfg.adversaries, float(waypoint[0]),
                          float(waypoint[1]), self.constants, self.survival, radial, tangential, out)
        return out

    def propagate(self, cloud, waypoint, steps, states, gains=None):
        c = self.cfg
        radial, tangential = self._validate_input(cloud, waypoint, gains)
        if (states.dtype != np.uint64 or states.shape != (len(cloud),)
                or not states.flags.c_contiguous or not states.flags.writeable):
            raise ValueError("invalid particle RNG states")
        if not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        out = np.empty_like(cloud)
        self.lib.propagate(cloud, out, states, len(cloud), c.adversaries, steps,
                           float(waypoint[0]), float(waypoint[1]), self.constants,
                           self.survival, radial, tangential)
        return out

    def objective_samples(self, cloud):
        c = self.cfg
        terminal = c.duration * np.clip(np.hypot(cloud[:, 0].astype(float)-c.target_x,
                     cloud[:, 1].astype(float)-c.target_y)/(c.extent*np.sqrt(2)), 0, 1)
        return cloud[:, -2].astype(float), cloud[:, -1].astype(float) + terminal

    def objective(self, cloud):
        q, cost = self.objective_samples(cloud)
        return float(q.mean()), float(cost.mean())

    def replay(self, launchers, policy, particles, seed, trace=False):
        if len(policy) > self.cfg.max_depth:
            raise ValueError("policy exceeds configured maximum depth")
        edge_gains = []
        for edge in policy:
            steps, q = edge["steps"], edge["waypoint"]
            if (not isinstance(steps, (int, np.integer)) or isinstance(steps, (bool, np.bool_))
                    or not 1 <= steps <= self.cfg.tau_max or len(q) != 2
                    or not np.isfinite(q).all() or np.min(q) < 0 or np.max(q) > self.cfg.extent):
                raise ValueError("policy edges require a sampled-domain waypoint and integer duration in 1..tau_max")
            if self.family == "predictive":
                if "radial_gain" not in edge or "tangential_gain" not in edge:
                    raise ValueError("predictive policy edges must record radial_gain and tangential_gain")
                edge_gains.append(validate_gains((edge["radial_gain"], edge["tangential_gain"]), self.cfg))
            else:
                if "radial_gain" in edge or "tangential_gain" in edge:
                    raise ValueError("predictive policy gains cannot be replayed under a baseline family")
                edge_gains.append(None)
        cloud = self.initial(launchers, particles)
        states = self.rng_states(seed, particles)
        frames = [cloud.copy()] if trace else None
        for edge, gains in zip(policy, edge_gains):
            q, steps = edge["waypoint"], int(edge["steps"])
            if trace:
                for _ in range(steps):
                    cloud = self.propagate(cloud, q, 1, states, gains)
                    frames.append(cloud.copy())
            else:
                cloud = self.propagate(cloud, q, steps, states, gains)
        return (cloud, np.stack(frames)) if trace else cloud
