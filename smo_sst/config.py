from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json
import math
import tomllib


BASELINE_METHODS = ("reactive_sst", "target_sst", "reactive_rrt", "target_rrt")
METHODS = BASELINE_METHODS + ("predictive_sst",)
LABELS = dict(zip(METHODS, ("Reactive SMO-SST", "Target-only SMO-SST",
                          "Reactive SMO-RRT", "Target-only SMO-RRT", "Predictive SMO-SST")))
POLICY_FAMILIES = dict(zip(METHODS, ("reactive", "target", "reactive", "target", "predictive")))


@dataclass(frozen=True)
class Config:
    # Values absent from the active manuscript come from the prior local protocol.
    dt: float = 0.1
    vmax: float = 1.0
    amax: float = 0.5
    omega_max: float = 0.3
    extent: float = 64.0
    target_x: float = 60.0
    target_y: float = 60.0
    adversaries: int = 8
    launcher_min: float = 8.0
    launcher_max: float = 56.0
    adv_speed: float = 1.5
    p_fire: float = 0.3
    rmin: float = 3.0
    rmax: float = 8.0
    capture_radius: float = 0.25
    lifetime: int = 54
    repulsion_radius: float = 8.0
    repulsion_gain: float = 2.0
    smoothing: float = 0.25
    predictive_horizon: float = 4.0
    predictive_guard_radius: float = 1.0
    predictive_velocity_epsilon: float = 0.1
    predictive_urgency_time: float = 2.0
    predictive_radial_max: float = 2.0
    predictive_tangential_max: float = 2.0
    predictive_baseline_probability: float = 0.2
    noise_position: float = 0.06
    noise_heading: float = 0.03
    noise_speed_fraction: float = 0.06
    noise_adversary: float = 0.01
    max_depth: int = 20
    tau_max: int = 100
    particles: int = 256
    rollout_threads: int = 4
    witness_radius: float = 0.025
    risk_resolution: float = 0.01
    mode_penalty: float = 1.0
    shortlist: int = 8
    index_block: int = 128
    distance: str = "bounded"
    environments: int = 30
    seconds: float = 120.0
    iterations: int | None = None
    seed: int = 2000
    evaluation_particles: int = 4096
    bootstrap_resamples: int = 10000
    confidence: float = 0.95
    checkpoint_seconds: float = 5.0

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, (int, float)) and not math.isfinite(v):
                raise ValueError(f"{f.name} must be finite")
        for key in ("dt", "vmax", "amax", "omega_max", "extent", "smoothing",
                    "witness_radius", "mode_penalty", "seconds", "checkpoint_seconds",
                    "predictive_horizon", "predictive_guard_radius", "predictive_velocity_epsilon",
                    "predictive_urgency_time"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in ("adversaries", "max_depth", "tau_max", "particles", "shortlist",
                    "index_block", "environments", "evaluation_particles", "bootstrap_resamples", "rollout_threads"):
            v = getattr(self, key)
            if not isinstance(v, int) or isinstance(v, bool) or v < (0 if key == "adversaries" else 1):
                raise ValueError(f"{key} must be a valid integer")
        if not isinstance(self.lifetime, int) or self.lifetime < 2:
            raise ValueError("lifetime must be an integer >= 2")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.iterations is not None and (not isinstance(self.iterations, int) or self.iterations < 1):
            raise ValueError("iterations must be null or a positive integer")
        if not 0 < self.confidence < 1 or not 0 < self.risk_resolution <= 1:
            raise ValueError("invalid confidence or risk resolution")
        if not 0 <= self.p_fire <= 1 or not 0 <= self.rmin < self.rmax:
            raise ValueError("invalid firing probability/ranges")
        if not 0 <= self.predictive_baseline_probability <= 1:
            raise ValueError("predictive_baseline_probability must be in [0,1]")
        if self.predictive_guard_radius < self.capture_radius:
            raise ValueError("predictive_guard_radius must be at least the capture radius")
        if self.capture_radius <= 0 or self.repulsion_radius <= self.capture_radius:
            raise ValueError("invalid capture/repulsion radii")
        if not 0 <= self.launcher_min < self.launcher_max <= self.extent:
            raise ValueError("invalid launcher bounds")
        for key in ("adv_speed", "repulsion_gain", "noise_position", "noise_heading",
                    "noise_speed_fraction", "noise_adversary", "predictive_radial_max",
                    "predictive_tangential_max"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if self.distance not in ("bounded", "exact"):
            raise ValueError("distance must be bounded or exact")

    @property
    def horizon(self):
        return self.max_depth * self.tau_max

    @property
    def duration(self):
        return self.horizon * self.dt

    @property
    def cost_bound(self):
        return 2.1 * self.duration

    @property
    def state_dim(self):
        return 6 + 4 * self.adversaries

    @property
    def metric_diameter(self):
        # Conservative diameter on the finite-horizon reachable box, unwrapped theta.
        p = self.horizon * (self.vmax * self.dt + self.noise_position)
        theta = self.horizon * (self.omega_max * self.dt + self.noise_heading)
        ego = math.sqrt(8 * p * p + 4 * theta * theta + 4 * self.vmax**2)
        travel = self.lifetime * (self.adv_speed * self.dt + self.noise_adversary)
        z = math.sqrt(self.adversaries * (2 * (self.launcher_max - self.launcher_min
                                             + 2 * travel)**2 + self.lifetime**2))
        return max(1.0, ego + z + self.mode_penalty)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        p = Path(path)
        data = json.loads(p.read_text()) if p.suffix == ".json" else tomllib.loads(p.read_text())
        return cls(**data)
