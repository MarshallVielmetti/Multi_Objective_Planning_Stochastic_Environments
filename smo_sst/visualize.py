"""Publication figures and deterministic fresh-rollout animations, generated from saved runs."""
from pathlib import Path
import json
import os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent.parent / ".native" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from .config import Config, LABELS, METHODS, POLICY_FAMILIES
from .kernel import Simulator
from .metrics import pareto_indices


COLORS = {m: color for m, color in zip(METHODS, ("#147D92", "#E59635", "#7256A8", "#C85B63", "#398347"))}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.labelcolor": "#233340", "text.color": "#233340", "axes.edgecolor": "#9AABB4",
        "grid.color": "#D7E1E7", "grid.alpha": .6, "figure.facecolor": "white",
        "axes.facecolor": "#FAFCFD", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white"})


def save_figure(fig, output, name, dpi):
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "svg", "png"):
        p = output/f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(str(p))
    plt.close(fig)
    return paths


def figures(study, output=None, environment=0, dpi=400):
    style()
    study = Path(study)
    output = Path(output) if output else study/"figures"
    report = json.loads((study/"report.json").read_text())
    manifest = json.loads((study/"manifest.json").read_text())
    c = Config(**manifest["config"])
    methods = manifest["methods"]
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 2.5*len(methods)), 4.2), layout="constrained")
    for ax, metric, title in zip(axes, ("nhv", "evaluation_nhv"),
                                ("Construction front", "Fresh evaluation of selected policies")):
        for j, m in enumerate(methods):
            s = report["methods"][m][metric]
            err = None if s["low"] is None else np.array([[max(0,s["mean"]-s["low"])],
                                                          [max(0,s["high"]-s["mean"])]])
            ax.errorbar(j, s["mean"], yerr=err, fmt="o", color=COLORS[m], capsize=5, ms=7, lw=2)
        ax.set_xticks(range(len(methods)), [LABELS[m].replace(" ", "\n", 1) for m in methods], fontsize=8)
        ax.set(ylabel="Normalized hypervolume", title=title, ylim=(0, 1))
        ax.grid(axis="y")
    fig.suptitle(f"{c.environments} paired environments · {c.confidence:.0%} bootstrap confidence intervals", fontsize=12)
    paths += save_figure(fig, output, "nhv", dpi)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    for m in methods:
        runs = [json.loads((study/f"env-{i:03d}"/f"{m}.json").read_text()) for i in range(c.environments)]
        end = min(r["history"][-1]["seconds"] for r in runs)
        x = np.linspace(0, end, 150)
        curves = []
        for r in runs:
            t = np.array([p["seconds"] for p in r["history"]])
            h = np.array([p["nhv"] for p in r["history"]])
            curves.append(h[np.searchsorted(t, x, side="right").clip(1, len(t))-1])
        curves = np.array(curves)
        ax.plot(x, curves.mean(axis=0), label=LABELS[m], color=COLORS[m], lw=2)
        if len(curves) > 1:
            # Descriptive spread; CIs are in the nHV figure, not inferred from this band.
            lo, hi = np.quantile(curves, [.25, .75], axis=0)
            ax.fill_between(x, lo, hi, color=COLORS[m], alpha=.12)
    ax.set(xlabel="Planner wall time (s)", ylabel="Construction nHV", title="Search progress · bands show environment IQR")
    ax.grid(); ax.legend(fontsize=8, loc="lower right")
    paths += save_figure(fig, output, "progress", dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.4), layout="constrained")
    for m in methods:
        r = json.loads((study/f"env-{environment:03d}"/f"{m}.json").read_text())
        points = np.array([(v["risk"], v["cost"]/c.cost_bound) for v in r["evaluation"]])
        points = points[pareto_indices(points)]
        ax.plot(points[:, 0], points[:, 1], "o-", ms=4, lw=1.5, label=LABELS[m], color=COLORS[m])
    ax.set(xlabel="Capture probability", ylabel="Expected cost / reference cost",
           title=f"Fresh-evaluation Pareto fronts · environment {environment}", xlim=(-.015, 1.015))
    ax.grid(); ax.legend(fontsize=8)
    paths += save_figure(fig, output, f"fronts-{environment:03d}", dpi)
    return paths


def _scene(ax, cfg, launchers):
    for xy in launchers:
        ax.add_patch(Circle(xy, cfg.rmax, facecolor="#EDBA83", edgecolor="#DEAE79", alpha=.12, lw=.7))
    ax.scatter(*launchers.T, marker="^", s=60, color="#BA6730", edgecolors="white", lw=.6, label="Launchers", zorder=3)
    ax.scatter([0], [0], marker="o", s=65, color="#147D92", label="Start", zorder=5)
    ax.scatter([cfg.target_x], [cfg.target_y], marker="*", s=180, color="#D79927", edgecolors="white", label="Reference target", zorder=5)
    ax.set(xlabel="Position x", ylabel="Position y", aspect="equal")
    ax.grid(alpha=.3)


def trajectory(study, method="reactive_sst", environment=0, point=None, particles=32,
               seed=98765, output=None, dpi=200, fps=24, speed=8., max_frames=240, gif=True):
    if particles < 1 or dpi < 1 or fps < 1 or speed <= 0 or max_frames < 2:
        raise ValueError("positive rendering parameters required; max_frames >= 2")
    style()
    study = Path(study)
    cfg = Config(**json.loads((study/"manifest.json").read_text())["config"])
    run = json.loads((study/f"env-{environment:03d}"/f"{method}.json").read_text())
    if point is None:
        # Choose maximum single-policy rectangle area on fresh estimates.
        scores = [(1-p["risk"])*(1-p["cost"]/cfg.cost_bound) for p in run["evaluation"]]
        point = int(np.argmax(scores))
    if not 0 <= point < len(run["front"]):
        raise ValueError(f"point must be in [0,{len(run['front'])-1}]")
    policy = run["front"][point]["policy"]
    launchers = np.array(run["launchers"], dtype=float).reshape(cfg.adversaries, 2)
    sim = Simulator(cfg, family=POLICY_FAMILIES[method])
    _, trace = sim.replay(launchers, policy, particles, seed, trace=True)
    output = Path(output) if output else study/"figures"
    output.mkdir(parents=True, exist_ok=True)
    stem = f"trajectory-{environment:03d}-{method}-point{point}"
    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    _scene(ax, cfg, launchers)
    lo = min(-3., float(trace[:, :, :2].min())-2)
    hi = max(cfg.extent+3., float(trace[:, :, :2].max())+2)
    ax.set(xlim=(lo, hi), ylim=(lo, hi))
    for i in range(particles):
        color = "#C85B63" if trace[-1, i, -2] else "#147D92"
        ax.plot(trace[:, i, 0], trace[:, i, 1], color=color, alpha=.22, lw=.7)
    ax.plot(trace[:, :, 0].mean(axis=1), trace[:, :, 1].mean(axis=1), color="#134E67", lw=2, label="Particle mean")
    ax.set_title(f"{LABELS[method]} · fresh policy rollouts", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    paths = save_figure(fig, output, stem, dpi)
    if not gif:
        return paths
    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    _scene(ax, cfg, launchers)
    ax.set(xlim=(lo, hi), ylim=(lo, hi))
    paths_artist = LineCollection([], colors="#147D92", linewidths=.7, alpha=.24)
    ax.add_collection(paths_artist)
    ego = ax.scatter([], [], s=16, zorder=6)
    adv = ax.scatter([], [], s=32, marker="x", color="#C34D4D", lw=1.1, zorder=5)
    mean_line, = ax.plot([], [], color="#134E67", lw=2)
    title = ax.set_title("", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    stride = max(1, int(round(speed/(fps*cfg.dt))))
    frames = np.unique(np.r_[np.arange(0, len(trace), stride), len(trace)-1]).astype(int)
    if len(frames) > max_frames:
        frames = np.unique(np.linspace(0, len(trace)-1, max_frames).astype(int))
    def update(t):
        s = trace[t]
        ego.set_offsets(s[:, :2])
        ego.set_color(np.where(s[:, -2, None] > 0, np.array([[.78,.25,.30, .8]]),
                              np.array([[.07,.49,.58,.8]])))
        # One representative particle's hybrid adversaries, clearly identified in metadata.
        z = s[0, 4:4+4*cfg.adversaries].reshape(cfg.adversaries, 4)
        adv.set_offsets(z[z[:, 3] == 1, :2])
        paths_artist.set_segments([trace[:t+1, i, :2] for i in range(particles)])
        mean_line.set_data(trace[:t+1, :, 0].mean(axis=1), trace[:t+1, :, 1].mean(axis=1))
        title.set_text(f"{LABELS[method]}  |  t = {t*cfg.dt:.1f} s\n"
                       f"Captured: {int(s[:, -2].sum())}/{particles} · red ×: particle 1 adversaries")
        return ego, adv, paths_artist, mean_line, title
    anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)
    gif_path = output/f"{stem}.gif"
    anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    (output/f"{stem}.json").write_text(json.dumps({"method": method, "environment": environment,
        "front_point": point, "node": run["front"][point]["node"], "seed": seed, "particles": particles,
        "frame_steps": frames.tolist(), "fps": fps, "dt": cfg.dt, "policy": policy,
        "adversary_display": "particle 0 only", "fresh_rollouts": True}, indent=2)+"\n")
    return paths+[str(gif_path)]
