#!/usr/bin/env python3
"""Render a compact GIF for every saved point in one environment's Pareto portfolios."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle

from smo_sst.config import Config, LABELS, POLICY_FAMILIES
from smo_sst.experiment import seed_for
from smo_sst.kernel import Simulator
from smo_sst.visualize import style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one compact fresh-rollout GIF for every saved Pareto-front point."
    )
    parser.add_argument("study", type=Path, help="completed study directory")
    parser.add_argument("--environment", type=int, default=0, help="zero-based environment ID")
    parser.add_argument("--output", type=Path, default=None, help="gallery output directory")
    parser.add_argument("--methods", nargs="+", default=None, help="methods to render")
    parser.add_argument("--particles", type=int, default=16, help="qualitative replay particles")
    parser.add_argument("--dpi", type=int, default=72, help="GIF rendering resolution")
    parser.add_argument("--fps", type=int, default=12, help="GIF frames per second")
    parser.add_argument("--speed", type=float, default=20.0, help="simulated seconds per GIF second")
    parser.add_argument("--max-frames", type=int, default=48, help="maximum GIF frames")
    parser.add_argument("--overwrite", action="store_true", help="replace existing GIFs")
    return parser.parse_args()


def add_scene(ax, cfg: Config, launchers: np.ndarray) -> None:
    for xy in launchers:
        ax.add_patch(
            Circle(xy, cfg.rmax, facecolor="#EDBA83", edgecolor="#DEAE79", alpha=0.12, lw=0.5)
        )
    ax.add_patch(
        Rectangle((0, 0), cfg.extent, cfg.extent, fill=False, edgecolor="#9AABB4", lw=0.7)
    )
    ax.scatter(
        *launchers.T,
        marker="^",
        s=24,
        color="#BA6730",
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    ax.scatter([0], [0], marker="o", s=28, color="#147D92", zorder=5)
    ax.scatter(
        [cfg.target_x],
        [cfg.target_y],
        marker="*",
        s=75,
        color="#D79927",
        edgecolors="white",
        linewidths=0.35,
        zorder=5,
    )
    ax.set(xlim=(-3, cfg.extent + 3), ylim=(-3, cfg.extent + 3), aspect="equal")
    ax.set_xticks([0, cfg.extent / 2, cfg.extent])
    ax.set_yticks([0, cfg.extent / 2, cfg.extent])
    ax.tick_params(labelsize=6, length=2)
    ax.grid(alpha=0.25, linewidth=0.5)


def frame_steps(trace_length: int, cfg: Config, fps: int, speed: float, maximum: int) -> np.ndarray:
    stride = max(1, int(round(speed / (fps * cfg.dt))))
    frames = np.unique(np.r_[np.arange(0, trace_length, stride), trace_length - 1]).astype(int)
    if len(frames) > maximum:
        frames = np.unique(np.linspace(0, trace_length - 1, maximum).astype(int))
    return frames


def render_point(
    cfg: Config,
    method: str,
    point_index: int,
    total_points: int,
    front: dict,
    evaluation: dict,
    launchers: np.ndarray,
    evaluation_seed: int,
    output: Path,
    particles: int,
    dpi: int,
    fps: int,
    speed: float,
    max_frames: int,
    overwrite: bool,
) -> dict:
    method_output = output / method
    method_output.mkdir(parents=True, exist_ok=True)
    gif_path = method_output / f"point-{point_index:02d}.gif"
    replay_seed = seed_for(evaluation_seed, point_index, 71)
    sim = Simulator(cfg, family=POLICY_FAMILIES[method])
    _, trace = sim.replay(launchers, front["policy"], particles, replay_seed, trace=True)
    frames = frame_steps(len(trace), cfg, fps, speed, max_frames)

    if overwrite or not gif_path.exists():
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.08, top=0.84)
        add_scene(ax, cfg, launchers)
        paths = LineCollection([], colors="#147D92", linewidths=0.55, alpha=0.20)
        ax.add_collection(paths)
        ego = ax.scatter([], [], s=10, zorder=6)
        adversaries = ax.scatter([], [], s=18, marker="x", color="#C34D4D", lw=0.8, zorder=5)
        mean_line, = ax.plot([], [], color="#134E67", lw=1.3)
        failures = int(evaluation["failures"])
        evaluated_particles = int(evaluation["particles"])
        normalized_cost = evaluation["cost"] / cfg.cost_bound
        ax.set_title(
            f"{LABELS[method]} · point {point_index + 1}/{total_points}\n"
            f"fresh risk {failures}/{evaluated_particles} = {evaluation['risk']:.5f} · "
            f"cost / Cref = {normalized_cost:.4f}",
            loc="left",
            fontsize=7.2,
            pad=4,
        )

        def update(t: int):
            state = trace[t]
            ego.set_offsets(state[:, :2])
            captured = state[:, -2] > 0
            colors = np.empty((particles, 4))
            colors[~captured] = (0.07, 0.49, 0.58, 0.82)
            colors[captured] = (0.78, 0.25, 0.30, 0.86)
            ego.set_color(colors)
            hybrid = state[0, 4 : 4 + 4 * cfg.adversaries].reshape(cfg.adversaries, 4)
            adversaries.set_offsets(hybrid[hybrid[:, 3] == 1, :2])
            paths.set_segments([trace[: t + 1, i, :2] for i in range(particles)])
            mean_line.set_data(
                trace[: t + 1, :, 0].mean(axis=1), trace[: t + 1, :, 1].mean(axis=1)
            )
            return ego, adversaries, paths, mean_line

        animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
        temporary = gif_path.with_name(f".{gif_path.stem}.{os.getpid()}.tmp.gif")
        try:
            animation.save(temporary, writer=PillowWriter(fps=fps), dpi=dpi)
            os.replace(temporary, gif_path)
        finally:
            temporary.unlink(missing_ok=True)
            plt.close(fig)

    return {
        "method": method,
        "method_label": LABELS[method],
        "point": point_index,
        "node": front["node"],
        "depth": front["depth"],
        "fresh_risk": evaluation["risk"],
        "fresh_failures": evaluation["failures"],
        "fresh_particles": evaluation["particles"],
        "fresh_cost": evaluation["cost"],
        "fresh_cost_normalized": evaluation["cost"] / cfg.cost_bound,
        "construction_risk": front["risk"],
        "construction_cost": front["cost"],
        "construction_cost_normalized": front["cost"] / cfg.cost_bound,
        "animation_seed": replay_seed,
        "animation_particles": particles,
        "frame_steps": frames.tolist(),
        "gif": str(gif_path.relative_to(output)),
    }


def write_manifest(output: Path, rows: list[dict], metadata: dict) -> None:
    payload = dict(metadata)
    payload["points"] = rows
    (output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    columns = [
        "method",
        "method_label",
        "point",
        "node",
        "depth",
        "fresh_risk",
        "fresh_failures",
        "fresh_particles",
        "fresh_cost",
        "fresh_cost_normalized",
        "construction_risk",
        "construction_cost",
        "construction_cost_normalized",
        "animation_seed",
        "animation_particles",
        "gif",
    ]
    with (output / "points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.environment < 0:
        raise ValueError("environment must be nonnegative")
    if args.particles < 1 or args.dpi < 1 or args.fps < 1 or args.speed <= 0 or args.max_frames < 2:
        raise ValueError("positive rendering parameters required; max-frames must be at least 2")

    manifest = json.loads((args.study / "manifest.json").read_text())
    cfg = Config(**manifest["config"])
    if args.environment >= cfg.environments:
        raise ValueError(f"environment must be in [0,{cfg.environments - 1}]")
    methods = args.methods or manifest["methods"]
    unsupported = [method for method in methods if method not in manifest["methods"]]
    if unsupported:
        raise ValueError(f"methods not present in study manifest: {', '.join(unsupported)}")
    output = args.output or Path(f"figures/pareto_env_{args.environment:03d}")
    output.mkdir(parents=True, exist_ok=True)
    style()

    existing = []
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text()).get("points", [])
    by_key = {(row["method"], row["point"]): row for row in existing}

    for method in methods:
        run_path = args.study / f"env-{args.environment:03d}" / f"{method}.json"
        run = json.loads(run_path.read_text())
        if len(run["front"]) != len(run["evaluation"]):
            raise ValueError(f"front/evaluation length mismatch in {run_path}")
        launchers = np.asarray(run["launchers"], dtype=float).reshape(cfg.adversaries, 2)
        print(f"{LABELS[method]}: {len(run['front'])} points", flush=True)
        for index, (front, evaluation) in enumerate(zip(run["front"], run["evaluation"])):
            if front["node"] != evaluation["node"]:
                raise ValueError(f"node mismatch for {method} point {index}")
            row = render_point(
                cfg,
                method,
                index,
                len(run["front"]),
                front,
                evaluation,
                launchers,
                run["evaluation_seed"],
                output,
                args.particles,
                args.dpi,
                args.fps,
                args.speed,
                args.max_frames,
                args.overwrite,
            )
            by_key[(method, index)] = row
            print(f"  point {index + 1}/{len(run['front'])}: {row['gif']}", flush=True)

    method_order = {method: index for index, method in enumerate(manifest["methods"])}
    rows = sorted(by_key.values(), key=lambda row: (method_order[row["method"]], row["point"]))
    write_manifest(
        output,
        rows,
        {
            "source_study": str(args.study),
            "environment": args.environment,
            "cost_reference": cfg.cost_bound,
            "fresh_evaluation_particles": cfg.evaluation_particles,
            "animation": {
                "particles": args.particles,
                "fps": args.fps,
                "speed": args.speed,
                "max_frames": args.max_frames,
                "dpi": args.dpi,
                "adversary_display": "particle 0 only",
                "seed_rule": "seed_for(run.evaluation_seed, point_index, 71)",
            },
        },
    )
    print(f"wrote {len(rows)} gallery records to {output}", flush=True)


if __name__ == "__main__":
    main()
