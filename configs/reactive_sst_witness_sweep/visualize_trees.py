#!/usr/bin/env python3
"""Plot final active-node trees across the reactive SMO-SST delta_w sweep.

Run from ``code/`` and pass the paired environment ID:

    uv run --locked python configs/reactive_sst_witness_sweep/visualize_trees.py 0

Each panel places a retained active node at its saved particle-mean ``(x, y)``
state.  An edge is drawn only when both the child and its saved parent are
active; inactive genealogical ancestors are deliberately omitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from smo_sst.config import Config  # noqa: E402
from smo_sst.visualize import save_figure, style  # noqa: E402


METHOD = "reactive_sst"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing sweep artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in sweep artifact: {path}") from exc


def _active_tree(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    try:
        with np.load(path) as archive:
            required = {"ids", "parents", "xy", "active"}
            if not required.issubset(archive.files):
                missing = ", ".join(sorted(required - set(archive.files)))
                raise ValueError(f"{path}: missing arrays: {missing}")
            ids = np.asarray(archive["ids"], dtype=np.int64)
            parents = np.asarray(archive["parents"], dtype=np.int64)
            xy = np.asarray(archive["xy"], dtype=float)
            active = np.asarray(archive["active"], dtype=bool)
    except FileNotFoundError as exc:
        raise ValueError(f"missing saved tree: {path}; rerun the sweep with --save-tree") from exc

    count = len(ids)
    if parents.shape != (count,) or active.shape != (count,) or xy.shape != (count, 2):
        raise ValueError(f"{path}: inconsistent tree array shapes")
    if count == 0 or len(np.unique(ids)) != count or not np.isfinite(xy).all():
        raise ValueError(f"{path}: invalid node IDs or mean states")

    order = np.argsort(ids)
    sorted_ids = ids[order]
    has_parent = parents >= 0
    parent_positions = np.searchsorted(sorted_ids, parents[has_parent])
    if np.any(parent_positions >= count) or np.any(sorted_ids[parent_positions] != parents[has_parent]):
        raise ValueError(f"{path}: a retained node references a missing parent")
    parent_indices = np.full(count, -1, dtype=np.int64)
    parent_indices[has_parent] = order[parent_positions]
    edge_mask = active & has_parent & active[parent_indices.clip(min=0)]
    children = np.flatnonzero(edge_mask)
    segments = np.stack((xy[parent_indices[children]], xy[children]), axis=1)
    return xy[active], segments, int(active.sum())


def load_environment(sweep: Path, environment: int) -> tuple[Config, np.ndarray, list[dict]]:
    sweep = sweep.resolve()
    studies = sorted({path.parent for path in sweep.glob("*/manifest.json")})
    if not studies:
        raise ValueError(f"no sweep studies with manifest.json found under {sweep}")

    rows = []
    baseline_config = None
    baseline_launchers = None
    resolved_config = None
    seen_radii: set[float] = set()
    for study in studies:
        manifest = _read_json(study / "manifest.json")
        config_dict = manifest.get("config", {})
        if manifest.get("methods") != [METHOD]:
            raise ValueError(f"{study}: expected exactly the {METHOD!r} method")
        environments = int(config_dict.get("environments", 0))
        if not 0 <= environment < environments:
            raise ValueError(
                f"environment must be in [0, {environments - 1}] for {study.name}; got {environment}"
            )
        radius = float(config_dict.get("witness_radius", 0.0))
        if not np.isfinite(radius) or radius <= 0 or radius in seen_radii:
            raise ValueError(f"{study}: witness_radius must be unique, positive, and finite")
        seen_radii.add(radius)

        run_path = study / f"env-{environment:03d}" / f"{METHOD}.json"
        if not run_path.exists():
            print(
                f"warning: skipping incomplete delta_w={radius:g}; "
                f"environment {environment} is not available",
                file=sys.stderr,
            )
            continue

        scientific_config = {
            key: value for key, value in config_dict.items()
            if key not in {"witness_radius", "rollout_threads"}
        }
        if baseline_config is None:
            baseline_config = scientific_config
            resolved_config = Config(**config_dict)
        elif scientific_config != baseline_config:
            raise ValueError(f"{study}: scientific configuration differs by more than witness_radius")

        run = _read_json(run_path)
        if run.get("environment") != environment or run.get("method") != METHOD:
            raise ValueError(f"{run_path}: environment or method metadata does not match")
        launchers = np.asarray(run.get("launchers"), dtype=float)
        if launchers.ndim != 2 or launchers.shape[1] != 2 or not np.isfinite(launchers).all():
            raise ValueError(f"{run_path}: invalid launcher positions")
        if baseline_launchers is None:
            baseline_launchers = launchers
        elif not np.array_equal(launchers, baseline_launchers):
            raise ValueError(f"{study}: launcher placements are not paired")

        points, edges, active_count = _active_tree(run_path.with_suffix(".tree.npz"))
        if active_count != run.get("active_nodes"):
            raise ValueError(f"{run_path}: saved tree and run disagree on active-node count")
        rows.append({
            "radius": radius,
            "points": points,
            "edges": edges,
            "active_count": active_count,
        })

    if not rows:
        raise ValueError(f"no completed tree artifacts found for environment {environment}")
    return resolved_config, baseline_launchers, sorted(rows, key=lambda row: row["radius"])


def _tree_style(active_count: int, edge_count: int) -> tuple[float, float, float]:
    point_size = float(np.clip(8000 / max(active_count, 1), 0.15, 5.0))
    edge_width = float(np.clip(3000 / max(edge_count, 1), 0.12, 0.6))
    edge_alpha = float(np.clip(12000 / max(edge_count, 1), 0.08, 0.35))
    return point_size, edge_width, edge_alpha


def visualize_trees(sweep: Path, environment: int, output: Path | None = None,
                    dpi: int = 400, columns: int = 4) -> list[str]:
    if environment < 0 or dpi < 1 or columns < 1:
        raise ValueError("environment must be nonnegative; dpi and columns must be positive")
    cfg, launchers, rows = load_environment(sweep, environment)
    output = output.resolve() if output else sweep.resolve() / "figures"
    style()

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    ncols = min(columns, len(rows))
    nrows = ceil(len(rows) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.25 * ncols, 3.15 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )

    all_points = np.concatenate([row["points"] for row in rows], axis=0)
    lower = min(0.0, float(all_points.min()))
    upper = max(float(cfg.extent), float(all_points.max()))
    padding = max(1.0, 0.02 * (upper - lower))
    limits = (lower - padding, upper + padding)

    for ax, row in zip(axes.flat, rows):
        ax.add_patch(Rectangle(
            (0, 0), cfg.extent, cfg.extent, fill=False, ls="--", lw=0.7,
            edgecolor="#93A4AE", zorder=0,
        ))
        for launcher in launchers:
            ax.add_patch(Circle(
                launcher, cfg.rmax, facecolor="#E59635", edgecolor="none",
                alpha=0.045, zorder=0,
            ))

        point_size, edge_width, edge_alpha = _tree_style(
            row["active_count"], len(row["edges"])
        )
        lines = LineCollection(
            row["edges"], colors="#4B7890", linewidths=edge_width,
            alpha=edge_alpha, rasterized=True, zorder=1,
        )
        ax.add_collection(lines)
        ax.scatter(
            row["points"][:, 0], row["points"][:, 1], s=point_size,
            color="#145E78", alpha=0.62, linewidths=0, rasterized=True, zorder=2,
        )
        ax.scatter(
            launchers[:, 0], launchers[:, 1], marker="^", s=18,
            color="#B9652D", edgecolors="white", linewidths=0.35, zorder=4,
        )
        ax.scatter([0], [0], marker="o", s=20, color="#2D8A83", zorder=5)
        ax.scatter(
            [cfg.target_x], [cfg.target_y], marker="*", s=55,
            color="#D59A20", edgecolors="white", linewidths=0.35, zorder=5,
        )
        ax.set(
            xlim=limits, ylim=limits, aspect="equal",
            title=rf"$\delta_w = {row['radius']:g}$",
        )
        ax.text(
            0.025, 0.025,
            f"{row['active_count']:,} nodes\n{len(row['edges']):,} active edges",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
            zorder=6,
        )
        ax.grid(alpha=0.25, lw=0.5)

    for ax in axes.flat[len(rows):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Mean ego position x")
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("Mean ego position y")

    legend = (
        Line2D([], [], color="#4B7890", lw=1.1, marker="o", markersize=3,
               markerfacecolor="#145E78", label="Active tree"),
        Line2D([], [], color="#B9652D", marker="^", ls="none", markersize=5,
               label="Launchers"),
        Line2D([], [], color="#2D8A83", marker="o", ls="none", markersize=4,
               label="Start"),
        Line2D([], [], color="#D59A20", marker="*", ls="none", markersize=7,
               label="Reference target"),
    )
    fig.suptitle(
        f"Final reactive SMO-SST active trees · paired environment {environment}",
        fontsize=12, y=0.985,
    )
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.945), fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.87,
                        wspace=0.10, hspace=0.19)
    stem = f"delta_w_trees_env_{environment:03d}"
    return save_figure(fig, output, stem, dpi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot the final active-node tree for every delta_w in a paired environment"
    )
    parser.add_argument("environment", type=int, help="zero-based paired environment ID")
    parser.add_argument(
        "--sweep", type=Path,
        default=CODE_DIR / "runs" / "reactive_sst_witness_sweep",
        help="directory containing one saved study per witness radius",
    )
    parser.add_argument("--output", type=Path, help="output directory (default: SWEEP/figures)")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--columns", type=int, default=4, help="number of grid columns")
    args = parser.parse_args(argv)
    try:
        print("\n".join(visualize_trees(
            args.sweep, args.environment, args.output, args.dpi, args.columns
        )))
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
