#!/usr/bin/env python3
"""Visualize the completed reactive SMO-SST delta_w sweep.

The script reads the saved study manifests, reports, and per-environment runs;
it never reruns the planner.  Run it from ``code/`` with

    uv run --locked python configs/reactive_sst_witness_sweep/visualize.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from smo_sst.visualize import save_figure, style  # noqa: E402


METHOD = "reactive_sst"
METRICS = ("nhv", "evaluation_nhv", "nodes", "iterations_per_second")
RUNTIME_ONLY_CONFIG = {"rollout_threads"}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing sweep artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in sweep artifact: {path}") from exc


def load_sweep(root: Path) -> list[dict]:
    """Load, validate, and sort the paired studies in a sweep directory."""
    root = root.resolve()
    studies = sorted({path.parent for path in root.glob("*/manifest.json")})
    if not studies:
        raise ValueError(f"no sweep studies with manifest.json found under {root}")

    rows: list[dict] = []
    baseline_config = None
    baseline_pairing = None
    seen_radii: set[float] = set()
    for study in studies:
        manifest = _read_json(study / "manifest.json")
        report = _read_json(study / "report.json")
        config = manifest.get("config", {})
        radius = float(config.get("witness_radius", 0.0))
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(f"{study}: witness_radius must be positive and finite")
        if radius in seen_radii:
            raise ValueError(f"duplicate witness_radius {radius:g}")
        seen_radii.add(radius)
        if manifest.get("methods") != [METHOD]:
            raise ValueError(f"{study}: expected exactly the {METHOD!r} method")

        comparison_config = {
            k: v for k, v in config.items()
            if k != "witness_radius" and k not in RUNTIME_ONLY_CONFIG
        }
        if baseline_config is None:
            baseline_config = comparison_config
        elif comparison_config != baseline_config:
            raise ValueError(f"{study}: configuration differs by more than witness_radius")

        environments = int(config.get("environments", 0))
        runs = []
        for environment in range(environments):
            run = _read_json(study / f"env-{environment:03d}" / f"{METHOD}.json")
            if run.get("environment") != environment or run.get("method") != METHOD:
                raise ValueError(f"{study}: invalid record for environment {environment}")
            runs.append(run)
        pairing = [
            (run.get("environment_seed"), run.get("planner_seed"), run.get("launchers"))
            for run in runs
        ]
        if baseline_pairing is None:
            baseline_pairing = pairing
        elif pairing != baseline_pairing:
            raise ValueError(f"{study}: environment seeds or launcher placements are not paired")

        try:
            summaries = report["methods"][METHOD]
            for metric in METRICS:
                summary = summaries[metric]
                raw_mean = float(np.mean([run[metric] for run in runs]))
                if not np.isclose(summary["mean"], raw_mean, rtol=1e-10, atol=1e-12):
                    raise ValueError(f"{study}: stale report mean for {metric}")
                if summary["n"] != environments:
                    raise ValueError(f"{study}: report uses the wrong environment count")
        except KeyError as exc:
            raise ValueError(f"{study}: report is missing {exc.args[0]!r}") from exc

        rows.append({
            "study": study,
            "radius": radius,
            "environments": environments,
            "rollout_threads": int(config.get("rollout_threads", 1)),
            "confidence": float(summaries["nhv"]["confidence"]),
            "summaries": summaries,
        })
    return sorted(rows, key=lambda row: row["radius"])


def _error(summary: dict) -> np.ndarray | None:
    if summary["low"] is None or summary["high"] is None:
        return None
    mean = float(summary["mean"])
    return np.array([[max(0.0, mean - summary["low"])],
                     [max(0.0, summary["high"] - mean)]])


def _write_summary(rows: list[dict], output: Path) -> Path:
    path = output / "delta_w_sweep.csv"
    fields = ["delta_w", "environments", "rollout_threads"]
    for metric in METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_ci_low", f"{metric}_ci_high"))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            record = {
                "delta_w": item["radius"],
                "environments": item["environments"],
                "rollout_threads": item["rollout_threads"],
            }
            for metric in METRICS:
                summary = item["summaries"][metric]
                record.update({
                    f"{metric}_mean": summary["mean"],
                    f"{metric}_ci_low": summary["low"],
                    f"{metric}_ci_high": summary["high"],
                })
            writer.writerow(record)
    return path


def visualize(root: Path, output: Path | None = None, dpi: int = 400,
              reference: float = 0.025) -> list[str]:
    if dpi < 1 or not np.isfinite(reference) or reference <= 0:
        raise ValueError("dpi and reference must be positive")
    rows = load_sweep(root)
    output = output.resolve() if output else root.resolve() / "figures"
    output.mkdir(parents=True, exist_ok=True)
    style()

    import matplotlib.pyplot as plt
    from matplotlib.ticker import EngFormatter

    radii = np.array([row["radius"] for row in rows])
    confidence = rows[0]["confidence"]
    if any(row["confidence"] != confidence for row in rows):
        raise ValueError("all sweep reports must use the same confidence level")

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), layout="constrained")
    quality = axes[0]
    series = (
        ("nhv", "Construction", "#147D92", "o", 0.975),
        ("evaluation_nhv", "Fresh evaluation", "#C85B63", "s", 1.025),
    )
    for metric, label, color, marker, shift in series:
        for x, row in zip(radii * shift, rows):
            summary = row["summaries"][metric]
            quality.errorbar(x, summary["mean"], yerr=_error(summary), fmt=marker,
                             color=color, capsize=3, ms=5, lw=1.4)
        quality.plot(radii * shift, [row["summaries"][metric]["mean"] for row in rows],
                     color=color, marker=marker, ms=0, lw=1.5, label=label)
    quality.set(ylabel="Normalized hypervolume", title="Solution quality")
    quality.legend(fontsize=8, loc="lower right")

    panels = ((axes[1], "nodes", "Retained nodes", "#7256A8", True),)
    for ax, metric, ylabel, color, logarithmic_y in panels:
        for x, row in zip(radii, rows):
            summary = row["summaries"][metric]
            ax.errorbar(x, summary["mean"], yerr=_error(summary), fmt="o", color=color,
                        capsize=3, ms=5, lw=1.4)
        ax.plot(radii, [row["summaries"][metric]["mean"] for row in rows],
                color=color, lw=1.5)
        ax.set(ylabel=ylabel, title="Pruning" if metric == "nodes" else "Throughput")
        if logarithmic_y:
            ax.set_yscale("log")
        ax.yaxis.set_major_formatter(EngFormatter(sep=""))

    throughput = axes[2]
    workers = sorted({row["rollout_threads"] for row in rows})
    markers = ("o", "s", "^", "D")
    for worker_count, marker in zip(workers, markers):
        group = [row for row in rows if row["rollout_threads"] == worker_count]
        group_x = np.array([row["radius"] for row in group])
        group_y = [row["summaries"]["iterations_per_second"]["mean"] for row in group]
        for x, row in zip(group_x, group):
            summary = row["summaries"]["iterations_per_second"]
            throughput.errorbar(x, summary["mean"], yerr=_error(summary), fmt=marker,
                                color="#E59635", capsize=3, ms=5, lw=1.4)
        throughput.plot(group_x, group_y, color="#E59635", marker=marker, ms=5, lw=1.5,
                        label=f"{worker_count} workers")
    throughput.set(ylabel="Planner iterations / s", title="Throughput")
    throughput.yaxis.set_major_formatter(EngFormatter(sep=""))
    if len(workers) > 1:
        throughput.legend(fontsize=8, loc="upper left")

    tick_labels = [f"{radius:g}" for radius in radii]
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(radii, tick_labels, rotation=35, ha="right")
        ax.set_xlabel(r"Witness radius $\delta_w$")
        ax.axvline(reference, color="#566773", ls="--", lw=1, alpha=0.75)
        ax.grid(which="major", alpha=0.65)
    fig.suptitle(
        f"Reactive SMO-SST witness-radius sweep · {rows[0]['environments']} paired environments · "
        f"{confidence:.0%} bootstrap CIs",
        fontsize=12,
    )
    paths = save_figure(fig, output, "delta_w_sweep", dpi)
    paths.append(str(_write_summary(rows, output)))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot quality, retained nodes, and throughput for a saved delta_w sweep"
    )
    parser.add_argument(
        "sweep",
        nargs="?",
        type=Path,
        default=CODE_DIR / "runs" / "reactive_sst_witness_sweep",
        help="directory containing one saved study per witness radius",
    )
    parser.add_argument("--output", type=Path, help="output directory (default: SWEEP/figures)")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--reference", type=float, default=0.025,
                        help="reference radius shown as a dashed vertical line")
    args = parser.parse_args(argv)
    try:
        print("\n".join(visualize(args.sweep, args.output, args.dpi, args.reference)))
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
