"""Paired wall-clock experiment, resumable artifacts, and independent evaluation."""
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import numpy as np
from scipy.stats import beta
from .config import Config, METHODS, BASELINE_METHODS, LABELS, POLICY_FAMILIES
from .kernel import Simulator, library
from .metrics import mean_interval, normalized_hypervolume
from .planner import Planner


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def seed_for(seed, environment, stream):
    return int(np.random.SeedSequence([seed, environment, stream]).generate_state(1)[0])


def provenance():
    folder = Path(__file__).parent
    digest = hashlib.sha256()
    for p in sorted([*folder.glob("*.py"), *folder.glob("*.cpp")]):
        digest.update(p.name.encode()+p.read_bytes())
    return {"source_sha256": digest.hexdigest(), "python": sys.version,
            "platform": platform.platform(), "machine": platform.machine(),
            "dependencies": {p: importlib.metadata.version(p) for p in
                             ("numpy", "scipy", "matplotlib", "pillow")},
            "compiler": os.environ.get("CXX", "c++"),
            "build_flags": "-O3 -std=c++17 -shared -fPIC -pthread"}


def evaluate(result, cfg, launchers, seed):
    """Fresh IID rollouts after freezing the selected retained construction front."""
    sim = Simulator(cfg, family=POLICY_FAMILIES[result["method"]])
    evaluated = []
    # Simultaneous intervals conditional on this frozen finite policy portfolio.
    alpha = (1-cfg.confidence)/(2*max(1, len(result["front"])))
    n = cfg.evaluation_particles
    for index, point in enumerate(result["front"]):
        cloud = sim.replay(launchers, point["policy"], n, seed_for(seed, index, 71))
        risks, costs = sim.objective_samples(cloud)
        failures = int(risks.sum())
        risk_lo = 0. if failures == 0 else float(beta.ppf(alpha/2, failures, n-failures+1))
        risk_hi = 1. if failures == n else float(beta.ppf(1-alpha/2, failures+1, n-failures))
        radius = cfg.cost_bound*np.sqrt(np.log(2/alpha)/(2*n))
        evaluated.append({"node": point["node"], "risk": failures/n, "cost": float(costs.mean()),
            "risk_interval": [risk_lo, risk_hi],
            "cost_interval": [max(0., float(costs.mean())-radius),
                              min(cfg.cost_bound, float(costs.mean())+radius)],
            "failures": failures, "particles": n})
    result["evaluation"] = evaluated
    result["evaluation_nhv"] = normalized_hypervolume(
        [(p["risk"], p["cost"]) for p in evaluated], cfg.cost_bound)
    lo = normalized_hypervolume([(p["risk_interval"][1], p["cost_interval"][1])
                                for p in evaluated], cfg.cost_bound)
    hi = normalized_hypervolume([(p["risk_interval"][0], p["cost_interval"][0])
                                for p in evaluated], cfg.cost_bound)
    result["evaluation_nhv_interval"] = {"low": lo, "high": hi, "confidence": cfg.confidence,
        "method": "simultaneous Clopper-Pearson risk / Hoeffding cost; frozen portfolio only"}


def run_study(cfg, output, methods=BASELINE_METHODS, resume=False, save_tree=False):
    output = Path(output)
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("provide distinct supported methods")
    library()  # Build before any planner's timing starts.
    manifest = {"schema_version": 2, "config": cfg.to_dict(), "methods": list(methods),
                "policy_families": {m: POLICY_FAMILIES[m] for m in methods},
                "provenance": provenance(), "budget_type": "iterations" if cfg.iterations else "wall-clock",
                "metric": {"reference": [1., cfg.cost_bound], "ideal": [0., 0.],
                           "normalization": cfg.cost_bound, "includes_empty_policy": True}}
    path = output / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if not resume:
            raise ValueError(f"{output} already contains a study; use --resume or a new output")
        if old != manifest:
            raise ValueError("resume requires identical configuration, methods, source, and runtime provenance")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("refusing to mix a study into a nonempty directory without its manifest")
    else:
        atomic_json(path, manifest)
    for environment in range(cfg.environments):
        environment_seed = seed_for(cfg.seed, environment, 0)
        launchers = np.random.default_rng(environment_seed).uniform(
            cfg.launcher_min, cfg.launcher_max, (cfg.adversaries, 2))
        order = np.random.default_rng(seed_for(cfg.seed, environment, 1)).permutation(methods)
        for method in order:
            runfile = output / f"env-{environment:03d}" / f"{method}.json"
            if runfile.exists():
                # Parse and check before trusting a completed artifact.
                old = json.loads(runfile.read_text())
                if old["environment"] != environment or old["method"] != method or "evaluation_nhv" not in old:
                    raise ValueError(f"invalid completed run: {runfile}")
                continue
            seed = seed_for(cfg.seed, environment, 2)
            print(f"environment {environment+1}/{cfg.environments}: {method}", flush=True)
            planner = Planner(cfg, launchers, method, seed)
            result = planner.run()
            result.update(environment=environment, environment_seed=environment_seed, planner_seed=seed,
                          evaluation_seed=seed_for(cfg.seed, environment, 3),
                          launchers=launchers.tolist())
            if save_tree:
                runfile.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(runfile.with_suffix(".tree.npz"), **planner.tree_arrays())
            del planner  # Release particle clouds before the independent evaluation phase.
            start = time.perf_counter()
            evaluate(result, cfg, launchers, result["evaluation_seed"])
            result["evaluation_seconds"] = time.perf_counter()-start
            atomic_json(runfile, result)
            print(f"  nHV={result['nhv']:.4f}, fresh={result['evaluation_nhv']:.4f}, "
                  f"{result['iterations_per_second']:.0f} iter/s, {result['nodes']} nodes", flush=True)
    return analyze(output)


def analyze(output):
    output = Path(output)
    manifest = json.loads((output/"manifest.json").read_text())
    cfg = Config(**manifest["config"])
    methods = manifest["methods"]
    runs = {m: {} for m in methods}
    for path in sorted(output.glob("env-*/*.json")):
        row = json.loads(path.read_text())
        m, e = row["method"], row["environment"]
        if m not in runs or e in runs[m]:
            raise ValueError("unexpected/duplicate method-environment record")
        runs[m][e] = row
    expected = set(range(cfg.environments))
    if any(set(runs[m]) != expected for m in methods):
        raise ValueError("study incomplete: all configured method/environment pairs are required for reporting")
    kwargs = dict(confidence=cfg.confidence, resamples=cfg.bootstrap_resamples, seed=cfg.seed)
    report = {"schema_version": 1, "environments": cfg.environments, "reference": [1., cfg.cost_bound],
              "methods": {}, "paired_differences": {}, "provenance": manifest["provenance"]}
    metrics = ("nhv", "evaluation_nhv", "iterations", "nodes", "iterations_per_second")
    for m in methods:
        report["methods"][m] = {key: mean_interval([runs[m][i][key] for i in sorted(expected)], **kwargs)
                                  for key in metrics}
    for a, b in combinations(methods, 2):
        report["paired_differences"][f"{a} - {b}"] = {
            key: mean_interval([runs[a][i][key]-runs[b][i][key] for i in sorted(expected)], **kwargs)
            for key in ("nhv", "evaluation_nhv")}
    atomic_json(output/"report.json", report)
    with (output/"summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric", "mean", "ci_low", "ci_high", "confidence", "environments", "ci_method"])
        for method, values in report["methods"].items():
            for metric, stats in values.items():
                writer.writerow([method, metric, stats["mean"], stats["low"], stats["high"],
                                 stats["confidence"], stats["n"], stats["method"]])
    def cell(s):
        return (f"{s['mean']:.3f} [{s['low']:.3f}, {s['high']:.3f}]" if s["low"] is not None
                else f"{s['mean']:.3f} [CI unavailable]")
    md = [f"Paired study: {cfg.environments} environments; {cfg.confidence:.0%} environment-bootstrap confidence intervals.",
          "", "| Method | Construction nHV (CI) | Fresh-evaluation nHV (CI) | Mean iterations | Mean nodes | Iter/s |",
          "|---|---:|---:|---:|---:|---:|"]
    tex = [r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Method & nHV (CI) & Fresh nHV (CI) & Iterations & Nodes \\", r"\midrule"]
    for method, s in report["methods"].items():
        md.append(f"| {LABELS[method]} | {cell(s['nhv'])} | {cell(s['evaluation_nhv'])} | "
                  f"{s['iterations']['mean']:.0f} | {s['nodes']['mean']:.0f} | {s['iterations_per_second']['mean']:.0f} |")
        tex.append(f"{LABELS[method]} & {cell(s['nhv'])} & {cell(s['evaluation_nhv'])} & "
                   f"{s['iterations']['mean']:.0f} & {s['nodes']['mean']:.0f}" + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (output/"report.md").write_text("\n".join(md)+"\n")
    (output/"table.tex").write_text("\n".join(tex)+"\n")
    return report


def benchmark(cfg, seconds=5., output=None, threshold=1000., methods=BASELINE_METHODS):
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("provide distinct supported methods")
    library()
    launchers = np.random.default_rng(cfg.seed).uniform(cfg.launcher_min, cfg.launcher_max,
                                                      (cfg.adversaries, 2))
    rows = []
    for m in methods:
        c = replace(cfg, seconds=seconds, iterations=None)
        # Warm native execution separately; timing uses a fresh complete growing tree.
        warm = Planner(replace(c, iterations=1), launchers, m, cfg.seed)
        warm.step()
        del warm
        planner = Planner(c, launchers, m, cfg.seed)
        r = planner.run()
        rows.append({k: r[k] for k in ("method", "iterations", "elapsed_seconds", "iterations_per_second",
                     "nodes", "witnesses", "cloud_memory_bytes", "metrics", "nhv")})
        print(f"{LABELS[m]}: {r['iterations_per_second']:.1f} iter/s; "
              f"{r['iterations']} iterations, {r['nodes']} retained", flush=True)
        del planner
    result = {"config": cfg.to_dict(), "methods": list(methods), "seconds_per_method": seconds, "provenance": provenance(),
              "target_iterations_per_second": threshold,
              "passed": all(r["iterations_per_second"] > threshold for r in rows), "runs": rows}
    if output:
        atomic_json(output, result)
    return result
