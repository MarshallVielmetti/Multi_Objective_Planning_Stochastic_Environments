import argparse
from dataclasses import replace
from .config import Config, METHODS, BASELINE_METHODS


def main(argv=None):
    p = argparse.ArgumentParser(description="Sequential SMO-SST: experiments, confidence intervals, and visualizations")
    sub = p.add_subparsers(dest="command", required=True)
    def options(parser):
        parser.add_argument("--config", help="flat TOML or JSON configuration")
        parser.add_argument("--particles", type=int)
        parser.add_argument("--threads", type=int, dest="rollout_threads")
    run = sub.add_parser("run", help="run/resume a paired study and generate statistical reports")
    options(run)
    run.add_argument("--output", required=True)
    run.add_argument("--environments", type=int)
    run.add_argument("--seconds", type=float)
    run.add_argument("--iterations", type=int, help="deterministic test budget; replaces wall-clock budget")
    run.add_argument("--seed", type=int)
    run.add_argument("--evaluation-particles", type=int)
    run.add_argument("--methods", nargs="+", choices=METHODS, default=list(BASELINE_METHODS))
    run.add_argument("--resume", action="store_true")
    run.add_argument("--save-tree", action="store_true")
    bench = sub.add_parser("benchmark", help="benchmark complete growing planners after native warm-up")
    options(bench)
    bench.add_argument("--seconds", type=float, default=10.)
    bench.add_argument("--output", default="runs/benchmark.json")
    bench.add_argument("--threshold", type=float, default=1000.)
    bench.add_argument("--require-target", action="store_true", help="exit nonzero if any method misses throughput target")
    bench.add_argument("--methods", nargs="+", choices=METHODS, default=list(BASELINE_METHODS))
    report = sub.add_parser("report", help="regenerate all tables from a complete study")
    report.add_argument("study")
    fig = sub.add_parser("figures", help="render nHV CIs, progress, and Pareto fronts to PDF/SVG/PNG")
    fig.add_argument("study")
    fig.add_argument("--output")
    fig.add_argument("--environment", type=int, default=0)
    fig.add_argument("--dpi", type=int, default=400)
    gif = sub.add_parser("animate", help="render a selected policy to publication figures and a GIF")
    gif.add_argument("study")
    gif.add_argument("--method", choices=METHODS, default=METHODS[0])
    gif.add_argument("--environment", type=int, default=0)
    gif.add_argument("--point", type=int, help="index into the saved construction front; default: largest fresh rectangle")
    gif.add_argument("--particles", type=int, default=32)
    gif.add_argument("--seed", type=int, default=98765)
    gif.add_argument("--output")
    gif.add_argument("--dpi", type=int, default=140)
    gif.add_argument("--fps", type=int, default=24)
    gif.add_argument("--speed", type=float, default=8., help="nominal simulated seconds per playback second")
    gif.add_argument("--max-frames", type=int, default=240)
    gif.add_argument("--no-gif", action="store_true", help="only render static trajectory figures")
    args = p.parse_args(argv)
    try:
        if args.command in ("run", "benchmark"):
            from .experiment import run_study, benchmark
            c = Config.load(args.config) if args.config else Config()
            names = ("particles", "rollout_threads", "environments", "seconds", "iterations", "seed", "evaluation_particles")
            c = replace(c, **{key: getattr(args, key) for key in names
                             if hasattr(args, key) and getattr(args, key) is not None})
            if args.command == "run":
                run_study(c, args.output, args.methods, args.resume, args.save_tree)
                print(f"Saved reports to {args.output}/report.md")
            else:
                result = benchmark(c, args.seconds, args.output, args.threshold, args.methods)
                if args.require_target and not result["passed"]:
                    return 1
        elif args.command == "report":
            from .experiment import analyze
            analyze(args.study)
        elif args.command == "figures":
            from .visualize import figures
            print("\n".join(figures(args.study, args.output, args.environment, args.dpi)))
        elif args.command == "animate":
            from .visualize import trajectory
            print("\n".join(trajectory(args.study, args.method, args.environment, args.point,
                args.particles, args.seed, args.output, args.dpi, args.fps, args.speed,
                args.max_frames, not args.no_gif)))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        p.error(str(exc))
    return 0
