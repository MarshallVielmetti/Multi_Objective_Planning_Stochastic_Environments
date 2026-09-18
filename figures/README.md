# Generated figure gallery

This directory is a held-out presentation layer generated from saved experiment
artifacts. It is not an input to planning, statistical analysis, or manuscript builds.

## Provenance

- `delta_w_sweep.*`, `delta_w_sweep.csv`, and `delta_w_trees_env_000.*` use
  `runs/reactive_sst_witness_sweep/`.
- `paper/nhv.*`, `paper/progress.*`, `paper/fronts-000.*`, and all files beginning
  with `paper/trajectory-000-` use `runs/paper_final_improved_witness/`.
- `pareto_env_000/` contains all 55 policies in the four returned construction
  Pareto portfolios for environment 0 of `runs/paper_final_improved_witness/`.
  Its labels are the saved 4,096-particle fresh-evaluation estimates; `points.csv`
  and `manifest.json` retain the exact values and rendering metadata.
- The trajectory animations use environment 0, 32 fresh particles, seed 98765,
  96-dpi GIF frames, 18 frames/s, nominal 12x playback speed, and at most 120 frames.
  Adjacent JSON files record the selected front member, policy, and frame indices.
- The compact Pareto-portfolio animations use 16 particles, the matching saved
  evaluation-stream seed for each point, 72-dpi GIF frames, 12 frames/s, nominal
  20x playback speed, and at most 48 frames. Their motion is qualitative; the
  displayed risk and cost labels come from the 4,096-particle evaluation.

## Regenerate

Run these commands from `code/`:

```sh
uv run --locked python configs/reactive_sst_witness_sweep/visualize.py \
  runs/reactive_sst_witness_sweep --output figures

uv run --locked python configs/reactive_sst_witness_sweep/visualize_trees.py 0 \
  --sweep runs/reactive_sst_witness_sweep --output figures

uv run --locked smo-sst figures runs/paper_final_improved_witness \
  --output figures/paper --environment 0 --dpi 400

uv run --locked smo-sst animate runs/paper_final_improved_witness \
  --method reactive_sst --environment 0 --particles 32 --seed 98765 \
  --output figures/paper --dpi 96 --fps 18 --speed 12 --max-frames 120

uv run --locked smo-sst animate runs/paper_final_improved_witness \
  --method target_sst --environment 0 --particles 32 --seed 98765 \
  --output figures/paper --dpi 96 --fps 18 --speed 12 --max-frames 120

uv run --locked smo-sst animate runs/paper_final_improved_witness \
  --method reactive_rrt --environment 0 --particles 32 --seed 98765 \
  --output figures/paper --dpi 96 --fps 18 --speed 12 --max-frames 120

uv run --locked smo-sst animate runs/paper_final_improved_witness \
  --method target_rrt --environment 0 --particles 32 --seed 98765 \
  --output figures/paper --dpi 96 --fps 18 --speed 12 --max-frames 120

uv run --locked python scripts/render_pareto_front_gallery.py \
  runs/paper_final_improved_witness --environment 0 \
  --output figures/pareto_env_000 --particles 16 --dpi 72 \
  --fps 12 --speed 20 --max-frames 48
```

Use writable `UV_CACHE_DIR`, `MPLCONFIGDIR`, and `XDG_CACHE_HOME` locations when the
default caches are unavailable in a managed environment.
