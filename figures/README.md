# Generated figure gallery

This directory is a held-out presentation layer generated from saved experiment
artifacts. It is not an input to planning, statistical analysis, or manuscript builds.

## Provenance

- `delta_w_sweep.*`, `delta_w_sweep.csv`, and `delta_w_trees_env_000.*` use
  `runs/reactive_sst_witness_sweep/`.
- `paper/nhv.*`, `paper/progress.*`, `paper/fronts-000.*`, and all files beginning
  with `paper/trajectory-000-` use `runs/paper_final_improved_witness/`.
- The trajectory animations use environment 0, 32 fresh particles, seed 98765,
  96-dpi GIF frames, 18 frames/s, nominal 12x playback speed, and at most 120 frames.
  Adjacent JSON files record the selected front member, policy, and frame indices.

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
```

Use writable `UV_CACHE_DIR`, `MPLCONFIGDIR`, and `XDG_CACHE_HOME` locations when the
default caches are unavailable in a managed environment.
