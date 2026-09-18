#!/usr/bin/env bash
# Usage: bash run.sh [OUTPUT_ROOT [extra smo-sst run arguments, e.g. --resume]]
set -euo pipefail

config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$config_dir/../.."
output_root="${1:-runs/reactive_sst_witness_sweep}"
if (( $# > 0 )); then
    shift
fi

for config in "$config_dir"/delta_w_*.toml; do
    name="$(basename -- "$config" .toml)"
    printf '\nRunning reactive_sst with %s\n' "$name"
    uv run --locked smo-sst run --config "$config" \
        --methods reactive_sst --output "$output_root/$name" --save-tree "$@" --threads 16
done
