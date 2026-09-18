# Predictive SMO-SST validation

The implementation and policy choices are described in [PREDICTIVE_POLICY.md](../PREDICTIVE_POLICY.md).
These checks validate the additional `predictive_sst` group; the 30-environment, 120-second-per-method comparison has not been run.

## Correctness and compatibility

`cd code && .venv/bin/python -m pytest -q` passed all **54 tests**.
The predictive tests cover steering behavior, closest-approach and lifetime weighting, bounded controls, survival probabilities, gain sampling, exact target-only nesting, saved-policy replay, independent evaluation, reports, and rendering.
Serial and four-thread replay produce identical particles, including when replay is split into individual simulation steps.

Before the implementation change, snapshots were saved for each of the four original groups with 32 particles, two adversaries, `tau_max = 20`, 100 iterations, one rollout thread, launchers `[[1,1],[3,0]]`, and planner seed 13.
After the change, all four reproduced the same retained-node counts, front node IDs, objectives, and policy sequences.
A separate two-edge stochastic rollout with seed 7 reproduced each group's complete particle array bit for bit.
The snapshots were temporary local checks; the permanent tests additionally verify that a predictive family restricted to its zero-gain atom matches target-only parent selection, durations, process noise, particle states, and pruning.

## Sustained throughput

```sh
cd code
.venv/bin/smo-sst benchmark --config configs/predictive.toml \
  --methods predictive_sst --seconds 120 --require-target \
  --output validation/predictive-benchmark-120s.json
```

The complete sequential planner sustained **1,658.5 iterations/s** for 120 seconds on this macOS arm64 host, using 256 particles, eight adversaries, and four rollout workers.
It performed 199,017 iterations and retained 185,835 nodes.
The measured native rollout time was 62.4 seconds and witness-processing time was 50.0 seconds.
Particle-cloud storage at the end was approximately 7.34 GB, excluding Python and index overhead.
This is a growing-tree planner benchmark, not a rollout-only microbenchmark or a claim for other hardware.
The configuration, source fingerprint, runtime versions, and detailed counters are saved in [predictive-benchmark-120s.json](predictive-benchmark-120s.json).

## Short paired pilot

```sh
cd code
.venv/bin/smo-sst run --config configs/pilot.toml \
  --methods reactive_sst target_sst predictive_sst --output runs/predictive-pilot
```

This check used eight environments, three seconds per group/environment, 256 construction particles, 1,024 independent evaluation particles per selected policy, and 10,000 environment-bootstrap resamples.
The reported intervals are individual 95% BCa intervals across paired environments.
The policy defaults were specified before this pilot and were not tuned from these results.

| Group | Construction nHV | Fresh-evaluation nHV |
| --- | ---: | ---: |
| Reactive SMO-SST | 0.661 [0.635, 0.682] | 0.661 [0.634, 0.680] |
| Target-only SMO-SST | 0.655 [0.630, 0.677] | 0.654 [0.629, 0.676] |
| Predictive SMO-SST | 0.659 [0.631, 0.680] | 0.657 [0.630, 0.678] |

For fresh-evaluation nHV, predictive minus reactive was -0.0033 [-0.0078, 0.0022], and predictive minus target-only was 0.0033 [-0.0027, 0.0095].
Both paired intervals include zero; this pilot establishes neither superiority nor equivalence.
These short runs validate the experiment pipeline and must not be presented as the paper-scale comparison.
The machine-readable statistics and provenance are preserved in [predictive-pilot-report.json](predictive-pilot-report.json).
The full local study, saved policies, and reports are in `code/runs/predictive-pilot`.

## Visualization

```sh
cd code
.venv/bin/smo-sst figures runs/predictive-pilot --dpi 400
.venv/bin/smo-sst animate runs/predictive-pilot --method predictive_sst \
  --environment 0 --particles 40 --dpi 140 --max-frames 160
```

The figure interface exports vector PDF, editable SVG, and PNG for nHV confidence intervals, search progress, and Pareto fronts.
Predictive trajectory replay uses the saved radial and tangential gains and exports those formats plus a GIF and its replay metadata.
The rendered nHV figure and a midpoint GIF frame were visually checked for readable labels and layout.
The pilot GIF contains 160 frames at 980 by 980 pixels and is approximately 1.56 MB.
