# The data behind the post

Three session archives, trimmed to what the post's numbers rest on. Paths are the ones the code names, so the
analysis and chart commands in the top-level README run against this tree as it is; the GSM8K runs also need W0,
which `fetch` restores once its revision is pinned.

## Sessions

| Directory | Hardware | What it holds |
|---|---|---|
| `two_gpu-2026-10-03/` | two H100 SXM over NVLink (engine on GPU0, trainer on GPU1) | the GSM8K grid (A/B/D/E × seeds 0-8, C1 × seeds 0-1), the MATH tier (A/B/D/E × seeds 0-2) and its synchronous curve, the two canary reports and the two grid reports, the prompt filter, the run's package freeze |
| `h100nvl-2026-10-01/` | one H100 NVL | the fp32 teacher-forcing study, and the synchronous run W0 was taken from |
| `two_gpu-2026-10-02/` | two H100 SXM | the prompt filter at the path the code reads it from (identical to the 10-03 copy) |

C1 is abort and resume: the prompt is resubmitted with the partial answer and re-prefilled on the new weights. Its
two seeds were a consistency check against E, outside the contrast family, and the post does not report them. The
README's C, abort and discard, is `C2` in the code and has no published rows.

## A row

`<task>/<model>/<gpu>/<regime>-B<batch>G<group>M<concurrency>-K<lag>-S<split>[-<tag>]-s<seed>/`, one directory per
run of one regime and seed:

- `config.yaml`: the run's configuration, the environment (package versions, GPU model, driver, power limit and
  clocks, the container image) and the identity hashes (the starting weights, the prompt filter);
- `metrics.json`: steps, wall clock, throughput, the harness record (host CPU share, loop lag, fence latency), the
  token counters, `undelivered` (the row's own boundary record: tokens that arrived after a pause acknowledgement,
  zero at every weight update in every published row) and the validity verdict;
- `evals.jsonl`: held-out accuracy at every snapshot (the test split; `evals-500.jsonl` where a 500-problem split was
  also scored);
- `snapshots.json`: the snapshot schedule;
- `steps.parquet`: one row per optimiser step, including `frac_stale_kv`, the share of the step's consumed tokens
  that were generated over a stale cache;
- `syncs.parquet`: one row per weight update: pause time, sync time, answers in flight, tokens re-prefilled;
- `certificate.json`: the canary verdict for this row's settings and the report it came from;
- `ledger.parquet`, for the two D seed-0 rows only: every answer's token segments with the weights that sampled
  them and the cache they read.

The published GSM8K rows are the untagged ones; the MATH rows carry the tag `math-base`; the MATH synchronous curve
from the base model is the `sync-curve` row (development split), run on the two-GPU layout from
`configs/two_gpu/math.yaml` with the overrides its `config.yaml` records (200 steps, no warm start, no filter).

Each row's `environment.commit` names the tag the private development history carried at the run (`prereg-v1`
for GSM8K, twelve commits later for MATH); the harness is identical at both. `results/…` in the queue reports and
certificates is the box's path for what is now `two_gpu-2026-10-03/`.

## What the post reads

- The learning curves: every row's `evals.jsonl`.
- The contrasts and intervals: `docs/analysis/two_gpu-2026-10-03/{gsm8k,math-base}.json`, written by `bench.stats`
  from `evals.jsonl`, `snapshots.json`, `steps.parquet` and `metrics.json` with the settings in
  `configs/amendments/` (the top-level README has the two commands).
- Exposure: `equivalence[].d_stale_kv_share` in those files, the token-weighted share of tokens consumed after the
  first minute that were generated over a stale cache, averaged over the D seeds (0.0075 on GSM8K, 0.378 on MATH).
  Over whole runs, from `steps.parquet`, the figures are 0.018 and 0.37. The MATH D seed-0 ledger gives the
  per-answer view: 38% of the answers consumed after the first minute crossed a weight update, with 983 tokens after
  it on average.
- The teacher-forcing charts: `h100nvl-2026-10-01/phase1/gsm8k-qwen3-0.6b-h100-fp32compute/positions.parquet`,
  one row per scored position (the divergences under each condition and under each layer cut), and `summary.json`
  (the arguments and the per-lag table).

## How the analysis reads

- The endpoint is the area under held-out accuracy against wall clock over [0, T], T = 5 min on GSM8K, normalised
  to a mean height; accuracy at T is the secondary reading.
- The GSM8K contrast family was fixed before the grid ran: D − E, D − B, E − B and best − A, Holm-corrected, with the
  best asynchronous arm chosen from one throughput row per arm (`best_async: E` and `map_steps_per_hour` in
  `configs/amendments/gsm8k.yaml`: E 1,536, D 1,512, B 1,500 steps per hour; D then ran faster in the grid itself).
  Equivalence at matched optimiser steps uses a 2-point margin; 9 seeds came from a six-hour box cap.
- D − E was excluded from GSM8K's equivalence test before the grid because D's exposure in the pilot, 1.1%, was
  below the 5% floor (`h4: out`, `min_exposure: 0.05` in the same file), which is why `gsm8k.json` calls that
  comparison "uninformative".
- MATH was designed afterwards (`configs/amendments/math_base.yaml`): the same four regimes from the base model,
  600 s, unfiltered prompts, scored on MATH-500, 3 seeds, reported as 95% intervals with a 3-point margin and no
  equivalence verdict. `configs/matrix/math_main.yaml`, a MATH grid from a warm start, never ran: `prep/math/w0.json`
  records that no checkpoint had headroom over the plateau, so the tier ran from the base model instead.

## The record of the grids

- `two_gpu-2026-10-03/_queue/report-1791040260.json` (GSM8K, 38 rows) and `report-1791063299.json` (MATH, 12 rows):
  every row in the two reports valid on its first attempt, none excluded. One earlier MATH B seed-0 row, run at T/8
  snapshot spacing before the spacing was set to T/16, was replaced and is not in the data.
- `_queue/canary-1791020528-0/report.json` (GSM8K) and `canary-1791047947-0/report.json` (MATH): the loaded canary
  each grid ran under (`post_stale_on_decisive` 1.0, `boundary_exact_stale` true). Each row's `certificate.json`
  names its report.
- `_study/requirements-gpu.lock`: the package freeze of the box that ran the grids.
- `configs/amendments/gsm8k.yaml` and `math_base.yaml` (top level): the grid's values (T, snapshot spacing, seeds)
  and the analysis settings (the margin, the contrast family's best arm, the harness limits). The GSM8K file was
  written by the pilot's fit; the MATH file by hand. `configs/matrix/gsm8k_main.yaml` reads the GSM8K file.
- `prep/gsm8k/`: the frozen prompt filter (`pass8.json`), the length draws (`w0-lengths.parquet`) and W0's hash
  (`w0.json`). `prep/math/w0.json`: the MATH warm-start rule's outcome.

## W0's origin and the teacher-forcing study

- `h100nvl-2026-10-01/gsm8k/…/A-B32G8M32-K0-S1-s1-s0/`: the synchronous run from the base model that W0 is step 50
  of; `checkpoints.sha256` ties the hash `568716f7…8ea8` to `step0050-v50`. Its `config.yaml` is a reconstruction
  with no environment block, as its `note` says; its step-50 test score there (0.594) and the grid's W0 start
  (0.5906) were scored on different boxes.
- `h100nvl-2026-10-01/phase1/…-fp32compute/`: written by an earlier revision of `bench/stale_kv.py` (the flag names
  in `summary.json` are the old ones: `snapshots`, `start`, `offsets`, `layers` map to `--snapshot-dir`,
  `--old-version`, `--positions`, `--fresh-layers`), from a 40-step checkpoint series made with
  `configs/one_gpu/phase1_checkpoints.yaml` on that H100 NVL (the config's "one A100" comment is from an earlier
  plan); the series itself is not included. The current code recomputes the published shares from the parquet.
