# Half-written answers

What happens to an RL rollout when the sampling policy receives an update mid-answer. The harness runs the regimes
on stock vLLM 0.30 (`vllm serve`, weights pushed over NCCL), records every token's provenance, and measures what each
regime costs in learning and throughput: Qwen3-0.6B-Base on two H100 SXM, 9 seeds per regime on GSM8K from a
synchronous checkpoint, 3 seeds on MATH from the base model, and a teacher-forcing study of what a stale cache does
per token.

Write-up: [zhebrak.io/posts/half-written-answers](https://zhebrak.io/posts/half-written-answers/)

## Regimes

| | At the push | The rest of a running answer sees |
|---|---|---|
| A, synchronous | nothing in flight: generate the batch, train, push | — |
| B, drain | `pause(mode="wait")`, push | old weights throughout |
| C1, abort and resume | `pause(mode="abort")`, push, resubmit prompt + partial answer | new weights, the partial answer re-prefilled |
| C2, abort and discard | `pause(mode="abort")`, push, resubmit the prompt | the partial answer is thrown away (no published rows) |
| D, in flight, stale KV | `pause(mode="keep", clear_cache=False)`, push | new weights reading the old weights' cache |
| E, in flight, fresh KV | `pause(mode="keep", clear_cache=True)`, push | new weights, cache recomputed in the engine |

## Results

- **GSM8K.** E, the asynchronous arm fixed before the grid, beats A: +1.86 points of area under accuracy to 5 min,
  95% interval [+1.33, +2.39], the only one of four Holm-adjusted contrasts to separate. At matched optimiser steps
  B, D and E each learn the same as A within 2 points. Answers are short: 0.7% of the tokens consumed after the first
  minute came from a stale cache (1.8% over the whole run).
- **MATH.** 38% of the tokens consumed after the first minute came from a stale cache. D − E −0.1 points of area,
  95% interval [−3.0, +2.8]; D − B +2.3 [−1.0, +5.6]; E − B +2.4 [−3.6, +8.5]. Three seeds, intervals only.
- **Per token.** Pooled over caches 1 to 32 steps stale, a stale cache undoes 82% of an update's effect over the
  first 8 tokens after the swap and 49% from token 32 on; layers 15 to 21 hold 73% of the staleness (at one step:
  84%, 46%, 76%). One step moves the next-token distribution by a mean of about 5e-5 nats per token (median 4e-7).

## Running it

```bash
uv venv --python 3.12 .venv && uv pip install -r requirements-cpu.lock      # a laptop
.venv/bin/python -m pytest -q
.venv/bin/python -m halfwritten sim configs/cpu/all_regimes.yaml --regimes A,B,C1,C2,D,E
```

On a box with two H100 over NVLink, from a fresh clone with no `.venv`: `scripts/setup_box.sh` (installs
`requirements.txt`; the freeze of the box that ran the grids is `data/two_gpu-2026-10-03/_study/requirements-gpu.lock`).

```bash
.venv/bin/python -m halfwritten fetch gsm8k        # W0 and the prompt filter; needs W0's pinned revision (below)
.venv/bin/python -m halfwritten run configs/two_gpu/gsm8k.yaml --set regime=D --set seed=0 \
    --set budget_s=300 --set snapshot_every_s=37.5 --set eval_split=test    # one row as the grid ran it
.venv/bin/python -m bench.evaluate results/gsm8k/<model>/<gpu>/D-B32G8M32-K4-S1-s0       # score it
.venv/bin/python -m bench.queue run configs/matrix/gsm8k_main.yaml --gpu-wait-s 900     # the GSM8K grid
.venv/bin/python -m bench.queue run configs/matrix/math_base.yaml --gpu-wait-s 900      # the MATH tier
.venv/bin/python -m halfwritten run configs/one_gpu/phase1_checkpoints.yaml             # the checkpoint series
.venv/bin/python -m bench.stale_kv --snapshot-dir <that row>/work --old-version 4 --lags 0,1,2,4,8,16,32 \
    --prompts 32 --positions 96 --fresh-layers 1,7,14,21,27 --dtype float32 --out results/phase1/fp32
```

The analysis files and the charts page, from the data in the repo:

```bash
.venv/bin/python -m bench.stats data/two_gpu-2026-10-03/gsm8k --split test --delta 0.02 \
    --amendment configs/amendments/gsm8k.yaml --out docs/analysis/two_gpu-2026-10-03/gsm8k.json
.venv/bin/python -m bench.stats data/two_gpu-2026-10-03/math --tag math-base --split test --delta 0.03 --intervals-only \
    --equivalence "D-E" --h4 auto --family "D-E,D-B,E-B,D-A,E-A,B-A" \
    --amendment configs/amendments/math_base.yaml --out docs/analysis/two_gpu-2026-10-03/math-base.json
.venv/bin/python scripts/build_post.py      # docs/charts.html (renders Markdown through uvx)
```

W0 is step 50 of `configs/one_gpu/gsm8k_sync_curve.yaml` (sha256 `568716f7…8ea8`, origin in
`data/h100nvl-2026-10-01/gsm8k/`); `fetch` restores it once its Hub revision is pinned in `halfwritten/revisions.py`.
A regenerated W0 is a different starting point.

## Layout

```
halfwritten/   the harness: controller, ledger, fence, launch, engines (HTTP and fake), trainer
bench/         evaluate, queue, canary, stats, stale_kv, prompt_filter; the simulator's tools; dashboard/findings/figures,
               the development-time results page the chart build takes its explainer from (not built here)
scripts/       build_post.py (the charts page), setup_box.sh (a fresh box)
configs/       two_gpu/ (run configs), matrix/ (the grids; math_main.yaml never ran, a test plans it), amendments/ (analysis
               settings), one_gpu/ (the synchronous curve, the checkpoint series), cpu/ (the fake engine)
data/          the published rows, canary and grid reports, prompt filter, run environment, W0's origin, the study
docs/          charts.html, analysis/
tests/         pytest, with a fake engine that reproduces vLLM's pause modes, abort finals and delivery ordering
```

## Limits

0.6B, learning rate 1e-6, a lag of about one step between the cache's weights and the sampling weights, five- and
ten-minute budgets, one host; three MATH seeds; the teacher-forcing study is one seed and 32 prompts. The harness is
the code that ran: `halfwritten/`, `bench/queue.py`, `bench/evaluate.py`, `bench/canary.py` and
`bench/prompt_filter.py` are unchanged since the grids apart from comment lines; the analysis, study and page modules
changed after the runs without touching any published number. Comments refer to the development history's planning
records, which are not part of this release.
