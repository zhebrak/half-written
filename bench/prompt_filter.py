"""The pass@k prompt filter (PLAN.md §4.3): the training prompts the warm start W0 solves sometimes, not always.

    python -m bench.prompt_filter CONFIG --init-from W0_DIR [--k 8] [--limit 2000] --out prompts.json [--lengths-out L]

A base model's first steps mostly learn the answer format, so every arm starts from a shared warm start W0: a
snapshot of arm A past that phase (GSM8K: S1's step 50, NOTEBOOK.md; set `init_from` to it). The filter samples k
answers per training prompt from W0 at the training temperature and keeps the prompts W0 solves sometimes but not
always (pass rate in [1/k, (k-1)/k]): a group whose answers all score the same carries no gradient. Runs read the kept
indices through `prompt_filter`. With --lengths-out, the sampled answers' lengths are written as a parquet `tokens`
column: W0's empirical length distribution, which the systems map draws from (`forced_lengths=parquet:<path>`).
"""
import argparse
import asyncio
import json
import math
import time
from pathlib import Path

from halfwritten.engines.base import collect
from halfwritten.revisions import model_revision
from halfwritten.tasks import score
from halfwritten.types import GenRequest


async def pass_at_k(engine, items, reward, *, k: int, max_tokens: int, stop=(), temperature: float = 1.0,
                    concurrency: int = 256, lengths: list[int] | None = None, reward_pool=None) -> list[float]:
    """Fraction of k sampled answers that score 1, per item [(prompt ids, answer)]; every answer's length is appended
    to `lengths` if given."""
    sem = asyncio.Semaphore(concurrency)

    async def sample(i, j, prompt, answer):
        async with sem:
            got = await collect(engine, GenRequest(f"prep-{i}-{j}#0", tuple(prompt), max_tokens,
                                                   temperature=temperature, stop=tuple(stop)))
            if lengths is not None:
                lengths.append(len(got.tokens))
            return i, float(await score(reward, reward_pool, answer, got.tokens, got.finish) >= 1.0)

    hits = [0.0] * len(items)
    for i, ok in await asyncio.gather(*(sample(i, j, p, a) for i, (p, a) in enumerate(items) for j in range(k))):
        hits[i] += ok
    return [h / k for h in hits]


def keep(rates: list[float], k: int) -> list[int]:
    """Indices whose pass rate is neither 0 nor 1: at least one right and one wrong answer in k."""
    return [i for i, r in enumerate(rates) if 1 / k - 1e-9 <= r <= (k - 1) / k + 1e-9]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--init-from", help="the warm start's snapshot directory (default: the base model)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths-out", default=None, help="parquet of the sampled answers' lengths (the systems map's)")
    ap.add_argument("--set", action="append", default=[], help="config overrides, e.g. launch.head_fp32=true")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer

    from bench.evaluate import vllm_push
    from halfwritten.config import load, with_overrides
    from halfwritten.engines.vllm_http import run_uvloop
    from halfwritten.launch import tool_config, tool_engine
    from halfwritten.tasks import (
        STOPS,
        TextReward,
        encode,
        gsm8k_examples,
        math_examples,
        scoring,
        too_long,
    )
    cfg = tool_config(with_overrides(load(a.config), a.set), ipc=True)
    tok = AutoTokenizer.from_pretrained(cfg.model, revision=model_revision(cfg.model))
    rows = (gsm8k_examples("train") if cfg.task == "gsm8k" else math_examples("train"))[:a.limit]
    items = encode(tok, cfg.task, rows)
    long = set(too_long(items, cfg.max_tokens, cfg.max_model_len))  # never sampled, never kept (vLLM would refuse)
    out = Path(a.out)
    with scoring(TextReward(cfg.model, cfg.task)) as (reward, pool), \
            tool_engine(cfg, out.parent / "logs-prompt-filter") as (url, engine):
        if a.init_from:
            vllm_push(url, cfg.launch.engine_gpu)(1, Path(a.init_from))
        t0 = time.monotonic()
        lengths: list[int] = []
        fit = [i for i in range(len(items)) if i not in long]
        sampled = run_uvloop(pass_at_k(engine, [items[i] for i in fit], reward, k=a.k,
                                       max_tokens=cfg.max_tokens, stop=STOPS[cfg.task], temperature=cfg.temperature,
                                       lengths=lengths, reward_pool=pool))
        rates = [math.nan] * len(items)
        for i, r in zip(fit, sampled):
            rates[i] = r
    if a.lengths_out:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.table({"tokens": lengths}), a.lengths_out)
    kept = keep(rates, a.k)
    mean = sum(sampled) / len(sampled)
    from halfwritten.results import file_sha256
    from halfwritten.revisions import task_revisions
    identity = {"dataset_revisions": task_revisions(cfg.task),  # the indices below mean these rows (review F12)
                "init_sha256": file_sha256(Path(a.init_from) / "model.safetensors") if a.init_from else None}
    out.write_text(json.dumps({"task": cfg.task, "model": cfg.model, "init": a.init_from, **identity, "k": a.k,
                               "n": len(rates),
                               "kept": kept, "too_long": sorted(long), "mean_pass": mean,
                               "seconds": time.monotonic() - t0, "rates": [None if math.isnan(r) else r
                                                                           for r in rates]}))
    print(f"kept {len(kept)} of {len(rates)} prompts ({len(long)} too long; mean pass rate {mean:.3f})")


if __name__ == "__main__":
    main()
