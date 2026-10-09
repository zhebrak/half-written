"""The noise floor (H0, gate G0): how far the engine's log-probabilities sit from the trainer's at identical weights,
before any staleness. Every staleness effect must clear it.

    python -m bench.noise_floor CONFIG [--prompts 128] [--out noise_floor.json]

The engine (vllm serve, the study's flags) samples answers at the training temperature; the trainer's own code
(trainer/model.py: bf16 autocast, the chunked FP32 head) recomputes the log-probability of every sampled token from
the same weights, pushed into the engine first so both hold the trainer's bf16 copy bit for bit. Reported per
condition, over tokens: k1, k3 (KL(mu || pi)), the normalised ESS and truncated fraction of min(pi/mu, eps_max), and
quantiles of |log pi - log mu|. Conditions:
    steady         natural answers (EOS and stop strings), no pause: the floor training sees
    steady_forced  answers forced to max_tokens (ignore_eos, no stops), no pause: the baseline of the two below
    noop_keep      forced; a no-op swap mid-answer: pause(keep), push the same weights, resume (D's path, KV kept)
    noop_clear     the same with clear_cache (E's path: the context re-prefilled with prefill kernels)
Tokens after a no-op swap are scored separately ("after"), since E's recompute changes the kernels that produced the
KV. The swaps must force length so every answer spans them, and forced text past the natural end is text the model
would not have written, so they are compared with steady_forced, never with steady. The floor is compared with the
per-step drift KL(pi_s || pi_s+1) from Phase 1 (H0: floor <= 0.1 x drift).

This is the kernel floor at v0, the one point where the trainer's fp32 master and the engine's bf16 copy agree bit for
bit. During training they drift apart (an update at lr 1e-6 is about 1% of a bf16 ulp per element), so the operative
floor is arm A's lag-0 mismatch over the run: every A row logs it per step (steps.parquet: k3, ess, truncated), and
H0 is judged on both.
"""
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from halfwritten.revisions import model_revision


def mismatch(logp: np.ndarray, mu: np.ndarray, eps_max: float = 5.0) -> dict:
    from halfwritten import metrics
    lr = logp - mu
    w, trunc = metrics.truncated_weights(lr, eps_max)
    q = np.quantile(np.abs(lr), [0.5, 0.9, 0.99, 1.0]) if lr.size else [float("nan")] * 4
    return {"tokens": int(lr.size), "k1": metrics.k1(lr), "k3": metrics.k3(lr), "ess": metrics.ess(w),
            "truncated": trunc, "abs_p50": float(q[0]), "abs_p90": float(q[1]), "abs_p99": float(q[2]),
            "abs_max": float(q[3])}


async def sample(engine, items, *, max_tokens, stop, temperature, swap=None, swap_after: int = 32,
                 forced: bool | None = None, timeout_s: float = 600.0):
    """Answers [(prompt, tokens, mu, boundary)]; with `swap` (an async callable), it runs once every answer has
    swap_after tokens or has finished, and each answer's boundary is its token count at the swap (None without
    one). Forced answers (ignore_eos, no stop strings; the default with a swap, so every answer spans it) run to
    max_tokens, past where the model would have stopped: compare them only with forced answers."""
    forced = swap is not None if forced is None else forced
    from halfwritten.types import GenRequest
    answers = [[p, [], [], None] for p, _ in items]
    finished = [False] * len(items)
    reached = asyncio.Event()

    def check():
        if all(f or len(a[1]) >= swap_after for f, a in zip(finished, answers)):
            reached.set()

    async def one(i, prompt):
        req = GenRequest(f"nf-{i}#0", tuple(prompt), max_tokens, temperature=temperature,
                         stop=() if forced else tuple(stop), ignore_eos=forced)
        try:
            async for c in engine.generate(req):
                answers[i][1] += c.token_ids
                answers[i][2] += c.logprobs
                check()
        finally:
            finished[i] = True
            check()

    tasks = [asyncio.create_task(one(i, p)) for i, (p, _) in enumerate(items)]
    if swap is not None:
        await asyncio.wait_for(reached.wait(), timeout_s)
        await swap(answers)
    await asyncio.wait_for(asyncio.gather(*tasks), timeout_s)
    return answers


def trainer_logprobs(trainer, prompt, tokens) -> np.ndarray:
    import torch

    from halfwritten.types import TrainSample
    s = TrainSample("nf", "g", tuple(prompt), tuple(tokens), (0.0,) * len(tokens), 0.0, 0.0, (), (), ())
    with torch.no_grad():
        return trainer.logprobs([s], [0])[0].float().cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], help="config overrides, e.g. launch.head_fp32=true")
    ap.add_argument("--prompts", type=int, default=128)
    ap.add_argument("--out", default="noise_floor.json")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer

    from halfwritten.config import load, with_overrides
    from halfwritten.engines.vllm_http import run_uvloop
    from halfwritten.gpu import local_device
    from halfwritten.launch import tool_config, tool_engine
    from halfwritten.tasks import STOPS, encode, eval_examples
    from halfwritten.trainer.model import PolicyTrainer, PolicyTrainerSettings
    from halfwritten.trainer.sync import VllmSink
    cfg = tool_config(with_overrides(load(a.config), a.set), ipc=True)
    tok = AutoTokenizer.from_pretrained(cfg.model, revision=model_revision(cfg.model))
    items = encode(tok, cfg.task, eval_examples(cfg.task, "test")[:a.prompts])  # as the archived floors were
    trainer = PolicyTrainer(PolicyTrainerSettings(model=cfg.model, device=local_device(cfg.launch.engine_gpu),
                                            temperature=cfg.temperature,
                                            gradient_checkpointing=False, init_from=cfg.init_from))
    trainer.model.eval()
    tensors = {k: v.detach().to(trainer.dtype) for k, v in trainer.engine_tensors().items()}
    out = Path(a.out)
    from halfwritten.results import host_gpu_label
    report = {"model": cfg.model, "temperature": cfg.temperature, "prompts": len(items),
              "head_fp32": cfg.launch.head_fp32, "gpu": host_gpu_label()}
    with tool_engine(cfg, out.parent / "logs-noise") as (url, engine):
        sink = VllmSink(url, backend="ipc", packed=True)
        sink.push(0, tensors)  # the engine now holds exactly the trainer's bf16 weights
        version = [0]

        def noop(clear):
            async def swap(answers):
                await engine.pause("keep", clear)
                await asyncio.sleep(0.3)
                for ans in answers:
                    ans[3] = len(ans[1])
                version[0] += 1
                await asyncio.to_thread(sink.push, version[0], tensors)
                await engine.resume()
            return swap

        # steady: natural answers, the floor training sees. steady_forced: full-length answers with no swap, the
        # like-for-like baseline of the no-op swaps (which must force length so every answer spans the swap).
        conditions = (("steady", None, False), ("steady_forced", None, True), ("noop_keep", noop(False), True),
                      ("noop_clear", noop(True), True))
        for name, swap, forced in conditions:
            answers = run_uvloop(sample(engine, items, max_tokens=cfg.max_tokens, stop=STOPS[cfg.task],
                                 temperature=cfg.temperature, swap=swap, forced=forced))
            before, after = [], []
            for prompt, tokens, mu, boundary in answers:
                if not tokens:
                    continue
                lr = trainer_logprobs(trainer, prompt, tokens) - np.asarray(mu)
                cut = len(tokens) if boundary is None else boundary
                before.append(lr[:cut])
                after.append(lr[cut:])
            report[name] = {"before": mismatch_stats(np.concatenate(before)),
                            "after": mismatch_stats(np.concatenate(after))}
            print(name, json.dumps(report[name]), flush=True)
    out.write_text(json.dumps(report, indent=1))


def mismatch_stats(lr: np.ndarray) -> dict:
    """mismatch() of a log-ratio array (log pi - log mu)."""
    return mismatch(lr, np.zeros_like(lr))


if __name__ == "__main__":
    main()
