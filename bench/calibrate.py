"""Per-session calibration (PLAN.md §4.5, H1): the engine's step-time model, measured, for the regime calculator.

    python -m bench.calibrate CONFIG [--out calibration.json]

Through the same HTTP adapter the loop uses, it loads the engine two ways and reads vLLM's own step times
(server-side histograms; the client's wall time is kept beside them):
    decode   w requests of L forced tokens each (ignore_eos) at widths w in 1..256 and two prompt lengths P: ~L steps
             of width w over a context growing from P to P+L (mean context per step w * (P + L/2))
    prefill  w requests of max_tokens=1 with long prompts, as many per step as the token budget holds
and fits CostModel (step = alpha + beta*width + gamma*context + delta*prefill) in two stages (fit()). The fit's error
on the measured points is reported; bench.calculator --cost then predicts every regime with it. Against the fake
server the fit recovers the fake's own CostModel, which is the CPU test.
"""
import argparse
import asyncio
import json
from pathlib import Path

from halfwritten.engines.base import collect
from halfwritten.types import GenRequest


async def _timed(engine, reqs) -> float:
    """Seconds by the event loop's clock: monotonic time on a real engine, virtual time against FakeEngine."""
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await asyncio.gather(*(collect(engine, r) for r in reqs))
    return loop.time() - t0


async def measure(engine, *, widths=(1, 8, 32, 64, 128, 256), decode_len: int = 128, prompt_len: int = 64,
                  prefill_lens=(512, 1024, 2048), prefill_widths=(1, 4, 16), vocab: int = 1000,
                  max_model_len: int | None = None, max_num_batched_tokens: int = 8192,
                  server_metrics=None) -> list[dict]:
    """Points (width, context, prefill, seconds per step) under decode and prefill load. Prefill prompts are capped
    to fit the engine's context with their one output token (vLLM rejects a request that does not fit).

    With `server_metrics` (an async callable returning the engine's /metrics), each point's seconds are the engine's
    own: the mean inter-token latency for decode (one engine step), the mean prefill time for prefill. The client's
    wall time per step is kept as client_seconds: it adds HTTP and streaming costs, which grow with concurrency.
    A prefill step holds at most max_num_batched_tokens tokens (chunked prefill is off), so a wide prefill point is
    several steps of fewer requests each, and is recorded as one such step."""
    if max_model_len is not None:
        prefill_lens = sorted({min(p, max_model_len - 1) for p in prefill_lens})

    async def server_delta(before, key):
        if before is None:
            return None
        after = await server_metrics()
        n = after.get(f"{key}_count", 0) - before.get(f"{key}_count", 0)
        return (after.get(f"{key}_sum", 0) - before.get(f"{key}_sum", 0)) / n if n > 0 else None

    pts, n = [], 0
    # two prompt lengths, so a step's context is not just a multiple of its width (else the per-sequence and
    # per-context-token costs cannot be told apart); the long prompts at a few widths only
    long_len = min(8 * prompt_len, (max_model_len or 4096) - decode_len - 1)
    sweeps = [(prompt_len, widths), (long_len, tuple(w for w in widths if w in (widths[1], widths[len(widths) // 2],
                                                                                  widths[-1])))]
    for plen, ws in sweeps:
        for w in ws:
            n += 1
            reqs = [GenRequest(f"cal-d{n}-{i}#0", tuple((i * 7 + j) % vocab for j in range(plen)), decode_len,
                               temperature=1.0, ignore_eos=True) for i in range(w)]
            before = await server_metrics() if server_metrics else None
            wall = await _timed(engine, reqs)
            server = await server_delta(before, "vllm:inter_token_latency_seconds")
            pts.append({"kind": "decode", "width": w, "prompt_len": plen, "context": w * (plen + decode_len / 2),
                        "prefill": w * (plen - 1) / decode_len, "client_seconds": wall / decode_len,
                        "seconds": server if server is not None else wall / decode_len})
    for p in prefill_lens:
        for w in prefill_widths:
            n += 1
            per_step = max(1, min(w, max_num_batched_tokens // p))
            reqs = [GenRequest(f"cal-p{n}-{i}#0", tuple((i * 13 + j) % vocab for j in range(p)), 1, temperature=1.0)
                    for i in range(w)]
            before = await server_metrics() if server_metrics else None
            wall = await _timed(engine, reqs)
            server = await server_delta(before, "vllm:request_prefill_time_seconds")
            steps = -(-w // per_step)
            pts.append({"kind": "prefill", "width": per_step, "context": per_step * p, "prefill": per_step * (p - 1),
                        "client_seconds": wall / steps, "seconds": server if server is not None else wall / steps})
    return pts


def fit(points: list[dict]) -> dict:
    """CostModel in two stages: alpha, beta, gamma by least squares on the decode points (clean, one step each),
    then delta as the median per-token residual of the prefill points (few and noisy: a joint fit lets them bend
    the decode terms, even negative). Errors are reported per kind."""
    import numpy as np

    from halfwritten.engines.fake import CostModel
    dec = [p for p in points if p["kind"] == "decode"]
    pre = [p for p in points if p["kind"] == "prefill"]
    X = np.column_stack([np.ones(len(dec)), [p["width"] for p in dec], [p["context"] for p in dec]])
    (alpha, beta, gamma), *_ = np.linalg.lstsq(X, np.array([p["seconds"] for p in dec]), rcond=None)
    alpha, beta, gamma = max(alpha, 0.0), max(beta, 0.0), max(gamma, 0.0)
    resid = [(p["seconds"] - alpha - beta * p["width"] - gamma * p["context"]) / p["prefill"] for p in pre
             if p["prefill"] > 0]
    delta = max(float(np.median(resid)), 0.0) if resid else 0.0
    cm = CostModel(float(alpha), float(beta), float(gamma), delta)
    err = lambda ps: [abs(cm.step_time(p["width"], p["context"], p["prefill"]) - p["seconds"]) / p["seconds"]  # noqa
                      for p in ps]
    e_dec, e_pre = err(dec), err(pre)
    return {"cost": {"alpha": cm.alpha, "beta": cm.beta, "gamma": cm.gamma, "delta": cm.delta},
            "rel_error_median": float(np.median(e_dec + e_pre)), "rel_error_max": float(max(e_dec + e_pre)),
            "decode_rel_error_max": float(max(e_dec)) if e_dec else None,
            "prefill_rel_error_median": float(np.median(e_pre)) if e_pre else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], help="config overrides, e.g. launch.head_fp32=true")
    ap.add_argument("--out", default="calibration.json")
    a = ap.parse_args(argv)
    from halfwritten.config import load, with_overrides
    from halfwritten.engines.vllm_http import run_uvloop
    from halfwritten.launch import tool_config, tool_engine
    cfg = tool_config(with_overrides(load(a.config), a.set))
    out = Path(a.out)
    with tool_engine(cfg, out.parent / "logs-calibrate") as (url, engine):
        kv = run_uvloop(engine.start()).kv_tokens
        points = run_uvloop(measure(engine, max_model_len=cfg.max_model_len,
                             max_num_batched_tokens=max(8192, cfg.max_model_len), server_metrics=engine.metrics))
    result = {"model": cfg.model, "gpu_memory_utilization": cfg.launch.gpu_memory_utilization, "kv_tokens": kv,
              "points": points, **fit(points)}
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "points"}, indent=1))


if __name__ == "__main__":
    main()
