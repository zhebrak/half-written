"""How long a pause takes under load, and whose fault it is (engine, HTTP server, or our client's event loop).

    python -m bench.pause_latency CONFIG [--set ...] [--streams 16,64,256] [--pauses 8] [--out pause_latency.json]

The two-GPU D row paused for 0.94 s at p50 with 256 answers streaming, against 26 ms with 64 on one GPU. For each
concurrency the engine generates that many long answers (ignore_eos) through the same client the controller uses,
then pauses (keep, no cache clear) and resumes a few times. Each pause is timed three ways:
    in_loop   POST /pause from the event loop that also consumes every stream (what the controller measures)
    curl      the same request from a separate process, at the same moment (the server's own latency)
    loop_lag  how late a 5 ms asyncio sleep wakes in that loop, and its main thread's CPU (how busy our client
              is; launch.host_watch, as the runs record it)
If in_loop is slow but curl fast, the client is the bottleneck; if both are slow, the server or the engine.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from halfwritten.engines.base import collect
from halfwritten.types import GenRequest


async def measure(engine, url: str, streams: int, pauses: int, max_tokens: int = 1800, settle_s: float = 4.0,
                  gap_s: float = 1.5) -> dict:
    from halfwritten.launch import host_watch
    tokens, chunks = [0], [0]

    async def one(i):
        req = GenRequest(f"pl-{streams}-{i}#0", tuple((i * 11 + j) % 1000 + 10 for j in range(64)), max_tokens,
                         temperature=1.0, ignore_eos=True)
        def count(c):
            tokens[0] += len(c.token_ids)
            chunks[0] += 1
        try:
            await collect(engine, req, on_chunk=count)
        except asyncio.CancelledError:
            pass

    async with host_watch(0.005) as host:  # the runs' own meter: the client loop's lag and main-thread CPU
        tasks = [asyncio.create_task(one(i)) for i in range(streams)]
        await asyncio.sleep(settle_s)
        t0, n0, c0 = time.perf_counter(), tokens[0], chunks[0]
        in_loop, curl = [], []
        for _ in range(pauses):
            cmd = ["curl", "-s", "-o", "/dev/null", "-w", "%{time_total}",
                   f"{url}/is_paused"]  # a cheap request alongside, timed out of process
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE)
            t = time.perf_counter()
            await engine.pause("keep", False)
            in_loop.append(time.perf_counter() - t)
            out, _ = await proc.communicate()
            try:
                curl.append(float(out.decode() or "nan"))
            except ValueError:
                curl.append(float("nan"))
            await asyncio.sleep(0.1)
            await engine.resume()
            await asyncio.sleep(gap_s)
        dt = time.perf_counter() - t0
        rate, chunk_rate = (tokens[0] - n0) / dt, (chunks[0] - c0) / dt
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    q = lambda v, p: float(np.nanquantile(v, p)) if v else None  # noqa: E731
    return {"streams": streams, "tokens_per_s": rate, "chunks_per_s": chunk_rate,
            "pause_in_loop_p50": q(in_loop, 0.5), "pause_in_loop_p90": q(in_loop, 0.9),
            "side_request_curl_p50": q(curl, 0.5), "side_request_curl_p90": q(curl, 0.9),
            "loop_lag_p50": host["loop_lag_s"].get("p50"), "loop_lag_p99": host["loop_lag_s"].get("p99"),
            "main_thread_cpu_frac": host["main_thread_cpu_frac"], "pauses": len(in_loop)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--streams", default="16,64,256")
    ap.add_argument("--pauses", type=int, default=8)
    ap.add_argument("--out", default="pause_latency.json")
    a = ap.parse_args(argv)
    from halfwritten.config import load, with_overrides
    from halfwritten.engines.vllm_http import run_uvloop
    from halfwritten.launch import tool_config, tool_engine
    widths = [int(x) for x in a.streams.split(",")]
    cfg = tool_config(with_overrides(load(a.config), a.set), f"G={max(8, -(-max(widths) // 32))}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with tool_engine(cfg, out.parent / "logs-pause") as (url, engine):

        async def sweep():
            res = []
            for n in widths:
                r = await measure(engine, url, n, a.pauses)
                print(json.dumps(r), flush=True)
                res.append(r)
            return res
        rows = run_uvloop(sweep())
    result = {"config": a.config, "set": a.set, "stream_interval": cfg.launch.stream_interval, "rows": rows}
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
