"""VllmHttpEngine against a stand-in `vllm serve` on a real socket: the adapter parses streams and admin calls, and
the controller over HTTP keeps an exact ledger, aborts included."""
import asyncio

import pytest

from bench.canary import compare
from halfwritten.config import RunConfig
from halfwritten.controller import Controller
from halfwritten.engines.base import EngineError
from halfwritten.engines.fake import CostModel, FakeEngine, LengthModel
from halfwritten.engines.fake_server import serve
from halfwritten.engines.vllm_http import (
    VllmHttpEngine,
    kv_tokens,
    parse_chunk,
    parse_metrics,
    serve_command,
)
from halfwritten.rewards import sim_reward
from halfwritten.tasks import synthetic
from halfwritten.trace import Tracer
from halfwritten.trainer.client import SimTrainer
from halfwritten.types import GenRequest

FAST = CostModel(alpha=2e-3, beta=2e-5, gamma=0.0, delta=1e-6)


def test_serve_command_sets_the_flags_every_arm_needs():
    argv, env = serve_command("Qwen/Qwen3-0.6B-Base", max_num_seqs=128, async_scheduling=False, lora_rank=1)
    for flag in ("processed_logprobs", "--no-enable-prefix-caching", "--no-enable-chunked-prefill",
                 "--no-async-scheduling", "--enable-lora", "--weight-transfer-config"):
        assert flag in argv
    assert argv[argv.index("--max-num-seqs") + 1] == "128"
    assert env == {"VLLM_SERVER_DEV_MODE": "1", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"}
    assert "--no-async-scheduling" not in serve_command("m")[0]
    # loopback only (external review F08): the admin endpoints are unauthenticated and vllm binds all interfaces
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    # the exact model files, and the engine's sampler seed (external review F12)
    assert argv[argv.index("--revision") + 1] == "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    assert serve_command("m", seed=3)[0][serve_command("m", seed=3)[0].index("--seed") + 1] == "3"
    assert "--revision" not in serve_command("local/path")[0]


def test_parsers():
    c = parse_chunk('{"choices": [{"text": "x", "token_ids": [5, 6], "logprobs": {"token_logprobs": [-0.1, -2],'
                    ' "tokens": ["a", "b"]}, "finish_reason": null, "prompt_token_ids": [1, 2]}]}')
    assert c.token_ids == (5, 6) and c.logprobs == (-0.1, -2.0) and c.finish is None
    abort = parse_chunk(b'{"choices": [{"token_ids": [], "logprobs": null, "finish_reason": "abort"}]}')
    assert abort.finish == "abort"
    for bad in ('{"choices": [{"token_ids": [5], "logprobs": {"token_logprobs": []}}]}',  # ids without logprobs
                '{"choices": [{"token_ids": [5], "logprobs": {"token_logprobs": [null]}}]}',  # a non-number
                '{"choices": [{"token_ids": [5]'):  # cut short
        with pytest.raises(EngineError):
            parse_chunk(bad)
    text = 'vllm:num_requests_running{model_name="a"} 3\nvllm:num_requests_running{model_name="b"} 2\n# HELP x\n'
    assert parse_metrics(text) == {"vllm:num_requests_running": 5.0}
    assert kv_tokens('vllm:cache_config_info{block_size="16",num_gpu_blocks="100"} 1.0') == 1600


def test_stream_and_admin_round_trip():
    async def main():
        fake = FakeEngine(lengths=LengthModel(mean=20, sigma=0.0), cost=FAST)
        await fake.start()
        runner, url = await serve(fake)
        eng = VllmHttpEngine(url, "fake", max_num_seqs=64, max_model_len=4096)
        try:
            info = await eng.start()
            chunks = [c async for c in eng.generate(GenRequest("a#0", (1, 2, 3), 64))]
            await eng.pause("keep", False)
            paused = await eng.is_paused()
            fake.apply_weights(3)
            await eng.resume()
            return info, chunks, paused, await eng.weight_version(), await eng.counters()
        finally:
            await eng.close()
            await runner.cleanup()
            await fake.close()

    info, chunks, paused, version, counters = asyncio.run(main())
    assert info.kv_tokens == 16 * 65536 and info.max_num_seqs == 64
    assert sum(len(c.token_ids) for c in chunks) == 20 and chunks[-1].finish == "stop"
    assert paused and version == 3 and counters.generated_tokens_total == 20


@pytest.mark.parametrize("regime", ["D", "C1", "E"])
def test_controller_over_http_keeps_an_exact_ledger(regime):
    async def main():
        loop = asyncio.get_running_loop()
        fake = FakeEngine(lengths=LengthModel(mean=60, sigma=0.9), cost=FAST, max_num_seqs=64)
        runner, url = await serve(fake)
        eng = VllmHttpEngine(url, "fake", max_num_seqs=64, max_model_len=4096)
        trainer = SimTrainer(fake, Tracer(loop.time, "trainer"), s_per_token=2e-5, s_fixed=0.02, push_s=0.01)
        # real time: the quiet window must outlast a chunk's delivery even on a loaded machine (E2, PLAN.md §5.3)
        cfg = RunConfig(regime=regime, B=4, G=4, M=4, K=2, max_tokens=256, budget_s=3.0)
        controller = Controller(cfg, eng, trainer, synthetic(64, 16), sim_reward())
        try:
            await fake.start()  # the server's engine; the controller only starts its HTTP view of it
            summary = await controller.run()
        finally:
            await runner.cleanup()
        return summary, compare(controller.ledger, fake.truth)

    summary, report = asyncio.run(main())
    assert summary.valid, (summary.error, summary.violations[:3])
    assert summary.steps >= 2 and len(summary.syncs) >= 2
    assert report["ok"], report


def test_stream_errors_and_forced_lengths_cross_the_wire():
    with pytest.raises(EngineError, match="out of memory"):
        parse_chunk('{"error": {"message": "CUDA out of memory", "code": 500}}')
    with pytest.raises(EngineError, match="bad prompt"):
        parse_chunk('{"object": "error", "message": "bad prompt", "code": 400}')
    with pytest.raises(EngineError):
        parse_chunk('{"id": "x"}')

    async def main():
        fake = FakeEngine(lengths=LengthModel(mean=5, sigma=0.0), cost=FAST)
        await fake.start()
        runner, url = await serve(fake)
        eng = VllmHttpEngine(url, "fake", max_num_seqs=64, max_model_len=4096)
        try:
            return [c async for c in eng.generate(GenRequest("a#0", (1, 2), 40, ignore_eos=True))]
        finally:
            await eng.close()
            await runner.cleanup()
            await fake.close()

    chunks = asyncio.run(main())
    assert sum(len(c.token_ids) for c in chunks) == 40 and chunks[-1].finish == "length"  # past its natural EOS


def test_serve_command_for_sizes_the_server_from_the_run():
    from halfwritten.config import RunConfig
    from halfwritten.engines.vllm_http import serve_command_for
    argv, env = serve_command_for(RunConfig(regime="F", lora=True, K=8, M=16, G=8, lora_rank=1), model="m")
    assert argv[argv.index("--max-loras") + 1] == "10" and argv[argv.index("--max-num-seqs") + 1] == "128"
    argv, _ = serve_command_for(RunConfig(regime="D", K=4, M=4, G=8), model="m")
    assert "--enable-lora" not in argv


def test_one_engine_serves_several_event_loops():
    """bench scripts call asyncio.run more than once with one engine; pooled connections of a closed loop must not
    be reused (it raised "Event loop is closed")."""
    import subprocess
    import sys
    import time

    from halfwritten.engines.vllm_http import healthy
    port = 8195
    p = subprocess.Popen([sys.executable, "-m", "halfwritten.engines.fake_server", "--port", str(port), "--lengths",
                          "lognormal:20,0.3", "--vocab", "512", "--max-num-seqs", "16", "--max-model-len", "512"])
    try:
        for _ in range(200):
            if healthy(f"http://127.0.0.1:{port}"):
                break
            time.sleep(0.05)
        e = VllmHttpEngine(f"http://127.0.0.1:{port}", "fake", max_num_seqs=16, max_model_len=512)

        async def gen():
            n = 0
            async for c in e.generate(GenRequest("x#0", (1, 2, 3), 10)):
                n += len(c.token_ids)
            return n
        asyncio.run(e.start())
        assert asyncio.run(gen()) > 0 and asyncio.run(gen()) > 0
        asyncio.run(e.close())
    finally:
        p.terminate()
        p.wait(10)


def test_streams_survive_unicode_line_separators_inside_json():
    """vLLM writes UTF-8 JSON raw, so a token decoding to U+2028 (or U+0085, U+2029) sits unescaped inside a data
    line; a reader splitting like str.splitlines (httpx's aiter_lines did) cuts it and the half-line fails to parse
    (seen on the A100)."""
    import json as _json

    from aiohttp import web

    async def completions(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i, text in enumerate(["a b", "c\u0085d e\x1c"]):
            body = {"choices": [{"index": 0, "text": text, "token_ids": [10 + i],
                                 "logprobs": {"token_logprobs": [-0.5]}, "finish_reason": "length" if i else None}]}
            await resp.write(("data: " + _json.dumps(body, ensure_ascii=False) + "\n\n").encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def main():
        app = web.Application()
        app.router.add_post("/v1/completions", completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        e = VllmHttpEngine(f"http://127.0.0.1:{port}", "m", max_num_seqs=4, max_model_len=64)
        try:
            return [c async for c in e.generate(GenRequest("u#0", (1, 2), 2))]
        finally:
            await e.close()
            await runner.cleanup()

    chunks = asyncio.run(main())
    assert [list(c.token_ids) for c in chunks] == [[10], [11]] and chunks[-1].finish == "length"


def test_fp32_head_is_vllms_head_dtype_override():
    from halfwritten.config import RunConfig, with_overrides
    from halfwritten.engines.vllm_http import serve_command_for
    argv, _ = serve_command(model="m", head_fp32=True)
    assert argv[argv.index("--hf-overrides") + 1] == '{"head_dtype": "float32"}'
    assert "--hf-overrides" not in serve_command(model="m")[0]
    cfg = with_overrides(RunConfig(model="m"), ["launch.head_fp32=true"])
    argv, _ = serve_command_for(cfg, head_fp32=cfg.launch.head_fp32)
    assert "--hf-overrides" in argv


def test_pause_latency_benchmark_runs_against_the_fake_server():
    from bench.pause_latency import measure
    from halfwritten.engines.fake_server import serve

    async def main():
        fake = FakeEngine(lengths=LengthModel(mean=50, sigma=0.3), cost=CostModel(2e-3, 0, 0, 0), vocab=512)
        await fake.start()
        runner, url = await serve(fake)
        engine = VllmHttpEngine(url, "fake", max_num_seqs=16, max_model_len=4096)
        try:
            return await measure(engine, url, streams=4, pauses=2, max_tokens=400, settle_s=0.2, gap_s=0.05)
        finally:
            await engine.close()
            await runner.cleanup()

    r = asyncio.run(main())
    assert r["pauses"] == 2 and r["tokens_per_s"] > 0 and r["pause_in_loop_p50"] >= 0
    assert r["side_request_curl_p50"] is not None and r["loop_lag_p50"] is not None


def test_api_servers_and_stream_interval_reach_the_command():
    argv, _ = serve_command(model="m", api_servers=4, stream_interval=8)
    assert argv[argv.index("--api-server-count") + 1] == "4" and argv[argv.index("--stream-interval") + 1] == "8"
    assert "--api-server-count" not in serve_command(model="m")[0]


def test_a_sigterm_ends_the_run_with_its_row_and_a_second_exits_at_once():
    """The first SIGTERM stops a live run as a budget would (its summary comes back, invalid); a second one exits
    without waiting for the trainer's goodbye. The handler in place before the run is restored either way."""
    import os
    import signal

    from halfwritten.launch import sigterm_stops

    def run_once(kills):
        async def main():
            loop = asyncio.get_running_loop()
            fake = FakeEngine(lengths=LengthModel(mean=60, sigma=0.9), cost=FAST, max_num_seqs=64)
            trainer = SimTrainer(fake, Tracer(loop.time, "trainer"), s_per_token=2e-5, s_fixed=0.02, push_s=0.01)
            cfg = RunConfig(regime="D", B=4, G=4, M=4, K=2, max_tokens=256, budget_s=30.0)
            controller = Controller(cfg, fake, trainer, synthetic(64, 16), sim_reward())
            loop.call_later(0.5, lambda: [os.kill(os.getpid(), signal.SIGTERM) for _ in range(kills)])
            with sigterm_stops(controller, grace_s=5.0):
                return await controller.run(), controller, loop.time()
        return asyncio.run(main())

    before = signal.getsignal(signal.SIGTERM)
    summary, controller, _ = run_once(1)
    assert controller.signalled and not summary.valid and "stopped by a signal" in summary.error
    assert summary.steps >= 1 and signal.getsignal(signal.SIGTERM) is before
    with pytest.raises(SystemExit) as e:
        run_once(2)
    assert e.value.code == 128 + signal.SIGTERM and signal.getsignal(signal.SIGTERM) is before


def test_the_host_watch_measures_only_the_live_run():
    """A stall while the run is live is recorded with its time; the end-of-run summary (not live) is not."""
    import time

    from halfwritten.launch import host_watch

    async def main():
        live = [True]
        async with host_watch(period=0.01, live=lambda: live[0]) as host:
            await asyncio.sleep(0.05)
            time.sleep(0.2)  # a live stall
            await asyncio.sleep(0.05)
            live[0] = False
            await asyncio.sleep(0.02)
            time.sleep(0.5)  # the summary, after the run
            await asyncio.sleep(0.02)
        return host

    host = asyncio.run(main())
    assert 0.15 < host["loop_lag_s"]["p100"] < 0.4 and 0.04 < host["worst_stall_at_s"] < 0.15


def test_the_host_watch_starts_with_the_run_clock():
    """Engine startup and the first push come before the run's clock starts (controller.t_start): a stall there is
    not the live run's, and the worst stall's time is on the run's own clock."""
    import time

    from halfwritten.launch import host_watch

    async def main():
        loop, t = asyncio.get_running_loop(), {}
        async with host_watch(period=0.01, start=lambda: t.get("start")) as host:
            await asyncio.sleep(0.02)
            time.sleep(0.4)  # startup: not live
            await asyncio.sleep(0.02)
            t["start"] = loop.time()
            await asyncio.sleep(0.1)
            time.sleep(0.15)  # live, 0.1 s into the run
            await asyncio.sleep(0.05)
        return host

    host = asyncio.run(main())
    assert 0.1 < host["loop_lag_s"]["p100"] < 0.3 and 0.07 < host["worst_stall_at_s"] < 0.16


def test_idle_connections_close_before_the_servers_keepalive_does():
    """vLLM closes an idle connection after 5 s (VLLM_HTTP_TIMEOUT_KEEP_ALIVE): a client that kept it longer could
    send /resume on a socket being closed (seen after a lock-step B drain)."""
    from halfwritten.engines.vllm_http import VllmHttpEngine

    async def main():
        eng = VllmHttpEngine("http://127.0.0.1:1", "m", max_num_seqs=1, max_model_len=16)
        session = eng._session
        keep = session.connector._keepalive_timeout
        await session.close()
        return keep

    assert asyncio.run(main()) < 5


def test_the_inplace_fix_clears_the_stored_flag_after_the_load(monkeypatch):
    """launch.lora_inplace_fix (engines/vllm_serve.py): the load itself runs in place, and the record later requests
    carry no longer says so, so the engine reloads once, not on every step. vLLM's class is stood in for."""
    import sys
    import types

    from halfwritten.engines import vllm_serve

    loads = []

    class Models:  # vllm.entrypoints.openai.models.serving.OpenAIServingModels, as far as the patch reads it
        def __init__(self):
            self.lora_requests = {}

        async def load_lora_adapter(self, request, base_model_name=None):
            loads.append(request.load_inplace)  # what the engine saw at the load
            self.lora_requests[request.lora_name] = request
            return "ok"

    module = types.ModuleType("serving")
    module.OpenAIServingModels = Models
    for name in ("vllm", "vllm.entrypoints", "vllm.entrypoints.openai", "vllm.entrypoints.openai.models"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.models.serving", module)
    vllm_serve.clear_inplace_flag()
    vllm_serve.clear_inplace_flag()  # idempotent
    models, request = Models(), types.SimpleNamespace(lora_name="policy", load_inplace=True)
    assert asyncio.run(models.load_lora_adapter(request)) == "ok"
    assert loads == [True] and models.lora_requests["policy"].load_inplace is False
