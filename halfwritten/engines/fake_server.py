"""A stand-in for `vllm serve` (v0.30, VLLM_SERVER_DEV_MODE=1) over a FakeEngine: the endpoints and chunk shapes
VllmHttpEngine uses, served by aiohttp on a real socket in real time. Tests run it in their own event loop; a run
launches it as a subprocess, exactly where `vllm serve` would be:

    python -m halfwritten.engines.fake_server --port 8123 --lengths lognormal:120,1.0 --vocab 512

Weights: POST /update_weight_version applies the version as the engine's new weights (the trainer's VersionSink),
and the four weight-transfer routes accept a push whose finish names the version.
"""
import argparse
import asyncio
import json

from aiohttp import web

from ..lengths import parse
from ..types import GenRequest
from .fake import FakeEngine


def make_app(engine: FakeEngine, model: str = "fake") -> web.Application:
    async def completions(request: web.Request):
        body = await request.json()
        rid = request.headers["X-Request-Id"]
        adapter = body["model"] if body["model"] != model else None
        req = GenRequest(rid, tuple(body["prompt"]), body["max_tokens"], body.get("temperature", 1.0), adapter,
                         ignore_eos=body.get("ignore_eos", False))
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        async for c in engine.generate(req):
            choice = {"index": 0, "text": "", "token_ids": list(c.token_ids),
                      "logprobs": {"token_logprobs": list(c.logprobs)}, "finish_reason": c.finish}
            await resp.write(f"data: {json.dumps({'id': rid, 'choices': [choice]})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def pause(request: web.Request):
        await engine.pause(request.query.get("mode", "abort"), request.query.get("clear_cache", "true") == "true")
        return web.json_response({})

    async def resume(request):
        await engine.resume()
        return web.json_response({})

    async def is_paused(request):
        return web.json_response({"is_paused": await engine.is_paused()})

    async def weight_info(request):
        return web.json_response({"weight_version": str(await engine.weight_version())})

    async def load(request):
        body = await request.json()
        await engine.load_adapter(body["lora_name"], body["lora_path"], body.get("load_inplace", False))
        return web.Response(text="ok")

    async def unload(request):
        await engine.unload_adapter((await request.json())["lora_name"])
        return web.Response(text="ok")

    async def metrics(request):
        c = await engine.counters()
        label = f'{{model_name="{model}"}}'
        lines = [f"vllm:num_requests_running{label} {c.running}", f"vllm:num_requests_waiting{label} {c.waiting}",
                 f"vllm:num_preemptions_total{label} {c.preemptions_total}",
                 f"vllm:generation_tokens_total{label} {c.generated_tokens_total}",
                 f"vllm:prompt_tokens_total{label} {c.prefill_tokens_total}",
                 f"vllm:kv_cache_usage_perc{label} {c.kv_usage}",
                 'vllm:cache_config_info{block_size="16",num_gpu_blocks="65536"} 1.0']
        return web.Response(text="\n".join(lines) + "\n")

    async def health(request):
        return web.Response()

    async def update_version(request):
        engine.apply_weights(int(str((await request.json())["new_version"]).removeprefix("v")))
        return web.json_response({})

    async def transfer_ok(request):
        return web.json_response({})

    async def finish_update(request):
        body = await request.json() if request.can_read_body else {}
        if (v := (body or {}).get("weight_version")) is not None:
            engine.apply_weights(int(str(v).removeprefix("v")))
        return web.json_response({})

    async def models(request):
        return web.json_response({"data": [{"id": model}]})

    app = web.Application()
    app.add_routes([web.post("/v1/completions", completions), web.post("/pause", pause), web.post("/resume", resume),
                    web.get("/is_paused", is_paused), web.get("/weight_info", weight_info),
                    web.post("/v1/load_lora_adapter", load), web.post("/v1/unload_lora_adapter", unload),
                    web.get("/metrics", metrics), web.get("/health", health),
                    web.post("/update_weight_version", update_version),
                    web.post("/init_weight_transfer_engine", transfer_ok),
                    web.post("/start_weight_update", transfer_ok), web.post("/update_weights", transfer_ok),
                    web.post("/finish_weight_update", finish_update), web.get("/v1/models", models)])
    return app


async def serve(engine: FakeEngine, port: int = 0) -> tuple[web.AppRunner, str]:
    """Starts the app (port 0: an ephemeral one); returns the runner (cleanup()) and the base URL. A run can end with
    streams still open (the engine paused at the budget, or answers the controller abandoned); cleanup() waits at most
    a second for them, not aiohttp's default 60 s: a stand-in has nothing worth draining."""
    runner = web.AppRunner(make_app(engine), handle_signals=False, shutdown_timeout=1.0)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


def main(argv=None):
    p = argparse.ArgumentParser(description="a stand-in for `vllm serve` over FakeEngine")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--lengths", default="lognormal:200,0.8")
    p.add_argument("--vocab", type=int, default=151_936)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=4096)
    a = p.parse_args(argv)

    async def run():
        engine = FakeEngine(lengths=parse(a.lengths), vocab=a.vocab, max_num_seqs=a.max_num_seqs,
                            max_model_len=a.max_model_len)
        await engine.start()
        await serve(engine, a.port)
        await asyncio.Event().wait()

    asyncio.run(run())


if __name__ == "__main__":
    main()
