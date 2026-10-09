"""The Engine interface over stock `vllm serve` (v0.30.0) with VLLM_SERVER_DEV_MODE=1, the study's only engine path.

Generation streams /v1/completions with prompt token ids and the vLLM extensions return_token_ids (token ids per
chunk) and logprobs=0 (the processed logprob of each sampled token, with --logprobs-mode processed_logprobs). Our
request id travels as X-Request-Id. The RL endpoints (vllm/entrypoints/serve/dev/rlhf/api_router.py) pause, resume
and report the weight version; weights arrive from the trainer over NCCL (trainer/sync.py), never through here.

The field names in a streamed chunk are taken from the 0.30 source; Phase 0 records a golden SSE transcript from a
real server (tests/fixtures) and test_vllm_http parses it, so a renamed field fails a CPU test, not a GPU run.
"""
import asyncio
import json
import re
from typing import Any

import aiohttp
import msgspec

from ..revisions import model_revision
from ..types import Chunk, GenRequest, PauseMode
from .base import EngineCounters, EngineError, EngineInfo

# RunConfig knobs -> server flags; every arm runs with the same set (PLAN.md §4.4)
# --generation-config vllm: vLLM's default ("auto") fills unset top_k / min_p / repetition_penalty from the model's
# generation_config.json (post-trained Qwen3 ships top_k 20), which would truncate mu against the trainer's full pi.
ENGINE_FLAGS = ["--logprobs-mode", "processed_logprobs", "--no-enable-prefix-caching", "--no-enable-chunked-prefill",
                "--generation-config", "vllm", "--stream-interval", "1"]


def serve_command(model: str, *, port: int = 8000, max_num_seqs: int = 256, max_model_len: int = 4096,
                  max_num_batched_tokens: int = 8192, gpu_memory_utilization: float = 0.9,
                  async_scheduling: bool = True, lora_rank: int | None = None, max_loras: int = 6,
                  weight_backend: str = "nccl", head_fp32: bool = False, stream_interval: int = 1,
                  api_servers: int = 1, seed: int = 0, extra: list[str] = ()) -> tuple[list[str], dict]:
    """argv and environment for `vllm serve`. seed is the engine's RNG seed: vLLM 0.30's V2 model runner draws each
    request's sampling seed from it (requests carry none), so seed blocks must differ in it as in everything else.
    head_fp32 has the LM head write its logits in fp32 (vLLM's
    `head_dtype` override, made for RL's training-inference consistency): bf16 logits are rounded by up to half an
    ulp, 0.0625 at a logit of 20 (docs/fp32_lm_head.md). Chunked prefill is off so that no prompt is half-prefilled at
    a pause (its KV would mix versions with no token delivered). Without chunking a prefill must fit whole in one
    step, and E and C1 re-prefill a whole context (prompt + answer so far), so max_num_batched_tokens must cover
    max_model_len. async_scheduling follows the session's canary (PLAN.md §5.3). max_loras bounds the adapters loaded
    at once: F holds up to K+2 versions (serve_command_for sets it). weight_backend "ipc" serves a trainer on the same
    GPU (colocated); its CUDA handles travel pickled over HTTP, which vLLM allows only with
    VLLM_ALLOW_INSECURE_SERIALIZATION=1."""
    if max_num_batched_tokens < max_model_len:
        raise ValueError(f"max_num_batched_tokens {max_num_batched_tokens} < max_model_len {max_model_len}: with "
                         "chunked prefill off, a re-prefilled context could never be scheduled")
    # loopback only: the pause, weight-update, abort and dev endpoints are unauthenticated even with --api-key, and
    # vllm serve binds every interface by default (docs.vllm.ai/en/latest/usage/security); operators use ssh -L
    argv = ["vllm", "serve", model, "--host", "127.0.0.1", "--port", str(port), "--max-num-seqs", str(max_num_seqs),
            "--max-model-len", str(max_model_len), "--max-num-batched-tokens", str(max_num_batched_tokens),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--weight-transfer-config", json.dumps({"backend": weight_backend}), *ENGINE_FLAGS]
    argv += ["--seed", str(seed)]
    if (rev := model_revision(model)) is not None:  # the model's exact files (halfwritten/revisions.py)
        argv += ["--revision", rev, "--tokenizer-revision", rev]
    if stream_interval != 1:  # tokens per streamed chunk; the canary must pass with it (bench/pause_latency.py)
        argv[argv.index("--stream-interval") + 1] = str(stream_interval)
    if api_servers > 1:  # API server processes sharing one engine core: streaming work spread over processes
        argv += ["--api-server-count", str(api_servers)]
    if not async_scheduling:
        argv.append("--no-async-scheduling")
    if head_fp32:
        argv += ["--hf-overrides", json.dumps({"head_dtype": "float32"})]
    env = {"VLLM_SERVER_DEV_MODE": "1"}
    if weight_backend == "ipc":
        env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    if lora_rank is not None:
        argv += ["--enable-lora", "--max-lora-rank", str(lora_rank), "--max-loras", str(max_loras)]
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    return argv + list(extra), env


def serve_command_for(cfg, *, model: str | None = None, port: int = 8000, async_scheduling: bool = True,
                      gpu_memory_utilization: float = 0.9, weight_backend: str = "nccl", head_fp32: bool = False,
                      stream_interval: int = 1, api_servers: int = 1, extra: list[str] = ()) -> tuple[list[str], dict]:
    """serve_command() for a RunConfig: concurrency M*G (the controller never lets vLLM queue), the run's context
    length, and LoRA slots for every adapter version that can be in use at once (F: K+2; in place: 1)."""
    from ..regimes import resolve
    weights = resolve(cfg.regime, cfg.lora).weights
    lora_rank = None if weights == "full" else cfg.lora_rank
    max_loras = cfg.K + 2 if weights == "lora_versioned" else 1
    return serve_command(model or cfg.model, port=port, max_num_seqs=cfg.M * cfg.G, max_model_len=cfg.max_model_len,
                         max_num_batched_tokens=max(8192, cfg.max_model_len),
                         gpu_memory_utilization=gpu_memory_utilization, async_scheduling=async_scheduling,
                         lora_rank=lora_rank, max_loras=max_loras, weight_backend=weight_backend,
                         head_fp32=head_fp32, stream_interval=stream_interval, api_servers=api_servers,
                         seed=cfg.seed, extra=extra)


class _Logprobs(msgspec.Struct):
    token_logprobs: list[float] = []


class _Choice(msgspec.Struct):
    token_ids: list[int] | None = None
    logprobs: _Logprobs | None = None
    finish_reason: str | None = None


class _Completion(msgspec.Struct):
    """The fields read from a streamed /v1/completions object; the rest (text, the prompt ids echoed in the first
    chunk) is skipped unparsed. Decoding into these types costs a third of json.loads plus picking fields."""
    choices: list[_Choice] = []
    error: Any = None
    message: str | None = None


_DECODE = msgspec.json.Decoder(_Completion)


def parse_chunk(payload: bytes | str) -> Chunk:
    """One streamed /v1/completions data payload -> Chunk. vLLM streams a failure as a data line holding an error
    object (no choices); that ends the request as an EngineError, not a KeyError deep in the controller."""
    try:
        obj = _DECODE.decode(payload)
    except msgspec.DecodeError as e:  # malformed JSON, or a field of the wrong type
        raise EngineError(f"unreadable chunk ({e}): {payload[:200]!r}") from None
    if obj.error is not None or not obj.choices:
        err = obj.error if obj.error is not None else obj.message or payload[:200]
        raise EngineError(f"stream error: {err.get('message', err) if isinstance(err, dict) else err}")
    choice = obj.choices[0]
    ids = tuple(choice.token_ids or ())
    logprobs = tuple(choice.logprobs.token_logprobs) if choice.logprobs else ()
    if len(logprobs) != len(ids):
        raise EngineError(f"chunk has {len(ids)} token ids but {len(logprobs)} logprobs")
    return Chunk(ids, logprobs, choice.finish_reason)


_METRIC = re.compile(r"^(vllm:[a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+|NaN)\s*$")
_CACHE_INFO = re.compile(r'^vllm:cache_config_info\{([^}]*)\}')


def parse_metrics(text: str) -> dict[str, float]:
    """Prometheus text -> {name: value summed over label sets}."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _METRIC.match(line)
        if m and m.group(2) != "NaN":
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def kv_tokens(text: str) -> int:
    """KV capacity in tokens, from the labels of vllm:cache_config_info (num_gpu_blocks x block_size); 0 if absent."""
    for line in text.splitlines():
        if m := _CACHE_INFO.match(line):
            labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            try:
                return int(labels["num_gpu_blocks"]) * int(labels["block_size"])
            except (KeyError, ValueError):
                return 0
    return 0


def healthy(url: str, timeout: float = 2.0) -> bool:
    """Whether the server answers /health with 200. Blocking, for start-up before any event loop exists."""
    import http.client
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=timeout) as r:
            return r.status == 200
    except (OSError, http.client.HTTPException):  # refused, reset, timed out, an error status: not up yet
        return False


_SESSIONS: list[tuple[asyncio.AbstractEventLoop, aiohttp.ClientSession]] = []
# vLLM's server closes a connection idle for VLLM_HTTP_TIMEOUT_KEEP_ALIVE (5 s); aiohttp keeps idle ones for 15 s, so
# a call after a long pause could go out on a socket the server was closing (a lock-step B row on two H100s: /resume
# after a drain, "Server disconnected"). Idle connections are dropped well before the server would drop them.
KEEPALIVE_S = 2.0


def run_uvloop(coro):
    """asyncio.run for everything that talks to a real engine, on uvloop: one controller process reads every
    streamed token, and at ~15k chunks/s on stock asyncio it saturated a core (two H100s, x2f), so old tokens queued
    unread at a pause. The HTTP sessions opened on the loop are closed before it ends."""
    import uvloop

    async def main():
        try:
            return await coro
        finally:
            loop = asyncio.get_running_loop()
            for held in [h for h in _SESSIONS if h[0] is loop]:
                _SESSIONS.remove(held)
                await held[1].close()
    return asyncio.run(main(), loop_factory=uvloop.new_event_loop)


class VllmHttpEngine:
    """`max_num_seqs` and `max_model_len` are what serve_command() launched the server with. The HTTP session
    belongs to the event loop it was made in, so a script calling run() more than once gets a fresh one per loop
    (pooled connections of a closed loop cannot be reused)."""

    def __init__(self, url: str, model: str, *, max_num_seqs: int, max_model_len: int, timeout: float = 600.0):
        self.url, self.model = url.rstrip("/"), model
        self.max_num_seqs, self.max_model_len = max_num_seqs, max_model_len
        self._timeout = aiohttp.ClientTimeout(total=None, sock_connect=10.0, sock_read=timeout)
        self._held: tuple[asyncio.AbstractEventLoop, aiohttp.ClientSession] | None = None

    @property
    def _session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        if self._held is None or self._held[0] is not loop or self._held[1].closed:
            self._held = (loop, aiohttp.ClientSession(
                timeout=self._timeout, connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=KEEPALIVE_S)))
            _SESSIONS[:] = [h for h in _SESSIONS if not h[0].is_closed()]
            _SESSIONS.append(self._held)
        return self._held[1]

    async def request(self, method: str, path: str, **kw) -> bytes:
        """An admin call's response body; a status of 400 or more is an EngineError."""
        async with self._session.request(method, self.url + path, **kw) as r:
            body = await r.read()
            if r.status >= 400:
                raise EngineError(f"{method} {path}: {r.status} {body[:200]!r}")
            return body

    async def start(self) -> EngineInfo:
        await self.request("GET", "/health")
        version = await self.weight_version()
        text = (await self.request("GET", "/metrics")).decode()
        return EngineInfo(self.model, self.max_model_len, self.max_num_seqs, kv_tokens(text), version or 0,
                          {"url": self.url})

    async def stream(self, body: dict, rid: str, raw: bool = False):
        """The Chunks of a streamed /v1/completions request, or (payload, Chunk) pairs with raw=True (transcripts:
        bench/canary.py). One generator, splitting lines itself, because every layer costs CPU on every token.
        Lines split on b"\n" only: vLLM writes JSON as raw UTF-8, so U+2028, U+2029 or U+0085 can sit unescaped
        inside a data line, and a reader splitting like str.splitlines cuts the JSON there."""
        async with self._session.post(self.url + "/v1/completions", json=body, headers={"X-Request-Id": rid}) as r:
            if r.status >= 400:
                raise EngineError(f"{rid}: {r.status} {(await r.read())[:200]!r}")
            buf = b""
            async for data in r.content.iter_any():
                buf += data
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    if not line.startswith(b"data: "):
                        continue
                    payload = line[6:].rstrip(b"\r")
                    if payload == b"[DONE]":
                        return
                    chunk = parse_chunk(payload)
                    yield (payload, chunk) if raw else chunk
                    if chunk.finish:
                        return
            # a last line without its newline is dropped: SSE dispatches an event only at its blank line

    def generate(self, req: GenRequest):
        body = {"model": req.adapter or self.model, "prompt": list(req.prompt_ids), "max_tokens": req.max_tokens,
                "temperature": req.temperature, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
                "logprobs": 0, "stream": True,
                "return_token_ids": True, "skip_special_tokens": False, "ignore_eos": req.ignore_eos}
        if req.seed is not None:
            body["seed"] = req.seed
        if req.stop:
            body["stop"] = list(req.stop)
        return self.stream(body, req.rid)

    async def pause(self, mode: PauseMode, clear_cache: bool) -> None:
        await self.request("POST", "/pause", params={"mode": mode, "clear_cache": str(clear_cache).lower()})

    async def resume(self) -> None:
        await self.request("POST", "/resume")

    async def is_paused(self) -> bool:
        return bool(json.loads(await self.request("GET", "/is_paused"))["is_paused"])

    async def weight_version(self) -> int | None:
        """Our integer version, or None before the first push (vLLM 0.30 starts at "default")."""
        v = json.loads(await self.request("GET", "/weight_info")).get("weight_version")
        try:
            return int(str(v).removeprefix("v"))
        except (TypeError, ValueError):
            return None

    async def load_adapter(self, name: str, path: str, inplace: bool) -> None:
        await self.request("POST", "/v1/load_lora_adapter",
                           json={"lora_name": name, "lora_path": path, "load_inplace": inplace})

    async def unload_adapter(self, name: str) -> None:
        await self.request("POST", "/v1/unload_lora_adapter", json={"lora_name": name})

    async def metrics(self) -> dict[str, float]:
        """Every vllm: series from /metrics, summed over label sets (histograms as _sum and _count)."""
        return parse_metrics((await self.request("GET", "/metrics")).decode())

    async def counters(self) -> EngineCounters:
        m = await self.metrics()
        return EngineCounters(int(m.get("vllm:num_requests_running", 0)), int(m.get("vllm:num_requests_waiting", 0)),
                              int(m.get("vllm:num_preemptions_total", 0)),
                              int(m.get("vllm:generation_tokens_total", 0)), int(m.get("vllm:prompt_tokens_total", 0)),
                              m.get("vllm:kv_cache_usage_perc", m.get("vllm:gpu_cache_usage_perc", 0.0)))

    async def close(self) -> None:
        held, self._held = self._held, None
        if held is not None and held[0] is asyncio.get_running_loop():
            if held in _SESSIONS:
                _SESSIONS.remove(held)
            await held[1].close()
