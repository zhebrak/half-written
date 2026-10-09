"""The engine interface the controller drives. engines/vllm_http.py implements it over stock `vllm serve`
(VLLM_SERVER_DEV_MODE=1); engines/fake.py implements it in virtual time with ground truth for the tests.

Weights do not pass through this interface: the trainer pushes them to the engine directly (NCCL, or an adapter
file the controller then loads), and the controller only pauses, checks the version and resumes.
"""
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from ..types import Chunk, GenRequest, PauseMode


class EngineError(RuntimeError):
    pass


@dataclass
class Collected:
    """One answer, whole: what a tool reads after streaming a request to its end."""
    tokens: list[int]
    logprobs: list[float]
    finish: str | None


async def collect(engine, req, on_chunk=None) -> Collected:
    """Streams `req` through `engine` to its end and returns the answer; `on_chunk(chunk)` sees each chunk as it
    arrives (a tool counting throughput). The one loop every tool shares, where six copies stood."""
    tokens, logprobs, finish = [], [], None
    async for c in engine.generate(req):
        tokens += c.token_ids
        logprobs += getattr(c, "logprobs", None) or []
        finish = c.finish or finish
        if on_chunk is not None:
            on_chunk(c)
    return Collected(tokens, logprobs, finish)


@dataclass(frozen=True)
class EngineInfo:
    model: str
    max_model_len: int
    max_num_seqs: int
    kv_tokens: int  # KV capacity in tokens: the controller sizes concurrency so nothing is preempted for memory
    weight_version: int
    flags: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EngineCounters:
    running: int
    waiting: int
    preemptions_total: int  # KV-pressure preemptions and E's clear_cache preemptions alike
    generated_tokens_total: int
    prefill_tokens_total: int  # FakeEngine: every prefilled token, re-prefill included; vLLM's prompt-token counter
    # likely counts each prompt once, so E's recompute cost is the controller's own count (tokens["reprefilled"])
    kv_usage: float


class Engine(Protocol):
    async def start(self) -> EngineInfo: ...

    def generate(self, req: GenRequest) -> AsyncIterator[Chunk]:
        """Delta chunks of one request; the last carries `finish`. Leaving the iterator early cancels it."""
        ...

    async def pause(self, mode: PauseMode, clear_cache: bool) -> None:
        """Returns once paused: at the next step boundary for keep and abort, after the running requests drain for
        wait. Requests submitted while paused are queued and start after resume."""
        ...

    async def resume(self) -> None: ...

    async def is_paused(self) -> bool: ...

    async def weight_version(self) -> int | None: ...

    async def load_adapter(self, name: str, path: str, inplace: bool) -> None: ...

    async def unload_adapter(self, name: str) -> None: ...

    async def counters(self) -> EngineCounters: ...

    async def close(self) -> None: ...
