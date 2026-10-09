"""A deterministic engine with vLLM v0.30's pause semantics and ground truth, for tests and the regime calculator.

It schedules like a continuous-batching engine: each step admits waiting requests up to max_num_seqs, computes the
KV each running request still lacks under the version its weights (or adapter) have now, samples one token per
request, and takes alpha + beta*width + gamma*context + delta*prefill seconds of loop time. Decodes go first; a
prefill must fit whole in what is left of the step's token budget (max_num_batched_tokens, chunked prefill off), so
E's re-prefill after a swap spreads over several steps as it does in vLLM. Per request it keeps the oldest version
among the KV computed so far, so it knows whether each token's logits read KV of another version (`truth`), which
the controller's ledger must reproduce without seeing it.

Pause modes, as in vllm/v1/engine/core.py and scheduler.py:
- wait: stop admitting, keep stepping until nothing is running, then pause. Waiting requests (never started, or
  preempted for KV) stay queued and start after resume on the new weights.
- abort: at the next step boundary, finish every running and waiting request with finish "abort" and an empty
  chunk; tokens computed but not yet delivered are dropped, as vLLM drops in-flight async outputs.
- keep: freeze at the next step boundary. clear_cache=True drops every running request's KV and counts a
  preemption, so on resume it re-prefills prompt + answer on the new weights (E); clear_cache=False keeps the KV (D).
Requests submitted while paused are queued. Weights change only while paused or idle (apply_weights); LoRA adapters
load any time, in place or under a new name.

Behaviours of a real server the tests need to reproduce, each off by default:
- `delivery`: `immediate` at the end of the step; `one_step_late` one step time later (async scheduling: outputs are
  processed while the next step runs), so they can arrive after a pause is acknowledged; `held_until_resume` at the
  start of the next step, so the last outputs before a pause arrive only after resume (breaks E1 on purpose).
- `submit_latency`: a request reaches the engine that long after the client sends it (HTTP on its own connection),
  so it can arrive after a pause and be queued rather than aborted.
- `stats_lag`: counters are published with each step's outputs, as vLLM's scheduler stats are, so a preemption made
  while paused (E's cache reset) shows up only after the next step.
"""
import asyncio
import math
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from ..hashing import mix as _mix
from ..hashing import stable_key as _key
from ..lengths import LengthModel
from ..types import Chunk, GenRequest, PauseMode, parse_rid
from .base import EngineCounters, EngineError, EngineInfo

Delivery = Literal["immediate", "one_step_late", "held_until_resume", "bursty"]
BURST_S, BACKLOG_S = 0.08, 0.12  # "bursty": a backlogged server sends old outputs late, in bursts
__all__ = ["CostModel", "FakeEngine", "LengthModel"]


@dataclass(frozen=True)
class CostModel:
    """Step time in seconds: alpha + beta*width + gamma*context_tokens + delta*prefill_tokens. The defaults are of
    the order of a sub-1B model on one A100; bench.calibrate fits them to an engine's measured steps."""
    alpha: float = 2e-3
    beta: float = 2e-5
    gamma: float = 1e-8
    delta: float = 5e-6

    def step_time(self, width: int, context_tokens: int, prefill_tokens: int) -> float:
        return self.alpha + self.beta * width + self.gamma * context_tokens + self.delta * prefill_tokens


@dataclass
class _Req:
    req: GenRequest
    sid: str
    attempt: int
    n_prior: int  # answer tokens already inside prompt_ids (C1 continuations)
    cap: int  # n_prior + max_tokens
    target: int
    base: int  # per-attempt hash key
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    generated: int = 0
    computed: int = 0  # context positions whose KV exists
    min_kv: int | None = None  # oldest version among them
    done: bool = False
    pending: list = field(default_factory=list)  # (handle, chunk, truth) scheduled but not delivered
    deliver_at: float = 0.0  # when its last scheduled chunk arrives


class FakeEngine:
    def __init__(self, *, lengths: LengthModel = LengthModel(), cost: CostModel = CostModel(),
                 max_num_seqs: int = 256, kv_tokens: int = 1 << 20, max_model_len: int = 4096,
                 max_num_batched_tokens: int = 1 << 30, delivery: Delivery = "immediate",
                 submit_latency: float = 0.0, stats_lag: bool = False, vocab: int = 151_936, version: int = 0,
                 async_scheduling: bool = False, silent_waiting_aborts: bool = False):
        if max_num_batched_tokens < max_model_len:
            raise EngineError("with chunked prefill off, max_num_batched_tokens must be >= max_model_len")
        self.lengths, self.cost, self.delivery, self.vocab = lengths, cost, delivery, vocab
        self.max_num_seqs, self.kv_tokens, self.max_model_len = max_num_seqs, kv_tokens, max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.submit_latency, self.stats_lag = submit_latency, stats_lag
        self.silent_waiting_aborts = silent_waiting_aborts
        # vLLM's async scheduling plans step N+1 before step N's outputs are processed: when a pause lands, requests
        # admitted into that plan are running (their KV allocated) with no token yet, and a cache reset preempts them
        self.async_scheduling = async_scheduling
        self.version = version
        self.adapters: dict[str, int] = {}
        self._running: list[_Req] = []
        self._waiting: deque[_Req] = deque()
        self._state = "running"  # running | draining | paused
        self._pending_pause: tuple[PauseMode, bool] | None = None
        self._paused = None
        self._wake = None
        self._task = None
        self._held: list = []
        self._last_dt = cost.alpha
        self._prompt_len: dict[tuple[str, int], int] = {}
        self._truth: dict[tuple[str, int], list[tuple[int, bool]]] = {}
        self.preemptions_total = self.generated_tokens_total = self.prefill_tokens_total = 0
        self._published_preemptions = 0
        self.steps = 0

    # -- Engine protocol

    async def start(self) -> EngineInfo:
        self._paused, self._wake = asyncio.Event(), asyncio.Event()
        self._task = asyncio.get_running_loop().create_task(self._loop(), name="fake-engine")
        return EngineInfo("fake", self.max_model_len, self.max_num_seqs, self.kv_tokens, self.version,
                          {"delivery": self.delivery})

    async def generate(self, req: GenRequest):
        if self.submit_latency:
            await asyncio.sleep(self.submit_latency)
        sid, attempt, cont = parse_rid(req.rid)
        if req.adapter is not None and req.adapter not in self.adapters:
            raise EngineError(f"adapter {req.adapter!r} not loaded")
        if cont == 0:
            self._prompt_len[(sid, attempt)] = len(req.prompt_ids)
            self._truth[(sid, attempt)] = []
        n_prior = len(req.prompt_ids) - self._prompt_len[(sid, attempt)]
        cap = n_prior + req.max_tokens
        r = _Req(req, sid, attempt, n_prior, cap, cap if req.ignore_eos else self.lengths(sid, attempt),
                 _key("tok", sid, attempt))
        self._waiting.append(r)
        self._wake.set()
        try:
            while True:
                chunk = await r.queue.get()
                if chunk is None:  # the stream ends with no final chunk
                    return
                yield chunk
                if chunk.finish:
                    return
        finally:
            if not r.done:  # the client left early: cancel the request
                self._drop(r)

    async def pause(self, mode: PauseMode, clear_cache: bool) -> None:
        if mode not in ("wait", "abort", "keep"):
            raise EngineError(f"unknown pause mode {mode!r}")
        if self._state == "paused":
            return
        if mode == "wait":
            self._state = "draining"
        else:
            self._pending_pause = (mode, clear_cache)
        self._wake.set()
        await self._paused.wait()

    async def resume(self) -> None:
        self._state = "running"
        self._paused.clear()
        self._wake.set()

    async def is_paused(self) -> bool:
        return self._state == "paused"

    async def weight_version(self) -> int | None:
        return self.version

    async def load_adapter(self, name: str, path: str, inplace: bool) -> None:
        m = re.search(r"v(\d+)$", path)
        if not m:
            raise EngineError(f"adapter path {path!r} carries no version")
        if name in self.adapters and not inplace:
            raise EngineError(f"adapter {name!r} already loaded")
        self.adapters[name] = int(m.group(1))

    async def unload_adapter(self, name: str) -> None:
        if any(r.req.adapter == name for r in [*self._running, *self._waiting]):
            raise EngineError(f"adapter {name!r} in use")
        self.adapters.pop(name, None)

    async def counters(self) -> EngineCounters:
        used = sum(len(r.req.prompt_ids) + r.generated for r in self._running)
        preemptions = self._published_preemptions if self.stats_lag else self.preemptions_total
        return EngineCounters(len(self._running), len(self._waiting), preemptions,
                              self.generated_tokens_total, self.prefill_tokens_total, used / self.kv_tokens)

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # -- test and simulator hooks

    def apply_weights(self, version: int) -> None:
        """The trainer's full-weight push. Like vLLM's weight transfer it needs an engine that is paused, or idle
        (arm A syncs between batches without pausing)."""
        if self._state != "paused" and (self._running or self._waiting):
            raise EngineError("weights changed while generating")
        self.version = version

    def truth(self, sid: str, attempt: int = 0) -> list[tuple[int, bool]]:
        """Per delivered answer token of an attempt (continuations included): (version that sampled it, whether its
        logits read KV computed by another version)."""
        return list(self._truth[(sid, attempt)])

    # -- the step loop

    async def _loop(self):
        while True:
            if self._pending_pause:
                self._enter_pause(*self._pending_pause)
            elif self._state == "draining" and not self._running:
                self._set_paused()
            if self._state != "paused" and self._held:  # held outputs wait out a pause
                self._flush_held()
            if self._state != "paused" and self._has_work():
                await self._step()
                continue
            self._wake.clear()
            await self._wake.wait()

    def _has_work(self) -> bool:
        return bool(self._running) or (self._state == "running" and bool(self._waiting))

    def _enter_pause(self, mode: PauseMode, clear_cache: bool):
        self._pending_pause = None
        if mode == "abort":
            for r in self._running:
                self._abort(r)
            for r in self._waiting:  # vLLM ends a request it never scheduled without a final output (two H100s)
                self._abort(r, silent=self.silent_waiting_aborts)
            self._running.clear()
            self._waiting.clear()
        elif clear_cache:  # vLLM preempts every running request back to the waiting queue, KV dropped
            if self.async_scheduling:
                self._admit(compute=False)
            for r in self._running:
                r.computed, r.min_kv = 0, None
                self.preemptions_total += 1
            self._waiting.extendleft(reversed(self._running))
            self._running.clear()
        self._set_paused()

    def _set_paused(self):
        self._state = "paused"
        self._paused.set()

    def _abort(self, r: _Req, silent: bool = False):
        # outputs not yet delivered are lost, as in vLLM, and uncounted: vLLM counts generated tokens as its API
        # server processes outputs, and an aborted request's unprocessed ones never are
        lost = sum(len(chunk.token_ids) for _, chunk, _ in r.pending)
        lost += sum(len(h[1].token_ids) for h in self._held if h[0] is r)
        self.generated_tokens_total -= lost
        for handle, chunk, truth in r.pending:
            handle.cancel()
        r.pending.clear()
        self._held = [h for h in self._held if h[0] is not r]
        r.done = True
        r.queue.put_nowait(None if silent else Chunk((), (), "abort"))

    def _drop(self, r: _Req):
        r.done = True
        for handle, _, _ in r.pending:
            handle.cancel()
        if r in self._running:
            self._running.remove(r)
        elif r in self._waiting:
            self._waiting.remove(r)

    def _version_of(self, r: _Req) -> int:
        return self.adapters[r.req.adapter] if r.req.adapter is not None else self.version

    def _kv_need(self) -> int:
        return sum(len(r.req.prompt_ids) + r.generated + 1 for r in self._running)

    def _relieve_kv_pressure(self):
        while len(self._running) > 1 and self._kv_need() > self.kv_tokens:
            r = self._running.pop()
            r.computed, r.min_kv = 0, None
            self._waiting.appendleft(r)
            self.preemptions_total += 1

    def _admit(self, compute: bool = True) -> list:
        """Moves the waiting requests whose whole prefill fits what is left of the step's token budget (and the KV
        pool) to running, as vLLM's scheduler does with chunked prefill off; returns them. With compute False (a
        plan made before a pause landed) they hold no KV and no token yet."""
        admitted, budget = [], self.max_num_batched_tokens - len(self._running)
        while (self._state == "running" or not compute) and self._waiting and len(self._running) < self.max_num_seqs:
            r = self._waiting[0]
            need = len(r.req.prompt_ids) + r.generated
            if (need > budget and (self._running or admitted)) or self._kv_need() + need + 1 > self.kv_tokens:
                break
            self._running.append(self._waiting.popleft())
            admitted.append(r)
            budget -= need
        return admitted

    async def _step(self):
        """Decodes every running request, then admits waiting ones whose whole prefill fits what is left of the
        step's token budget (and the KV pool), as vLLM's scheduler does with chunked prefill off. Running means
        prefilled: a request waiting for budget has no KV and no token yet."""
        self._relieve_kv_pressure()
        work = list(self._running) + self._admit()
        out, context, prefill = [], 0, 0
        for r in work:
            need = len(r.req.prompt_ids) + r.generated  # every context position, the input token's included
            new = need - r.computed
            v = self._version_of(r)
            prefill += new - 1
            r.min_kv = v if r.computed == 0 else min(r.min_kv, v)
            r.computed = need
            mixed = r.min_kv < v
            pos = r.n_prior + r.generated
            h = _mix(r.base ^ _mix(pos * 1_000_003 + v * 7_919 + mixed))
            r.generated += 1
            finish = None
            if r.n_prior + r.generated >= min(r.target, r.cap):
                finish = "length" if r.req.ignore_eos or r.target > r.cap else "stop"
            out.append((r, Chunk((h % self.vocab,), (-0.05 - (h >> 32) % 2000 / 1000,), finish), (v, mixed)))
            context += need
        dt = self.cost.step_time(len(out), context, prefill)
        self.prefill_tokens_total += prefill
        self.generated_tokens_total += len(out)
        self.steps += 1
        await asyncio.sleep(dt)
        self._last_dt = dt
        self._published_preemptions = self.preemptions_total
        for r, chunk, truth in out:
            if r.done:  # the client cancelled it during the step
                continue
            if chunk.finish:
                r.done = True
                self._running.remove(r)
            self._deliver(r, chunk, truth)

    def _deliver(self, r: _Req, chunk: Chunk, truth):
        if self.delivery == "immediate":
            self._put(r, chunk, truth)
        elif self.delivery == "bursty":  # every output BACKLOG_S late, released at the next BURST_S boundary
            loop = asyncio.get_running_loop()
            r.deliver_at = max(math.ceil((loop.time() + BACKLOG_S) / BURST_S) * BURST_S, r.deliver_at)
            r.pending.append((loop.call_at(r.deliver_at, self._put_pending, r), chunk, truth))
        elif self.delivery == "one_step_late":
            loop = asyncio.get_running_loop()
            # a request's chunks arrive in order even when a long (re-prefill) step precedes a short one, and timers
            # due at the same instant fire in no set order: each timer delivers the oldest pending chunk
            r.deliver_at = max(loop.time() + self._last_dt, r.deliver_at)
            r.pending.append((loop.call_at(r.deliver_at, self._put_pending, r), chunk, truth))
        else:
            self._held.append((r, chunk, truth))

    def _put_pending(self, r: _Req):
        _, chunk, truth = r.pending.pop(0)
        self._put(r, chunk, truth)

    def _flush_held(self):
        held, self._held = self._held, []
        for r, chunk, truth in held:
            self._put(r, chunk, truth)

    def _put(self, r: _Req, chunk: Chunk, truth):
        self._truth[(r.sid, r.attempt)].append(truth)
        r.queue.put_nowait(chunk)
