"""The trainer as the controller sees it, and SimTrainer, which plays it in virtual time.

Protocol, in order: start() returns the version of the initial weights; submit(batch) hands over the batch of step
k; next_ready() yields WeightsReady(k+1) once that step is done and its weights are staged; push(v) sends the staged
weights of version v to the engine (Grant(v) -> Pushed(v) in the real worker, trainer/worker.py) and returns once the
engine holds them. The controller grants only while the engine is paused or idle (R3).

Publication (PLAN.md §4.1): `staged` copies the weights into a ping-pong buffer after each step and keeps training
while a sender pushes the newest staged version when granted, so the trainer never waits on an engine's pause.
`lockstep` blocks after WeightsReady(v) until v is pushed; it exists for the systems map's sensitivity row.
"""
import asyncio
from typing import Protocol

import numpy as np

from .. import metrics
from ..trace import Tracer
from ..types import CONTEXT_CODE, Pushed, TrainBatch, WeightsReady, snapshot_name


class TrainerClient(Protocol):
    async def start(self) -> int: ...

    async def submit(self, batch: TrainBatch) -> None: ...

    async def next_ready(self) -> WeightsReady: ...

    async def push(self, version: int) -> Pushed: ...

    async def snapshot(self, tag: str, version: int) -> None:
        """Saves the weights staged as `version`, the latest staged at the mark, whatever is staged by the time the
        copy runs (external review F02)."""

    async def close(self) -> dict: ...

    def abort(self) -> None:
        """Stops at once, without the goodbye: the controller is being torn down."""


class TrainerError(RuntimeError):
    pass


class ProcTrainerClient:
    """The trainer worker as a subprocess (worker.py) speaking wire.py over a unix socket. The worker's trace
    events arrive with its goodbye; `events` holds them after close()."""

    def __init__(self, settings: dict, sink: dict, *, workdir: str, publication: str = "staged",
                 python: str | None = None, env: dict | None = None, start_timeout: float = 900.0,
                 logdir: str | None = None):
        import sys
        self.settings, self.sink, self.publication = settings, sink, publication
        self.workdir, self.python, self.env, self.start_timeout = workdir, python or sys.executable, env, \
            start_timeout
        self.logdir = logdir or workdir
        self.events: list = []
        self.snapshots: list[tuple[str, str, float]] = []
        self.snapshot_failures: list[tuple[str, str]] = []
        self._ready: asyncio.Queue = asyncio.Queue()
        self._push: asyncio.Future | None = None
        self._bye: asyncio.Future | None = None

    async def start(self) -> int:
        import os
        from pathlib import Path

        from .wire import arecv, asend
        loop = asyncio.get_running_loop()
        Path(self.workdir).mkdir(parents=True, exist_ok=True)
        import tempfile
        # unix socket paths are capped at 107 bytes, so the socket lives in a short temporary directory
        path = os.path.join(tempfile.mkdtemp(prefix="hw-"), "trainer.sock")
        connected = loop.create_future()

        async def on_connect(reader, writer):
            if not connected.done():
                connected.set_result((reader, writer))

        self._server = await asyncio.start_unix_server(on_connect, path)
        Path(self.logdir).mkdir(parents=True, exist_ok=True)
        self._log = open(os.path.join(self.logdir, "trainer.log"), "ab")
        self.proc = await asyncio.create_subprocess_exec(
            self.python, "-m", "halfwritten.trainer.worker", "--socket", path,
            stdout=self._log, stderr=asyncio.subprocess.STDOUT, env=self.env)
        self.reader, self.writer = await asyncio.wait_for(connected, self.start_timeout)
        await asend(self.writer, "hello", {"settings": self.settings, "sink": self.sink,
                                           "publication": self.publication})
        msg = await asyncio.wait_for(arecv(self.reader), self.start_timeout)
        if msg[0] == "error":
            raise TrainerError(msg[1])
        self._bye = loop.create_future()
        self._reader_task = loop.create_task(self._read(), name="trainer-reader")
        return msg[1]

    async def _read(self):
        from .wire import arecv
        try:
            while True:
                msg = await arecv(self.reader)
                kind = msg[0]
                if kind == "weights_ready":
                    self._ready.put_nowait(msg[1])
                elif kind == "pushed":
                    if self._push is not None and not self._push.done():
                        self._push.set_result(msg[1])
                elif kind == "snapshot_done":
                    self.snapshots.append(msg[1:])
                elif kind == "snapshot_failed":
                    self.snapshot_failures.append(msg[1:])
                elif kind == "bye":
                    self.events = msg[1].pop("events", [])
                    self._bye.set_result(msg[1])
                    return
                elif kind == "error":
                    raise TrainerError(msg[1])
        except (asyncio.IncompleteReadError, ConnectionError, TrainerError) as e:
            err = e if isinstance(e, TrainerError) else TrainerError(f"trainer connection lost: {e!r}")
            self._ready.put_nowait(err)
            for fut in (self._push, self._bye):
                if fut is not None and not fut.done():
                    fut.set_exception(err)

    async def submit(self, batch: TrainBatch) -> None:
        from .wire import asend
        await asend(self.writer, "batch", batch)

    async def next_ready(self) -> WeightsReady:
        item = await self._ready.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def push(self, version: int) -> Pushed:
        from .wire import asend
        self._push = asyncio.get_running_loop().create_future()
        await asend(self.writer, "grant", version)
        return await self._push

    async def snapshot(self, tag: str, version: int, directory: str | None = None) -> None:
        from .wire import asend
        await asend(self.writer, "snapshot", tag, directory or self.workdir, version)

    async def close(self) -> dict:
        from .wire import asend
        summary = {}
        try:
            await asend(self.writer, "stop")
            summary = await asyncio.wait_for(asyncio.shield(self._bye), 120)
        except Exception as e:  # a dead worker still gets reaped below
            summary = {"error": repr(e)}
        try:
            await asyncio.wait_for(self.proc.wait(), 60)
        except TimeoutError:
            self.proc.kill()
            await self.proc.wait()
        self._server.close()
        self._log.close()
        return summary

    def abort(self) -> None:
        if getattr(self, "proc", None) is not None and self.proc.returncode is None:
            self.proc.kill()


class SimTrainer:
    """A trainer whose step takes s_fixed + s_per_token * tokens of loop time and whose policy differs from the
    engine's by a known amount: log pi - log mu = 0.002 * lag + 0.02 * [stale_kv] + noise(0.01). Full-weight pushes
    land in the FakeEngine (apply_weights); LoRA pushes return an adapter path for the controller to load. A step
    that overlaps a push (staged publication) is slowed by send_slowdown x the overlap, the contention an NCCL send
    causes on GPU1 (calibrated on a pod; 0 = free)."""

    def __init__(self, engine, tracer: Tracer, *, s_fixed: float = 0.2, s_per_token: float = 3e-5,
                 push_s: float = 0.05, lora: bool = False, publication: str = "staged", eps_max: float = 5.0,
                 weight_bytes: int = 1_200_000_000, adapter_bytes: int = 4_000_000, send_slowdown: float = 0.0,
                 seed: int = 0):
        self.engine, self.tracer = engine, tracer
        self.s_fixed, self.s_per_token, self.push_s = s_fixed, s_per_token, push_s
        self.send_slowdown = send_slowdown
        self._sends: list[list[float | None]] = []  # [start, end] of pushes, end None while in flight
        self.lora, self.publication, self.eps_max, self.seed = lora, publication, eps_max, seed
        self.bytes = adapter_bytes if lora else weight_bytes
        self.staged = self.pushed = -1
        self.snapshots: list[tuple[str, str, float]] = []  # (tag, path, seconds): nothing is written in simulation
        self.snapshot_failures: list[tuple[str, str]] = []
        self._batches: asyncio.Queue[TrainBatch] = asyncio.Queue()
        self._ready: asyncio.Queue[WeightsReady] = asyncio.Queue()
        self._pushed_event = asyncio.Event()
        self._task = None
        self.steps = 0

    async def start(self) -> int:
        self.staged = 0
        self.tracer.state("trainer", "idle")
        self.tracer.state("sender", "idle")
        self._task = asyncio.get_running_loop().create_task(self._work(), name="sim-trainer")
        return 0

    async def submit(self, batch: TrainBatch) -> None:
        self._batches.put_nowait(batch)

    async def next_ready(self) -> WeightsReady:
        return await self._ready.get()

    async def push(self, version: int) -> Pushed:
        if not self.pushed < version <= self.staged:
            raise RuntimeError(f"push v{version}: staged v{self.staged}, pushed v{self.pushed}")
        self.tracer.state("sender", "send", version=version)
        interval = [asyncio.get_running_loop().time(), None]
        self._sends.append(interval)
        await asyncio.sleep(self.push_s)
        interval[1] = asyncio.get_running_loop().time()
        if not self.lora:
            self.engine.apply_weights(version)
        self.pushed = version
        self._pushed_event.set()
        self.tracer.state("sender", "idle")
        return Pushed(version, self.bytes, self.push_s, f"sim://adapter/v{version}" if self.lora else None)

    async def snapshot(self, tag: str, version: int) -> None:
        if version > self.staged:
            raise TrainerError(f"snapshot of v{version}, staged only v{self.staged}")
        self.tracer.instant("trainer", "snapshot", tag=tag, version=version)
        self.snapshots.append((tag, snapshot_name(tag, version), 0.0))

    async def close(self) -> dict:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        return {"steps": self.steps, "staged": self.staged, "pushed": self.pushed}

    def abort(self) -> None:
        if self._task:
            self._task.cancel()

    async def _work(self):
        while True:
            batch = await self._batches.get()
            if self.publication == "lockstep":
                while self.pushed < self.staged:
                    self.tracer.state("trainer", "wait_push")
                    self._pushed_event.clear()
                    await self._pushed_event.wait()
            tokens = sum(len(s.token_ids) for s in batch.samples)
            self.tracer.state("trainer", "fwd_bwd", step=batch.step)
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            await asyncio.sleep(self.s_fixed + self.s_per_token * tokens)
            if self.send_slowdown and (overlap := self._overlap(t0, loop.time())):
                await asyncio.sleep(self.send_slowdown * overlap)
            stats = self._stats(batch)
            self.staged = batch.step + 1
            self.steps += 1
            self.tracer.state("trainer", "idle")
            self._ready.put_nowait(WeightsReady(self.staged, stats))

    def _overlap(self, t0: float, t1: float) -> float:
        """Seconds of [t0, t1] during which a push was in flight; forgets pushes that ended before t0."""
        self._sends = [iv for iv in self._sends if iv[1] is None or iv[1] > t0]
        return sum(max(0.0, min(t1 if e is None else e, t1) - max(s, t0)) for s, e in self._sends)

    def _stats(self, batch: TrainBatch) -> dict:
        if not batch.samples:
            return {}
        mu = np.concatenate([np.asarray(s.logprobs, dtype=float) for s in batch.samples])
        lag = np.concatenate([np.asarray(s.lag, dtype=float) for s in batch.samples])
        codes = np.concatenate([np.asarray(s.context_codes, dtype=np.int64) for s in batch.samples])
        noise = np.random.default_rng((self.seed, batch.step)).normal(0, 0.01, mu.size)
        logp = mu + 0.002 * lag + 0.02 * (codes == CONTEXT_CODE["stale_kv"]) + noise
        return metrics.token_stats(logp, mu, codes, self.eps_max)
