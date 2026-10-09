"""The async RL loop's controller: admission, streaming, batching and weight syncs, identical for every regime.

It owns the engine's pauses and resumes and so is the only place that can say which weights sampled each token
(ledger.py). The rules (PLAN.md §4.2, §5.2):

- Dispatch unit: a group of G independent requests for one prompt. At most M groups in flight, and at most
  (v_engine + K + 1) * B groups ever dispatched (AReaL's bound, counted in groups). Nothing is dispatched during a
  sync. A request already sent can still reach the engine after a pause (HTTP requests travel on their own
  connections); vLLM queues it and starts it after resume, so a zero-token answer at a boundary is labelled with the
  new version and, under abort, is not waited for.
- Batch k is the B completed groups with the oldest group version g (the oldest version any member started on). It is
  formed only if every group left behind has g >= k + 1 - K: the trainer waits for a straggler rather than dropping
  it, which bounds every token's lag by K without filtering out long answers.
- Weights: the trainer stages version v after each step and keeps training; the controller pushes the newest staged
  version once it is S ahead of the engine. A sync is: pause (per regime) -> fence -> grant -> push -> verify ->
  snapshot -> cut the ledger -> resume -> resubmit (C1, C2).
- The boundary of an in-flight answer is the number of its tokens delivered before resume is sent. The fence
  (fence.py, shared with the session gate) is the engine's own count: a keep pause is cut only once the tokens
  delivered equal the generated-token count read after the acknowledgement and both have held still (E2); after a
  drain or an abort the same count says nothing old is on its way. Tokens that arrive between the acknowledgement
  and the cut are counted as late_old_tokens and their latest arrival as the ack lag: they are old and labelled so,
  and they show how far behind delivery ran. A count that exceeds what was delivered at the cut is an I9 violation.
  Whether vLLM can still deliver an old-weights token after resume (E1) only the per-session canary can tell.
- Rewards are scored off the event loop when a pool is given (math-verify can take seconds), so scoring never stalls
  chunk delivery, the fence or dispatch.
- Forced lengths (cfg.forced_lengths: the systems map) send ignore_eos with max_tokens set to the length the
  attempt drew; a C1 continuation asks for the rest of it, a C2 restart draws anew.

Time is read only through the event loop, so the same code runs in virtual time against FakeEngine (clock.py).
"""
import asyncio
import json
from concurrent.futures import Executor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import metrics
from .advantages import advantages, zero_variance
from .config import RunConfig
from .engines.base import Engine, EngineError
from .fence import LAGGING_SETTLE_S, SETTLE_S, exact_cut
from .ledger import Ledger, LedgerError
from .lengths import LengthModel, parse
from .regimes import check_ledger, resolve
from .tasks import score as score_answer
from .trace import Tracer, check_lanes, lane_fractions
from .trainer.client import TrainerClient, TrainerError
from .types import BUDGET_MARK, CONTEXTS, GenRequest, Pushed, TrainBatch, TrainSample, rid


class ControllerError(RuntimeError):
    pass


# what ends a run as invalid, with its row written: a timeout on any call (R6), the controller's own checks, and the
# engine's, trainer's or ledger's errors (a dead engine used to raise out of run() and leave no row)
RUN_ERRORS = (TimeoutError, ControllerError, EngineError, TrainerError, LedgerError)
SHUTDOWN_CALL_S = 10.0  # after an error, each shutdown call to the engine or trainer gets this, then is given up


@dataclass(eq=False)
class _Group:
    gid: str
    prompt_ids: tuple[int, ...]
    answer: object
    t_dispatch: float
    samples: list = field(default_factory=list)
    pending: int = 0
    order: int = -1  # completion order


@dataclass(eq=False)
class _Sample:
    sid: str
    group: _Group
    attempt: int
    adapter: str | None
    cont: int = 0
    got: int = 0  # tokens the current request delivered (C1's answer also holds the earlier requests')
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    task: asyncio.Task | None = None
    aborted: bool = False
    reward: float | None = None


@dataclass(eq=False)
class _Sync:
    """One sync as it runs: what each phase decides, for the phases after it and the record (_sync_record)."""
    v: int
    t0: float
    t_paused: float = 0.0
    lengths_at_fence: dict = field(default_factory=dict)  # every in-flight answer's tokens at the acknowledgement
    fence_s: float = 0.0  # the pause's cost after the ack: the aborted streams' ends, then the settle
    pushed: Pushed | None = None
    first: int | None = None  # the engine's generated-token count read at the push (None without a pause)
    generated: int | None = None  # ... as it stood, settled, at the cut
    settled: bool = True
    ack_lag: float | None = None
    t_cut: float = 0.0
    delivered_at_cut: int = 0
    late: int = 0  # old tokens that arrived between the acknowledgement and the cut
    reprefill: int = 0
    queued: int = 0
    resubmit: list = field(default_factory=list)
    skipped: int = 0


@dataclass
class RunSummary:
    regime: str
    valid: bool
    error: str | None
    violations: list[str]
    wall_s: float
    steps: int
    groups_submitted: int
    samples: dict
    tokens: dict
    late_old_tokens: int
    versions_skipped: int
    inflight_mean: float  # time-weighted answers in flight: the engine's occupancy, drains included
    events: dict  # counted occurrences: queued at an abort, late zero-token aborts, reward timeouts
    lanes: dict
    syncs: list[dict]
    steps_log: list[dict]
    engine: dict
    trainer: dict
    host: dict | None = None  # the controller process itself (real runs): event-loop lag, main-thread CPU

    def metrics(self) -> dict:
        """The scalar summary written to metrics.json; per-sync and per-step records go to their own files."""
        d = {k: v for k, v in asdict(self).items() if k not in ("syncs", "steps_log")}
        d["syncs"] = len(self.syncs)
        d["sync_s"] = metrics.quantiles([s["sync_s"] for s in self.syncs])
        d["pause_s"] = metrics.quantiles([s["pause_s"] for s in self.syncs])
        d["fence_s"] = metrics.quantiles([s.get("fence_s", 0.0) for s in self.syncs])
        d["ack_lag_s"] = metrics.quantiles([s["ack_lag_s"] for s in self.syncs if s.get("ack_lag_s") is not None])
        und = [s["undelivered"] for s in self.syncs if s.get("undelivered") is not None]
        d["undelivered"] = {"syncs": len(und), "positive": sum(x > 0 for x in und), "min": min(und, default=None),
                            "max": max(und, default=None)}
        d["steps_per_hour"] = 3600 * self.steps / self.wall_s if self.wall_s else 0.0
        d["tokens_per_s"] = self.tokens["delivered"] / self.wall_s if self.wall_s else 0.0
        # useful tokens only: C2's discarded attempts and answers left unconsumed are not throughput
        d["goodput_tokens_per_s"] = self.tokens["consumed"] / self.wall_s if self.wall_s else 0.0
        lag = [x for s in self.steps_log for x in [s.get("lag_mean")] if x is not None]
        d["lag_mean"] = float(np.mean(lag)) if lag else None
        d["lag_max"] = max((s["lag_max"] for s in self.steps_log), default=None)
        d["depth_max"] = max((s["depth_max"] for s in self.steps_log), default=None)
        return d


class Controller:
    def __init__(self, cfg: RunConfig, engine: Engine, trainer: TrainerClient, prompts, reward, *,
                 ledger: Ledger | None = None,
                 tracer: Tracer | None = None, lengths: LengthModel | None = None,
                 reward_pool: Executor | None = None, stop: list[str] | None = None, journal: Path | None = None):
        """journal: a JSON-lines file each step's record is appended to as it happens, so a run that is killed (or
        dies) before its row is written keeps its steps (S1 lost its row that way)."""
        self.cfg = cfg.validate()
        self.journal = open(journal, "a", buffering=1) if journal else None
        self.regime = resolve(cfg.regime, cfg.lora)
        self.engine, self.trainer, self.prompts, self.reward = engine, trainer, iter(prompts), reward
        self.ledger = ledger or Ledger()
        self.tracer = tracer
        self.lengths = lengths or (parse(cfg.forced_lengths, cfg.seed) if cfg.forced_lengths else None)
        self.reward_pool = reward_pool
        self.stop = tuple(stop or ())
        self.snapshots: list[dict] = []
        self.v_engine = self.staged = self.next_step = self.steps_done = 0
        self.trainer_busy = self.syncing = self.stopping = self._resuming = self._abort_phase = False
        self.inflight_groups: dict[str, _Group] = {}
        self.completed_groups: list[_Group] = []
        self.live: dict[str, _Sample] = {}
        self.dispatched = self.n_completed = 0
        self.adapter_refs: dict[str, int] = {}
        self.violations: list[str] = []
        self.syncs: list[dict] = []
        self.steps_log: list[dict] = []
        self.tokens = {"delivered": 0, "consumed": 0, "discarded": 0, "reprefilled": 0}
        self.events = {"queued_at_abort": 0, "late_zero_token_aborts": 0, "finishless_aborts": 0, "reward_timeouts": 0,
                       "chunks_while_resuming": 0, "reset_preemptions_without_tokens": 0,
                       "count_lagged_syncs": 0}  # exact fence: the engine's count moved after its read at the ack
        self.late_old_tokens = self.versions_skipped = 0
        self._t_ack: float | None = None  # a keep pause's acknowledgement, until the cut
        self._late_s = 0.0  # how long after it this pause's latest old token arrived (the ack lag)
        # E's reset preempts every request vLLM is running: those that delivered tokens surely, and possibly ones it
        # scheduled whose first token has not reached us. Expected counts the first, allowed counts every live request
        self._preempt_expected = self._preempt_allowed = self._kv_preemptions = 0
        self._reset_unaccounted = False  # an E pause reset the cache but the sync stopped before counting it
        self._watch = None
        self._streams: set[asyncio.Task] = set()
        self._stop_reason: asyncio.Future | None = None
        self.signalled = False
        self._live_area = self._t_live = 0.0
        self._n_live = 0

    # -- running

    def now(self) -> float:
        return self._loop.time()

    async def run(self) -> RunSummary:
        """Runs until the wall-clock budget or the step limit, then cancels what is in flight and summarises. A
        timeout on any engine or trainer call (R6), or an engine, trainer or ledger error, ends the run as invalid
        with its row written (RUN_ERRORS); any other error propagates."""
        self._loop = asyncio.get_running_loop()
        self.tracer = self.tracer or Tracer(self.now, "controller")
        self._kick = {k: asyncio.Event() for k in ("admit", "batch", "sync")}
        self._stop_reason = self._loop.create_future()
        info = await self._call(self.engine.start())
        if self.cfg.M * self.cfg.G > info.max_num_seqs:
            raise ControllerError(f"M*G = {self.cfg.M * self.cfg.G} exceeds max_num_seqs {info.max_num_seqs}: "
                                  "the engine would queue requests, and wait would not drain them")
        await self._initial_push(await self._call(self.trainer.start()))
        start = await self._call(self.engine.counters())
        self.engine_counters = start  # the row's final counters, unless the shutdown can read them again
        self._preempt_expected = self._preempt_allowed = start.preemptions_total
        self._generated0 = start.generated_tokens_total  # an engine shared with earlier work starts above zero
        self.t_start = self._t_live = self.now()
        self.tracer.state("engine", "idle")
        self.tracer.state("batcher", "wait_data")
        self.tracer.state("sync", "between")
        coros = [self._admit_loop(), self._batch_loop(), self._ready_loop(), self._sync_loop()]
        if self.cfg.snapshot_every_s:
            coros.append(self._snapshot_loop())
        loops = [self._loop.create_task(c, name=c.__name__) for c in coros]
        for t in loops:
            t.add_done_callback(self._task_done)
        if self.cfg.budget_s is not None:
            self._loop.call_at(self.t_start + self.cfg.budget_s, self._stop, "budget")
        error = None
        try:
            reason = await self._stop_reason
            if reason == "budget" and self.cfg.snapshot_every_s:
                await self._snapshot(BUDGET_MARK)  # the primary endpoint: the weights at exactly T, in every arm
            elif reason == "signal":
                error = ControllerError(f"stopped by a signal after {self.steps_done} steps")
        except RUN_ERRORS as e:
            error = e
        except BaseException:  # cancelled or exiting: no graceful goodbye, which could outlast a kill timeout
            for t in [*loops, *self._streams]:
                t.cancel()
            self.trainer.abort()
            raise
        await self._shutdown(loops, failed=error is not None)
        if self.journal:
            self.journal.close()
        return self._summary(error)

    async def _call(self, aw):
        return await asyncio.wait_for(aw, self.cfg.timeout_s)

    def _task_done(self, task: asyncio.Task):
        if not task.cancelled() and task.exception() is not None and not self._stop_reason.done():
            self._stop_reason.set_exception(task.exception())

    def _stop(self, reason):
        if not self._stop_reason.done():
            self._stop_reason.set_result(reason)

    def request_stop(self) -> bool:
        """Ends the run soon, as a budget would, but invalid (a signal: a slot's time box, an operator); its row is
        still written. False if it was already ending."""
        if self._stop_reason is None or self._stop_reason.done():
            return False
        self.signalled = True
        self._stop("signal")
        return True

    def _fail(self, exc: BaseException):
        if not self._stop_reason.done():
            self._stop_reason.set_exception(exc)

    async def _initial_push(self, v0: int):
        """Pushes the trainer's initial weights, so the engine holds bit-for-bit what the trainer holds."""
        pushed = await self._call(self.trainer.push(v0))
        if self.regime.weights == "full":
            if (v := await self._call(self.engine.weight_version())) != v0:
                raise ControllerError(f"I6: engine holds v{v} after pushing v{v0}")
        elif self.regime.weights == "lora_inplace":
            await self._call(self.engine.load_adapter("policy", pushed.adapter_path, inplace=False))
        else:
            await self._call(self.engine.load_adapter(f"policy-v{v0}", pushed.adapter_path, inplace=False))
            self.adapter_refs[f"policy-v{v0}"] = 0
        self.v_engine = self.staged = self.next_step = self._v0 = v0

    def _kick_all(self):
        for ev in self._kick.values():
            ev.set()

    # -- admission and streaming

    async def _admit_loop(self):
        while True:
            self._kick["admit"].clear()
            while self._can_dispatch():
                self._dispatch()
            await self._kick["admit"].wait()

    def _can_dispatch(self) -> bool:
        c = self.cfg
        return (not self.stopping and not self.syncing and len(self.inflight_groups) < c.M
                and self.dispatched < (self.v_engine + c.K + 1) * c.B)

    def _dispatch(self):
        gid, prompt_ids, answer = next(self.prompts)
        t = self.now()
        g = _Group(gid, prompt_ids, answer, t)
        adapter = {"full": None, "lora_inplace": "policy",
                   "lora_versioned": f"policy-v{self.v_engine}"}[self.regime.weights]
        for j in range(self.cfg.G):
            sid = f"{gid}.{j}"
            s = _Sample(sid, g, self.ledger.open(sid, self.v_engine, group=gid, t=t, prompt_len=len(prompt_ids)),
                        adapter)
            g.samples.append(s)
            self.live[sid] = s
            self._start(s)
        if self.regime.weights == "lora_versioned":
            self.adapter_refs[adapter] += self.cfg.G
        g.pending = self.cfg.G
        self.inflight_groups[gid] = g
        self.dispatched += 1
        self._engine_lane()

    def _start(self, s: _Sample):
        """Sends the request for the rest of `s`'s answer: its prompt plus any partial tokens (C1), and the tokens it
        may still generate (the attempt's forced length, or max_tokens)."""
        total = min(self.lengths(s.sid, s.attempt), self.cfg.max_tokens) if self.lengths else self.cfg.max_tokens
        if len(s.group.prompt_ids) + total > self.cfg.max_model_len:  # vLLM would refuse it (tasks.too_long)
            raise ControllerError(f"{s.sid}: a {len(s.group.prompt_ids)}-token prompt leaves no room for {total} "
                                  f"tokens within max_model_len {self.cfg.max_model_len}")
        req = GenRequest(rid(s.sid, s.attempt, s.cont), s.group.prompt_ids + tuple(s.tokens), total - len(s.tokens),
                         self.cfg.temperature, s.adapter, ignore_eos=self.lengths is not None, stop=self.stop)
        s.got = 0
        s.task = self._loop.create_task(self._stream(s, req), name=req.rid)
        self._streams.add(s.task)
        s.task.add_done_callback(self._streams.discard)
        s.task.add_done_callback(self._task_done)

    async def _stream(self, s: _Sample, req: GenRequest):
        finish = None
        async for chunk in self.engine.generate(req):  # the stream ends after the chunk carrying finish
            if finish is not None:
                raise ControllerError(f"{req.rid}: chunk after finish {finish!r}")
            if chunk.token_ids:
                s.got += len(chunk.token_ids)
                if self._resuming:  # new-weights tokens racing the /resume response: expected, counted
                    self.events["chunks_while_resuming"] += 1
                s.tokens.extend(chunk.token_ids)
                s.logprobs.extend(chunk.logprobs)
                self.ledger.extend(s.sid, len(chunk.token_ids))
                self.tokens["delivered"] += len(chunk.token_ids)
            if self._t_ack is not None and chunk.token_ids:
                self._late_s = max(self._late_s, self.now() - self._t_ack)
            finish = chunk.finish
        if finish is None:
            # vLLM 0.30 ends a request it aborted before ever scheduling it with no final chunk (C1 on two H100s): with
            # no tokens from this request it is that abort; with tokens, an answer's end went missing
            if s.got or self.regime.pause != "abort":
                raise ControllerError(f"{req.rid}: stream ended without a finish reason")
            finish = "abort"
            self.events["finishless_aborts"] += 1
        if finish == "abort":
            if self._abort_phase:
                s.aborted = True
            elif not s.got and self.regime.pause == "abort":
                # aborted before its first token, but noticed only after its sync labelled it as queued: start over
                self.events["late_zero_token_aborts"] += 1
                s.cont += 1
                self._start(s)
            else:
                raise ControllerError(f"{req.rid}: aborted outside an abort-mode sync")
            return
        await self._finish(s, finish)

    async def _finish(self, s: _Sample, finish: str):
        self.ledger.close(s.sid, finish, self.now())
        del self.live[s.sid]
        if self.regime.weights == "lora_versioned":
            self.adapter_refs[s.adapter] -= 1
        self._engine_lane()
        self._kick["admit"].set()
        s.reward = await self._score(s, finish)
        self.ledger.annotate(s.sid, reward=s.reward)
        g = s.group
        g.pending -= 1
        if g.pending == 0:
            del self.inflight_groups[g.gid]
            g.order = self.n_completed
            self.n_completed += 1
            self.completed_groups.append(g)
            self._kick["batch"].set()
        self._kick["admit"].set()

    async def _score(self, s: _Sample, finish: str) -> float:
        try:
            return await score_answer(self.reward, self.reward_pool, s.group.answer, s.tokens, finish,
                                      self.cfg.reward_timeout_s if self.reward_pool is not None else None)
        except TimeoutError:  # counted in events; the analysis flags the row (results.harness_problem)
            self.events["reward_timeouts"] += 1
            return 0.0

    def _engine_lane(self):
        t, n = self.now(), len(self.live)
        self._live_area += (t - self._t_live) * self._n_live
        self._t_live, self._n_live = t, n
        if not self.syncing:
            self.tracer.state("engine", "generating" if n else "idle")

    # -- batching

    def _group_version(self, g: _Group) -> int:
        return min(self.ledger.first_version(s.sid) for s in g.samples)

    def _gate(self) -> tuple[str, list[_Group] | None]:
        B, floor = self.cfg.B, self.next_step + 1 - self.cfg.K
        if len(self.completed_groups) < B:
            return "wait_data", None
        ordered = sorted(self.completed_groups, key=lambda g: (self._group_version(g), g.order))
        batch, rest = ordered[:B], ordered[B:]
        if any(self._group_version(g) < floor for g in [*rest, *self.inflight_groups.values()]):
            return "wait_gate", None
        return "ready", batch

    async def _batch_loop(self):
        while True:
            self._kick["batch"].clear()
            if not self.trainer_busy and not self.stopping:
                state, groups = self._gate()
                if state == "ready":
                    await self._send_batch(groups)
                else:
                    self.tracer.state("batcher", state)
            await self._kick["batch"].wait()

    async def _send_batch(self, groups: list[_Group]):
        k, K = self.next_step, self.cfg.K
        rewards = [[s.reward for s in g.samples] for g in groups]
        samples, lags, depths, codes = [], [], [], []
        for g, adv in zip(groups, advantages(rewards)):
            for s, a in zip(g.samples, adv):
                lag = self.ledger.consume(s.sid, k)
                depth, ctx = self.ledger.kv_depth(s.sid), self.ledger.token_context_codes(s.sid)
                if lag.size and lag.max() > K:
                    self.violations.append(f"I7: {s.sid} consumed at step {k} with lag {lag.max()} > K={K}")
                samples.append(TrainSample(s.sid, g.gid, g.prompt_ids, tuple(s.tokens), tuple(s.logprobs),
                                           s.reward, float(a), tuple(lag.tolist()), tuple(depth.tolist()),
                                           tuple(ctx.tolist())))
                lags.append(lag)
                depths.append(depth)
                codes.append(ctx)
            self.completed_groups.remove(g)
        zero = sum(zero_variance(r) for r in rewards)
        lag, depth, code = (np.concatenate(x) if x else np.zeros(0, np.int64) for x in (lags, depths, codes))
        self.tokens["consumed"] += int(lag.size)
        record = {"step": k, "t": self.now() - self.t_start, "groups": len(groups), "zero_variance": zero,
                  "tokens": int(lag.size), "reward_mean": float(np.mean([s.reward for s in samples])),
                  "lag_mean": float(lag.mean()) if lag.size else None, "lag_max": int(lag.max()) if lag.size else 0,
                  "lag_hist": metrics.histogram(lag, K + 2), "depth_max": int(depth.max()) if depth.size else 0,
                  "length_mean": float(np.mean([len(s.token_ids) for s in samples])) if samples else 0.0}
        for i, c in enumerate(CONTEXTS):
            record[f"frac_{c}"] = float((code == i).mean()) if code.size else 0.0
        self.steps_log.append(record)
        if self.journal:
            self.journal.write(json.dumps(record, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))
                               + "\n")
        self.next_step += 1
        self.trainer_busy = True
        self.tracer.state("batcher", "training", step=k)
        self._watch = self._loop.call_later(self.cfg.timeout_s, self._fail,
                                            TimeoutError(f"trainer silent for {self.cfg.timeout_s}s on step {k}"))
        await self._call(self.trainer.submit(TrainBatch(k, tuple(samples), len(groups), zero)))
        self._kick["admit"].set()

    async def _ready_loop(self):
        while True:
            ready = await self.trainer.next_ready()
            if ready.version != self.staged + 1:
                raise ControllerError(f"R5: trainer staged v{ready.version} after v{self.staged}")
            if self._watch:
                self._watch.cancel()
            self.staged = ready.version
            self.trainer_busy = False
            self.steps_done += 1
            self.steps_log[ready.version - 1 - self._v0].update(ready.stats)
            if self.staged - self.v_engine > self.cfg.K + 1:
                self.violations.append(f"I13: trainer at v{self.staged}, engine at v{self.v_engine}")
            n = self.cfg.snapshot_every_steps
            if n and ready.version % n == 0:
                await self._snapshot(f"step{ready.version:04d}")
            if self.cfg.max_steps is not None and self.steps_done >= self.cfg.max_steps:
                self._stop("max_steps")
            self._kick["batch"].set()
            self._kick["sync"].set()

    async def _snapshot_loop(self):
        """At every snapshot_every_s of wall clock before the budget the trainer saves its latest staged weights for
        offline evaluation (worker.py copies them when the mark arrives); the mark at the budget itself, "tend", is
        taken by run() once the budget stops the run."""
        i = 0
        while True:
            i += 1
            at = i * self.cfg.snapshot_every_s
            if self.cfg.budget_s is not None and at >= self.cfg.budget_s - 1e-9:
                return
            await asyncio.sleep(self.t_start + at - self.now())
            await self._snapshot(f"t{i:02d}")

    async def _snapshot(self, tag: str):
        """The mark: the version staged now is the one saved (the worker pins it on arrival), so a snapshot never
        holds weights from after its mark, however late the copy runs."""
        self.snapshots.append({"tag": tag, "t": self.now() - self.t_start, "staged": self.staged,
                               "engine": self.v_engine, "steps": self.steps_done})
        await self._call(self.trainer.snapshot(tag, self.staged))

    # -- syncs

    async def _sync_loop(self):
        while True:
            self._kick["sync"].clear()
            if not self.stopping and self.staged - self.v_engine >= self.cfg.S:
                await self._sync(self.staged)
                continue
            await self._kick["sync"].wait()

    async def _generated(self) -> int:
        """The engine's count of tokens generated since this run began."""
        return (await self._call(self.engine.counters())).generated_tokens_total - self._generated0

    def _count_reset(self):
        self._preempt_expected += sum(1 for s in self.live.values() if s.tokens)
        self._preempt_allowed += len(self.live)

    async def _check_preemptions(self):
        """I5, cumulatively: preemptions beyond those E's resets can explain (every live request at each reset) are
        KV-pressure preemptions, which make a D answer re-prefill on new weights unseen. Counts can lag (vLLM
        publishes them with the next step), so only an excess is a violation, whenever it shows up."""
        total = (await self._call(self.engine.counters())).preemptions_total
        self.events["reset_preemptions_without_tokens"] = max(0, min(total, self._preempt_allowed)
                                                              - self._preempt_expected)
        excess = total - self._preempt_allowed
        if excess > self._kv_preemptions:
            self.violations.append(f"I5: {excess - self._kv_preemptions} KV-pressure preemptions "
                                   f"(at engine version v{self.v_engine})")
            self._kv_preemptions = excess

    async def _sync(self, v: int):
        """A sync, phase by phase: pause, push (the engine's count read alongside it, the weights verified, the
        count settled), cut, resume, record. The same calls in the same order for every regime; each phase is a
        method so the tests pin its order and the canary repeats the push's cut (fence.exact_cut)."""
        sync = _Sync(v=v, t0=self.now())
        await self._sync_pause(sync)
        await self._sync_push(sync)
        self._sync_cut(sync)
        await self._sync_resume(sync)
        self._sync_record(sync)
        await self._check_preemptions()  # after resume: the scrape stays off the pause's critical path

    async def _sync_pause(self, sync: "_Sync"):
        """The pause the regime calls for (none for A and F), its acknowledgement, and, under abort, the aborted
        streams' ends; every in-flight answer's length at that point is the fence's baseline for late tokens."""
        r = self.regime
        # F never pauses: answers keep streaming and new groups keep starting on the current adapter during its push
        self.syncing = r.pause is not None or r.synchronous
        self._abort_phase = r.pause == "abort"
        self.tracer.state("sync", "sync", version=sync.v)
        if r.pause is not None:
            self.tracer.state("engine", "draining" if r.pause == "wait" else "pausing")
            await self._call(self.engine.pause(r.pause, r.clear_cache))
            self._reset_unaccounted = r.pause == "keep" and r.clear_cache  # vLLM counts a reset's preemptions
        sync.t_paused = self.now()
        # a keep pause leaves answers in flight whose old tokens may still be on the way: their latest arrival after
        # the acknowledgement is this sync's ack lag
        self._t_ack, self._late_s = (sync.t_paused if r.pause == "keep" else None), 0.0
        if r.pause is not None:
            self.tracer.state("engine", "paused")
            if r.pause == "abort":  # running requests always abort; one with no token yet may be queued behind the
                # pause, whose stream will not end (a C1 continuation holds earlier tokens: count this request's)
                running = [s.task for s in self.live.values() if s.got and s.task and not s.task.done()]
                if running:
                    await self._call(asyncio.wait(running))
            sync.lengths_at_fence = {sid: len(s.tokens) for sid, s in self.live.items()}
            sync.fence_s = self.now() - sync.t_paused
        if self._reset_unaccounted:
            self._count_reset()
            self._reset_unaccounted = False

    async def _sync_push(self, sync: "_Sync"):
        """The push. Under a pause it is the exact cut (fence.exact_cut): the engine's generated-token count read
        alongside the push, the weights verified (_sync_verify), then the wait until every counted token has
        arrived; without one (A, F) the push and the verification alone."""
        r = self.regime
        if r.pause is None:
            await self._sync_verify(sync, await self._call(self.trainer.push(sync.v)))
            return
        self.tracer.state("engine", "receiving")
        # a server behind at most syncs (one saturated server: the count moved after the ack at 45 of 55) may be
        # behind at the next; a rare late burst (four servers: 3-5 syncs in 300, the push contending with output
        # processing) is handled within its own sync, and must not tax the rest of the run by when it came
        lagging = self.events["count_lagged_syncs"] > max(2, len(self.syncs) / 4)
        _, sync.first, sync.generated, sync.settled, waited = await exact_cut(
            self._call(self.trainer.push(sync.v)), self._generated, lambda: self.tokens["delivered"], self.now,
            settle_s=LAGGING_SETTLE_S if lagging else SETTLE_S, between=lambda pushed: self._sync_verify(sync, pushed))
        self.events["count_lagged_syncs"] += sync.generated > sync.first
        sync.fence_s += waited

    async def _sync_verify(self, sync: "_Sync", pushed):
        """R5 and I6: the trainer pushed at least the granted version (the newest staged version wins, and tokens are
        labelled with what the engine actually holds), and the engine holds it (full weights), or the adapter is
        loaded (in place, or under a new name with the unused older ones unloaded: F)."""
        if pushed.version < sync.v:
            raise ControllerError(f"R5: granted v{sync.v}, trainer pushed v{pushed.version}")
        sync.pushed, sync.v = pushed, pushed.version
        r, v = self.regime, pushed.version
        if r.weights == "full":
            if (held := await self._call(self.engine.weight_version())) != v:
                raise ControllerError(f"I6: engine holds v{held} after pushing v{v}")
        elif r.weights == "lora_inplace":
            await self._call(self.engine.load_adapter("policy", pushed.adapter_path, inplace=True))
        else:
            name = f"policy-v{v}"
            await self._call(self.engine.load_adapter(name, pushed.adapter_path, inplace=False))
            self.adapter_refs[name] = 0
            self.v_engine = v  # from here new groups start on `name`, so an unused older adapter stays unused
            for old in [n for n, refs in self.adapter_refs.items() if refs == 0 and n != name]:
                del self.adapter_refs[old]
                await self._call(self.engine.unload_adapter(old))

    def _sync_cut(self, sync: "_Sync"):
        """The boundary: every in-flight answer is cut at the tokens delivered now (keep: the new version over the
        old KV or re-prefilled; drain: nothing in flight; abort: resubmitted whole (C1) or discarded (C2)); a queued
        request starts on the new weights."""
        r, v = self.regime, sync.v
        sync.ack_lag = self._late_s if r.pause == "keep" else None
        self._t_ack = None
        sync.t_cut, sync.delivered_at_cut = self.now(), self.tokens["delivered"]
        sync.late = sum(len(s.tokens) - sync.lengths_at_fence[sid] for sid, s in self.live.items()
                        if sid in sync.lengths_at_fence)
        for s in list(self.live.values()):
            n = len(s.tokens)
            if r.pause is None:
                if r.synchronous:
                    self.violations.append(f"{s.sid}: in flight at a synchronous sync")
                continue
            if r.pause == "keep":
                self.ledger.cut(s.sid, n, v, r.post_swap_context if n else "pure")
                if r.clear_cache and n:
                    sync.reprefill += len(s.group.prompt_ids) + n
            elif r.pause == "wait":
                if n:
                    self.violations.append(f"I3: {s.sid} still running with {n} tokens after the drain")
                self.ledger.cut(s.sid, n, v, "pure")
            elif not s.aborted:
                if s.got:
                    self.violations.append(f"{s.sid}: running with {s.got} tokens but not aborted by pause(abort)")
                else:  # it reached the engine after the pause: queued, it starts on the new weights (a C1
                    # continuation's prefix too, re-prefilled: the cut at its open segment's start relabels it)
                    self.ledger.cut(s.sid, n, v, "pure")
                    sync.queued += 1
            elif r.on_abort == "resume_partial":
                self.ledger.cut(s.sid, n, v, r.post_swap_context if n else "pure")
                sync.reprefill += len(s.group.prompt_ids) + n if n else 0
                s.cont += 1
                sync.resubmit.append(s)
            else:
                self.tokens["discarded"] += self.ledger.discard(s.sid, "abort", sync.t_cut)
                s.attempt = self.ledger.open(s.sid, v, group=s.group.gid, t=sync.t_cut,
                                             prompt_len=len(s.group.prompt_ids))
                s.cont, s.tokens, s.logprobs = 0, [], []
                sync.resubmit.append(s)
            s.aborted = False
        self._abort_phase = False

    async def _sync_resume(self, sync: "_Sync"):
        r, v = self.regime, sync.v
        sync.skipped = v - self.syncs[-1]["version"] - 1 if self.syncs else v - self._v0 - 1
        self.v_engine = v
        if r.pause is not None:
            self._resuming = True
            await self._call(self.engine.resume())
            self._resuming = False
        for s in sync.resubmit:
            self._start(s)

    def _sync_record(self, sync: "_Sync"):
        """The sync's record (syncs.parquet), the run's counters, and I9: the engine's count at the pause against the
        tokens delivered by the cut."""
        v, pushed = sync.v, sync.pushed
        self.versions_skipped += sync.skipped
        self.tokens["reprefilled"] += sync.reprefill
        self.late_old_tokens += sync.late
        self.events["queued_at_abort"] += sync.queued
        self.syncs.append({"version": v, "t": sync.t0 - self.t_start, "pause_s": sync.t_paused - sync.t0,
                           "sync_s": self.now() - sync.t0, "push_s": pushed.seconds, "bytes": pushed.bytes,
                           "inflight": len(self.live), "late_old_tokens": sync.late,
                           "reprefill_tokens": sync.reprefill, "resubmitted": len(sync.resubmit),
                           "queued": sync.queued, "skipped": sync.skipped,
                           # the fence's cost and what set it: the latest old token after a keep pause's ack
                           "fence_s": sync.fence_s, "ack_lag_s": sync.ack_lag,
                           # > 0: tokens the engine had generated before the pause that had not arrived by the cut
                           "undelivered": None if sync.generated is None else sync.generated - sync.delivered_at_cut,
                           # how far the engine's count rose after its first read, at the ack: an API server still
                           # processing old outputs then (the second read's reason to exist)
                           "count_moved": None if sync.generated is None else sync.generated - sync.first})
        if sync.generated is not None and sync.generated != sync.delivered_at_cut:
            # fewer: old tokens on the way; more: the count is not ours
            self.violations.append(f"I9: the engine had generated {sync.generated} tokens by the pause at v{v}, "
                                   f"{sync.delivered_at_cut} had arrived by the cut")
        if not sync.settled:
            self.violations.append(f"I9: the engine's count was still moving when the fence gave up at v{v}")
        self.syncing = False
        self.tracer.state("sync", "between")
        self._engine_lane()
        self._kick_all()

    # -- shutdown

    async def _call_quietly(self, aw, what: str):
        """A shutdown call after the run has already failed: bounded, and its failure recorded, not raised, so the
        row is written whatever state the engine or trainer is in (a dead engine used to lose the row)."""
        try:
            return await asyncio.wait_for(aw, SHUTDOWN_CALL_S)
        except Exception as e:  # noqa: BLE001 - every failure here is reported in the row, none can be handled
            self.events.setdefault("shutdown_errors", []).append(f"{what}: {type(e).__name__}: {e}")
            return None

    async def _shutdown(self, loops, failed: bool = False):
        self.stopping = True
        if self._watch:
            self._watch.cancel()
        streams = list(self._streams)  # in flight, and finished ones still being scored
        for t in [*loops, *streams]:
            t.cancel()
        await asyncio.gather(*loops, *streams, return_exceptions=True)
        self.wall_s = self.now() - self.t_start
        self._engine_lane()
        if self._reset_unaccounted:  # the budget ended a sync between E's reset and its accounting
            self._count_reset()
        for sid in list(self.live):
            self.ledger.discard(sid, "cancelled", self.now())
        self.live.clear()
        if not failed:
            self.trainer_summary = await self._call(self.trainer.close())
            await self._check_preemptions()
            self.engine_counters = await self._call(self.engine.counters())
            await self._call(self.engine.close())
        else:
            self.trainer_summary = await self._call_quietly(self.trainer.close(), "trainer.close") or {}
            await self._call_quietly(self._check_preemptions(), "preemptions")
            counters = await self._call_quietly(self.engine.counters(), "engine.counters")
            if counters is not None:
                self.engine_counters = counters
            await self._call_quietly(self.engine.close(), "engine.close")
        self.tracer.close()

    def _summary(self, error) -> RunSummary:
        counts = self.ledger.counts()
        consumed = counts["consumed"]
        unconsumed = counts["closed"] - consumed
        cancelled = sum(d["reason"] == "cancelled" for d in self.ledger.discards())
        violations = list(self.violations)
        if self.dispatched * self.cfg.G != consumed + unconsumed + cancelled:
            violations.append(f"I8: {self.dispatched * self.cfg.G} answers dispatched, {consumed} consumed + "
                              f"{unconsumed} unconsumed + {cancelled} cancelled")
        violations += check_ledger(self.regime, self.ledger)
        violations += check_lanes(self.tracer.events)
        # every mark must be on disk: the one at T is the primary endpoint, so a run without it is not a result
        written = {t for t, *_ in getattr(self.trainer, "snapshots", [])}
        lost = [m["tag"] for m in self.snapshots if m["tag"] not in written]
        if error is None and lost:
            violations.append(f"I16: snapshots {lost} were never written")
        violations += [f"I16: snapshot {t} failed: {tb.strip().splitlines()[-1]}"
                       for t, tb in getattr(self.trainer, "snapshot_failures", [])]
        ev = self.tracer.events
        return RunSummary(
            regime=self.regime.label, valid=error is None and not violations,
            error=None if error is None else f"{type(error).__name__}: {error}", violations=violations,
            wall_s=self.wall_s, steps=self.steps_done, groups_submitted=self.dispatched,
            samples={"consumed": consumed, "unconsumed": unconsumed, "cancelled": cancelled,
                     "attempts": counts["attempts"], "discarded_attempts": counts["discarded"] - cancelled},
            tokens=dict(self.tokens), late_old_tokens=self.late_old_tokens, versions_skipped=self.versions_skipped,
            inflight_mean=self._live_area / self.wall_s if self.wall_s else 0.0, events=dict(self.events),
            lanes={lane: lane_fractions(ev, lane) for lane in ("engine", "batcher")},  # the trainer adds its own
            syncs=self.syncs, steps_log=self.steps_log, engine=asdict(self.engine_counters),
            trainer=self.trainer_summary)
