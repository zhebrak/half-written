"""A timeline of what the engine, the trainer and the sync were doing, as gap-free lanes of states.

Each lane is in exactly one state at a time: state() closes the lane's current state and opens the next, so the
time a lane spends in each state sums to its span (invariant I12). Lanes used by the controller and trainers:

    engine   generating | idle | draining | pausing | paused | receiving
    batcher  wait_data | wait_gate | training       (why the trainer has no batch, or that it has one)
    trainer  idle | fwd_bwd | stage | stage_wait | wait_push   (stage_wait: no staging buffer free; wait_push: a
                                                          lock-step trainer waits for its own push)
    sender   idle | send                            (the staged push, concurrent with fwd_bwd)
    sync     between | sync

Exports: Chrome trace JSON (open in Perfetto or chrome://tracing) and a pyarrow table (trace.parquet).
"""
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Event:
    kind: str  # state | counter | instant
    proc: str
    lane: str
    name: str
    t0: float
    t1: float
    value: float = 0.0
    args: dict | None = None


class Tracer:
    def __init__(self, now: Callable[[], float], proc: str = "controller"):
        self.now, self.proc = now, proc
        self.events: list[Event] = []
        self._open: dict[str, tuple[str, float, dict]] = {}

    def state(self, lane: str, name: str, **args) -> None:
        t = self.now()
        cur = self._open.get(lane)
        if cur is not None:
            if cur[0] == name and not args:
                return
            if t > cur[1]:
                self.events.append(Event("state", self.proc, lane, cur[0], cur[1], t, args=cur[2] or None))
        self._open[lane] = (name, t, args)

    def counter(self, name: str, value: float) -> None:
        t = self.now()
        self.events.append(Event("counter", self.proc, name, name, t, t, float(value)))

    def instant(self, lane: str, name: str, **args) -> None:
        t = self.now()
        self.events.append(Event("instant", self.proc, lane, name, t, t, args=args or None))

    def close(self) -> list[Event]:
        """Closes every open state at the current time and returns all events."""
        t = self.now()
        for lane, (name, t0, args) in self._open.items():
            if t > t0:
                self.events.append(Event("state", self.proc, lane, name, t0, t, args=args or None))
        self._open.clear()
        return self.events


def merge(*event_lists: list[Event]) -> list[Event]:
    return sorted((e for events in event_lists for e in events), key=lambda e: (e.t0, e.t1))


def lane_totals(events: list[Event], lane: str, proc: str | None = None) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for e in events:
        if e.kind == "state" and e.lane == lane and (proc is None or e.proc == proc):
            out[e.name] += e.t1 - e.t0
    return dict(out)


def lane_fractions(events: list[Event], lane: str, proc: str | None = None) -> dict[str, float]:
    totals = lane_totals(events, lane, proc)
    span = sum(totals.values())
    return {k: v / span for k, v in totals.items()} if span else {}


def check_lanes(events: list[Event], tol: float = 1e-9) -> list[str]:
    """I12: within each (proc, lane), consecutive states neither overlap nor leave a gap."""
    by_lane: dict[tuple[str, str], list[Event]] = defaultdict(list)
    for e in events:
        if e.kind == "state":
            by_lane[(e.proc, e.lane)].append(e)
    bad = []
    for (proc, lane), states in by_lane.items():
        states.sort(key=lambda e: e.t0)
        for a, b in zip(states, states[1:]):
            if abs(b.t0 - a.t1) > tol:
                bad.append(f"{proc}/{lane}: {a.name} ends at {a.t1:.6f}, {b.name} starts at {b.t0:.6f}")
    return bad


def to_chrome(events: list[Event]) -> dict:
    """Chrome trace format: one pid per process, one tid per lane, states as complete events (ph X)."""
    pids, tids, out = {}, {}, []
    for e in events:
        if e.proc not in pids:
            pids[e.proc] = len(pids) + 1
            out.append({"ph": "M", "name": "process_name", "pid": pids[e.proc], "tid": 0, "args": {"name": e.proc}})
        pid = pids[e.proc]
        key = (e.proc, e.lane)
        if key not in tids and e.kind != "counter":
            tids[key] = len(tids) + 1
            out.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tids[key], "args": {"name": e.lane}})
        ts = round(e.t0 * 1e6, 3)
        if e.kind == "state":
            out.append({"ph": "X", "name": e.name, "pid": pid, "tid": tids[key], "ts": ts,
                        "dur": round((e.t1 - e.t0) * 1e6, 3), "args": e.args or {}})
        elif e.kind == "counter":
            out.append({"ph": "C", "name": e.name, "pid": pid, "ts": ts, "args": {e.name: e.value}})
        else:
            out.append({"ph": "i", "s": "t", "name": e.name, "pid": pid, "tid": tids[key], "ts": ts,
                        "args": e.args or {}})
    return {"traceEvents": out, "displayTimeUnit": "ms"}


def to_table(events: list[Event]):
    import pyarrow as pa
    return pa.Table.from_pylist([{"kind": e.kind, "proc": e.proc, "lane": e.lane, "name": e.name, "t0": e.t0,
                                  "t1": e.t1, "value": e.value} for e in events])
