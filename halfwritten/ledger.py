"""Which weight version sampled each token, and what KV it attended to.

vLLM reports no per-request or per-token weight version (only the engine's current one, GET /weight_info), so the
controller, which owns every pause and resume, writes it here. A sample is one attempt at answering one prompt; its
tokens are tiled by segments [start, end) of one version and one context (types.CONTEXTS). Segment boundaries are
token counts *delivered to the client* before the resume that follows a swap (controller._sync, PLAN.md §5.3).

The KV of the last token delivered before a swap is computed after the swap, by the new weights (a decode step
computes the KV of its input token). So in D the stale region of a cut at n is positions < n-1 of the answer plus the
prompt, and every token after the cut attends to the prompt's KV, computed by the first segment's version. KV depth
therefore is `version - first version` for stale_kv tokens and 0 otherwise.

    ledger.open(sid, version)              a new attempt (or the next attempt after discard)
    ledger.extend(sid, n)                  n more tokens delivered
    ledger.cut(sid, at, version, context)  a swap after `at` delivered tokens; at == the open segment's start relabels
    ledger.discard(sid, reason)            drop the attempt (C2 abort, end of run); returns the wasted token count
    ledger.close(sid, finish)              the attempt finished; returns its segments
    ledger.consume(sid, step)              the attempt feeds optimiser step `step`; returns per-token lags
"""
from dataclasses import dataclass, field

import numpy as np

from .types import CONTEXT_CODE, CONTEXTS, Context, Finish, Segment


class LedgerError(Exception):
    pass


@dataclass
class _Attempt:
    sid: str
    group: str | None
    attempt: int
    starts: list[int] = field(default_factory=lambda: [0])
    versions: list[int] = field(default_factory=list)
    contexts: list[Context] = field(default_factory=lambda: ["pure"])
    n: int = 0
    finish: Finish | None = None
    consumed: int | None = None
    t_open: float = 0.0
    t_close: float | None = None
    meta: dict = field(default_factory=dict)

    def segments(self) -> tuple[Segment, ...]:
        ends = self.starts[1:] + [self.n]
        return tuple(Segment(s, e, v, c) for s, e, v, c in zip(self.starts, ends, self.versions, self.contexts))


class Ledger:
    def __init__(self):
        self._open: dict[str, _Attempt] = {}
        self._closed: dict[str, _Attempt] = {}
        self._discards: list[dict] = []
        self._attempts: dict[str, int] = {}

    # -- writing

    def open(self, sid: str, version: int, group: str | None = None, t: float = 0.0, **meta) -> int:
        """Opens the next attempt of `sid` at `version` and returns its attempt number (0 first)."""
        if sid in self._open or sid in self._closed:
            raise LedgerError(f"{sid}: already open or closed")
        attempt = self._attempts.get(sid, -1) + 1
        self._attempts[sid] = attempt
        self._open[sid] = _Attempt(sid, group, attempt, versions=[version], t_open=t, meta=dict(meta))
        return attempt

    def extend(self, sid: str, n: int) -> None:
        if n < 0:
            raise LedgerError(f"{sid}: extend by {n}")
        self._get(sid).n += n

    def cut(self, sid: str, at: int, version: int, context: Context) -> None:
        """A swap to `version` after `at` delivered tokens. `at` must be the attempt's current length: a token that
        arrives after the snapshot the cut was taken from would be labelled with the wrong version. A cut at the open
        segment's start (nothing delivered since the last boundary) relabels that segment's version and keeps its
        context: an attempt with no tokens yet stays pure, one cut twice in D stays stale_kv."""
        a = self._get(sid)
        if at != a.n:
            raise LedgerError(f"{sid}: cut at {at} but {a.n} tokens delivered")
        if version < a.versions[-1]:
            raise LedgerError(f"{sid}: version {version} after {a.versions[-1]}")
        if context not in CONTEXT_CODE:
            raise LedgerError(f"{sid}: unknown context {context!r}")
        if at == a.starts[-1]:
            a.versions[-1] = version
            return
        a.starts.append(at)
        a.versions.append(version)
        a.contexts.append(context)

    def discard(self, sid: str, reason: str, t: float = 0.0) -> int:
        """Drops the open attempt (its tokens are waste) and returns how many tokens it had."""
        a = self._open.pop(sid, None)
        if a is None:
            raise LedgerError(f"{sid}: not open")
        self._discards.append({"sid": sid, "group": a.group, "attempt": a.attempt, "reason": reason, "tokens": a.n,
                               "versions": list(a.versions), "t_open": a.t_open, "t_discard": t})
        return a.n

    def close(self, sid: str, finish: Finish, t: float = 0.0) -> tuple[Segment, ...]:
        """Closes the attempt; empty segments (a cut right before the last token) are dropped. Checks the tiling (I1)
        and that versions never decrease (I2)."""
        a = self._open.pop(sid, None)
        if a is None:
            raise LedgerError(f"{sid}: not open")
        keep = [i for i, seg in enumerate(a.segments()) if seg.end > seg.start] or [0]
        a.starts = [a.starts[i] for i in keep]
        a.versions = [a.versions[i] for i in keep]
        a.contexts = [a.contexts[i] for i in keep]
        a.starts[0] = 0
        a.finish, a.t_close = finish, t
        segs = a.segments()
        if segs[0].start != 0 or segs[-1].end != a.n or any(x.end != y.start for x, y in zip(segs, segs[1:])):
            raise LedgerError(f"{sid}: segments do not tile [0, {a.n}): {segs}")
        if any(y.version < x.version for x, y in zip(segs, segs[1:])):
            raise LedgerError(f"{sid}: versions decrease: {segs}")
        self._closed[sid] = a
        return segs

    def annotate(self, sid: str, **meta) -> None:
        self._get(sid, closed=True).meta.update(meta)

    def consume(self, sid: str, step: int) -> np.ndarray:
        """Marks a closed attempt as feeding optimiser step `step`; returns each token's lag, step - version (I2:
        never negative)."""
        a = self._closed.get(sid)
        if a is None:
            raise LedgerError(f"{sid}: not closed")
        if a.consumed is not None:
            raise LedgerError(f"{sid}: consumed twice")
        lag = step - self.token_versions(sid)
        if lag.size and lag.min() < 0:
            raise LedgerError(f"{sid}: negative lag at step {step}")
        a.consumed = step
        return lag

    # -- reading

    def _get(self, sid: str, closed: bool = False) -> _Attempt:
        a = self._open.get(sid) or (self._closed.get(sid) if closed else None)
        if a is None:
            raise LedgerError(f"{sid}: not open")
        return a

    def length(self, sid: str) -> int:
        return self._get(sid, closed=True).n

    def segments(self, sid: str) -> tuple[Segment, ...]:
        return self._get(sid, closed=True).segments()

    def first_version(self, sid: str) -> int:
        return self._get(sid, closed=True).versions[0]

    def attempt(self, sid: str) -> int:
        return self._get(sid, closed=True).attempt

    def token_versions(self, sid: str) -> np.ndarray:
        return _per_token(self.segments(sid), lambda s: s.version)

    def token_context_codes(self, sid: str) -> np.ndarray:
        return _per_token(self.segments(sid), lambda s: CONTEXT_CODE[s.context])

    def kv_depth(self, sid: str) -> np.ndarray:
        """Per token: its version minus the oldest version among the KV it attended to. Only stale_kv tokens attend
        to older KV, and the oldest is the prompt's, computed by the first segment's version."""
        segs = self.segments(sid)
        return _per_token(segs, lambda s: s.version - segs[0].version if s.context == "stale_kv" else 0)

    def rfc_spans(self, sid: str) -> list[dict]:
        """The attempt's versions in the token-span format of vLLM RFC #54443."""
        return [{"version": f"v{s.version}", "start": s.start, "end": s.end} for s in self.segments(sid)
                if s.end > s.start]

    def closed(self) -> list[str]:
        return list(self._closed)

    def discards(self) -> list[dict]:
        return list(self._discards)

    def counts(self) -> dict:
        closed = self._closed.values()
        return {"open": len(self._open), "closed": len(self._closed),
                "consumed": sum(a.consumed is not None for a in closed),
                "discarded": len(self._discards),
                "attempts": sum(n + 1 for n in self._attempts.values()),
                "discarded_tokens": sum(d["tokens"] for d in self._discards)}

    def table(self):
        """One row per closed attempt, segments as a list of structs (ledger.parquet)."""
        import pyarrow as pa
        rows = [{"sid": a.sid, "group": a.group, "attempt": a.attempt, "tokens": a.n, "finish": a.finish,
                 "consumed_step": a.consumed, "t_open": a.t_open, "t_close": a.t_close,
                 "reward": a.meta.get("reward"), "prompt_len": a.meta.get("prompt_len"),
                 "segments": [{"start": s.start, "end": s.end, "version": s.version, "context": s.context}
                              for s in a.segments()]}
                for a in self._closed.values()]
        return pa.Table.from_pylist(rows)

    def discards_table(self):
        import pyarrow as pa
        return pa.Table.from_pylist(self._discards)


def _per_token(segs, value) -> np.ndarray:
    out = np.empty(segs[-1].end if segs else 0, dtype=np.int64)
    for s in segs:
        out[s.start:s.end] = value(s)
    return out


__all__ = ["Ledger", "LedgerError", "CONTEXTS"]
