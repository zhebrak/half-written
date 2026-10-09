"""The values that cross module boundaries: requests and chunks between the controller and an engine, ledger
segments, and the messages between the controller and the trainer.

Request ids encode the sample, its attempt and its continuation: `<sid>#<attempt>` for a fresh attempt and
`<sid>#<attempt>+<k>` for the k-th continuation of that attempt (C1 resubmits prompt + partial tokens as a new
request). An engine never needs to parse them; FakeEngine does, so that a continuation keeps the answer length the
attempt drew.
"""
from dataclasses import dataclass, field
from typing import Literal

PauseMode = Literal["wait", "abort", "keep"]
Finish = Literal["stop", "length", "abort"]
Context = Literal["pure", "stale_kv", "recomputed", "resubmitted"]

# Context codes, as the per-token arrays store them. A token's context says what KV its logits attended to:
# pure        - KV computed by the same version that sampled it (no swap inside the sequence so far)
# stale_kv    - KV partly computed by older versions (D: the swap kept the old cache)
# recomputed  - every KV entry recomputed by the sampling version after a swap (E: in-engine re-prefill)
# resubmitted - as recomputed, but by a new request carrying prompt + partial tokens (C1)
CONTEXTS: tuple[Context, ...] = ("pure", "stale_kv", "recomputed", "resubmitted")
CONTEXT_CODE = {c: i for i, c in enumerate(CONTEXTS)}

BUDGET_MARK = "tend"  # the snapshot taken at exactly the wall-clock budget T: the primary endpoint's weights


def snapshot_name(tag: str, version: int) -> str:
    """A snapshot's directory name: the mark's tag and the version it holds ("t03-v17")."""
    return f"{tag}-v{version}"


def parse_snapshot(name: str) -> tuple[str, int | None]:
    """(tag, version) of a snapshot's name: ("t03", 17) for "t03-v17", ("init", 0) for the starting weights, and
    (name, None) when the name carries no version."""
    if name == "init":
        return "init", 0
    tag, sep, v = str(name).rpartition("-v")
    return (tag, int(v)) if sep and v.isdigit() else (str(name), None)


def rid(sid: str, attempt: int, continuation: int = 0) -> str:
    return f"{sid}#{attempt}" + (f"+{continuation}" if continuation else "")


def parse_rid(r: str) -> tuple[str, int, int]:
    """(sid, attempt, continuation) of a request id written by rid()."""
    head, _, cont = r.partition("+")
    sid, _, attempt = head.rpartition("#")
    return sid, int(attempt), int(cont or 0)


@dataclass(frozen=True, slots=True)
class GenRequest:
    rid: str
    prompt_ids: tuple[int, ...]  # C1 continuations: the original prompt + the partial answer, as token ids
    max_tokens: int
    temperature: float = 1.0
    adapter: str | None = None  # a LoRA adapter name; None serves the base weights
    seed: int | None = None  # canary only: a seeded generator replays after an E-mode preemption
    ignore_eos: bool = False  # forced lengths (systems map, canary): generate exactly max_tokens
    stop: tuple[str, ...] = ()  # stop strings; the matched text is not returned


@dataclass(frozen=True, slots=True)
class Chunk:
    """Tokens of one request as delivered to the client; the last chunk of a request carries `finish`. An abort
    ends with an empty chunk (vLLM's DELTA stream), so the client keeps the tokens it has accumulated."""
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    finish: Finish | None = None


@dataclass(frozen=True, slots=True)
class Segment:
    """Tokens [start, end) of one attempt, sampled by weight version `version` with KV of kind `context`."""
    start: int
    end: int
    version: int
    context: Context


@dataclass(frozen=True, slots=True)
class TrainSample:
    sid: str
    group: str
    prompt_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]  # mu: the engine's processed logprob of each token, stale-KV tokens included
    reward: float
    advantage: float
    lag: tuple[int, ...]  # per token: consuming step - version that sampled it
    depth: tuple[int, ...]  # per token: version that sampled it - oldest version among the KV it attended to
    context_codes: tuple[int, ...]  # per token: CONTEXT_CODE of its context


@dataclass(frozen=True, slots=True)
class TrainBatch:
    step: int  # the optimiser step this batch feeds; its gradient is taken at weights of version `step`
    samples: tuple[TrainSample, ...]
    groups: int
    zero_variance_groups: int  # consumed with zero advantage, never resampled


@dataclass(frozen=True, slots=True)
class WeightsReady:
    """The trainer finished a step and staged the weights of `version`; it does not wait for the push."""
    version: int
    stats: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Pushed:
    version: int
    bytes: int
    seconds: float
    adapter_path: str | None = None  # LoRA: where the adapter of `version` was written
