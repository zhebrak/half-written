"""The weight-update regimes: what happens to in-flight answers when the trainer publishes new weights.

    A   split synchronous   generate the batch, train, push; the engine is idle at every sync (K=0)
    B   drain               pause(wait): running answers finish on the old weights, then push
    C1  abort, resume       pause(abort), push, resubmit prompt + partial token ids (context re-prefilled)
    C2  abort, discard      pause(abort), push, resubmit the prompt only (partial answers are waste)
    D   in-flight, stale KV pause(keep, clear_cache=False), push: answers continue reading old-weights KV
    E   in-flight, fresh KV pause(keep, clear_cache=True), push: vLLM preempts and re-prefills them on new weights
    F   versioned LoRA      no pause: the new adapter gets a new name; answers finish on the adapter they started on

With lora=True, A-E push an adapter loaded in place under one name instead of full weights over NCCL. A1 (one GPU,
colocated) and A2 (two GPUs, colocated) are synchronous like A; the study emulates their timing from A's learning
curve (PLAN.md §4.6), so the controller does not run them.
"""
from dataclasses import dataclass, replace
from typing import Literal

from .types import Context, PauseMode

Weights = Literal["full", "lora_inplace", "lora_versioned"]


@dataclass(frozen=True)
class Regime:
    name: str
    pause: PauseMode | None  # None: A (the engine is idle at a sync) and F (versioned adapters need no pause)
    clear_cache: bool  # passed explicitly: vLLM's default (True) silently turns keep into E
    on_abort: Literal["resume_partial", "restart"] | None  # C1, C2
    post_swap_context: Context | None  # of the tokens after a boundary: D stale_kv, E recomputed, C1 resubmitted
    weights: Weights = "full"
    synchronous: bool = False  # A: K must be 0, the next batch is generated only after the push
    implemented: bool = True  # A1, A2: emulated, not run by the controller

    @property
    def label(self) -> str:
        return self.name + ("+lora" if self.weights == "lora_inplace" else "")


REGIMES: dict[str, Regime] = {
    "A": Regime("A", None, False, None, None, synchronous=True),
    "B": Regime("B", "wait", True, None, None),
    "C1": Regime("C1", "abort", True, "resume_partial", "resubmitted"),
    "C2": Regime("C2", "abort", True, "restart", None),
    "D": Regime("D", "keep", False, None, "stale_kv"),
    "E": Regime("E", "keep", True, None, "recomputed"),
    "F": Regime("F", None, False, None, None, weights="lora_versioned"),
    "A1": Regime("A1", None, False, None, None, synchronous=True, implemented=False),
    "A2": Regime("A2", None, False, None, None, synchronous=True, implemented=False),
}

# Contexts each regime's tokens may carry (invariant I3).
ALLOWED: dict[str, set[str]] = {
    "A": {"pure"}, "B": {"pure"}, "C2": {"pure"}, "F": {"pure"},
    "C1": {"pure", "resubmitted"}, "D": {"pure", "stale_kv"}, "E": {"pure", "recomputed"},
}


def parse_arm(name: str) -> tuple[str, bool]:
    """(regime, lora) of an arm's name: "D+lora" is D with LoRA weights. F is LoRA by definition, but its name alone
    does not say so: a config naming F must set lora=True (resolve refuses one that does not), and a matrix or the
    calculator sets it for F."""
    regime = name.removesuffix("+lora")
    return regime, name.endswith("+lora")


def resolve(name: str, lora: bool = False) -> Regime:
    """The regime called `name`, with LoRA weights when `lora`. F is LoRA by definition; A1 and A2 are emulated."""
    name, named_lora = parse_arm(name)
    r = REGIMES.get(name)
    lora = lora or named_lora
    if r is None:
        raise ValueError(f"unknown regime {name!r}; one of {', '.join(REGIMES)}")
    if not r.implemented:
        raise ValueError(f"{name} is emulated from A's learning curve (PLAN.md §4.6), not run")
    if r.weights == "lora_versioned" and not lora:
        raise ValueError("F versions LoRA adapters: run it with lora=True")
    return replace(r, weights="lora_inplace") if lora and r.weights == "full" else r


def check_ledger(r: Regime, ledger) -> list[str]:
    """I3: the shape every closed attempt must have under regime `r`. Returns the violations, empty when none."""
    allowed, bad = ALLOWED[r.name], []
    for sid in ledger.closed():
        segs = ledger.segments(sid)
        if r.post_swap_context is None and len(segs) != 1:
            bad.append(f"{sid}: {len(segs)} segments under {r.name}, which never cuts an answer")
        if segs[0].context != "pure":
            bad.append(f"{sid}: first segment is {segs[0].context}")
        if any(s.context not in allowed for s in segs):
            bad.append(f"{sid}: contexts {[s.context for s in segs]} not allowed under {r.name}")
        if any(y.version <= x.version for x, y in zip(segs, segs[1:])):
            bad.append(f"{sid}: segment versions do not increase")
    return bad
