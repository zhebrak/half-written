"""The fence: when every old-weights token the engine computed before a pause has arrived (PLAN.md §5.3, E2).

An answer in flight is cut where its delivered tokens end when resume is sent, so every token computed before the
pause must have arrived by then. The engine's own count says when: vLLM 0.30 records a generated token
(vllm:generation_tokens_total) after a step's outputs are queued to their streams, so once the tokens delivered
equal the count read after the acknowledgement, and a second read agrees, nothing old is still on its way. The runs
(controller.py) and the session gate (bench/canary.py) cut by this one rule, so the gate certifies what the runs do.
It matched at all 545 cuts of the certification rows on two H100s (NOTEBOOK.md 2026-10-02); a quiet window on the
stream, the fence before it, cost D and E about a fifth of a run paused and could only guess at a backlog.
"""
import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

SETTLE_S = 0.02  # the second read's wait, when the count has not moved after the ack
LAGGING_SETTLE_S = 0.1  # ... and once it has: an API server is behind (one server saturated: up to 12k tokens late)


async def wait_delivered(generated: int, read_generated: Callable[[], Awaitable[int]], delivered: Callable[[], int],
                         now: Callable[[], float], timeout_s: float = 2.0, settle_s: float = SETTLE_S,
                         max_s: float = 10.0) -> tuple[int, bool]:
    """Waits until every token the paused engine generated has arrived, by the engine's own count. `generated` is
    that count, read after the acknowledgement (vllm:generation_tokens_total since the run began). Once our
    delivered count reaches it, a second read settle_s later must agree: an API server still processing old outputs
    would raise it. (vLLM 0.30 records the count after a step's outputs are queued to their streams,
    v1/engine/async_llm.py output_handler, so a counted token is on its way to us; a server that had not yet pulled
    the step from the engine core counts it late, which this second read and the controller's `count_moved` watch.)
    A server that is behind counts a step's tokens only after its outputs are on their way, so while it works
    through them both counts can stand still briefly: the cut needs both to hold across the settle, and a count that
    has moved after the ack (a server behind) waits LAGGING_SETTLE_S from then on in this sync (the controller keeps
    it for the run when the count moves at most syncs).
    Returns (the final count, whether it settled). The wait gives up after timeout_s in which neither count moved (a
    controller stall alone does not use it up) or max_s in all; unsettled, the cut is not exact, and the caller says
    so."""
    t0, first = now(), generated
    deadline, seen = t0 + timeout_s, (generated, delivered())
    while now() < min(deadline, t0 + max_s):
        if (generated, delivered()) != seen:
            deadline, seen = now() + timeout_s, (generated, delivered())
        if delivered() < generated:
            await asyncio.sleep(0.002)
            continue
        got = delivered()
        await asyncio.sleep(settle_s if generated == first else max(settle_s, LAGGING_SETTLE_S))
        again = await read_generated()
        if again == generated and delivered() == got:
            return generated, True
        generated = again
    return generated, False


async def exact_cut(push: Awaitable[Any], read_generated: Callable[[], Awaitable[int]], delivered: Callable[[], int],
                    now: Callable[[], float], settle_s: float = SETTLE_S,
                    between: Callable[[Any], Awaitable[None]] | None = None) -> tuple[Any, int, int, bool, float]:
    """The exact fence around a push, as the controller sequences it and the canary repeats it: the engine is paused,
    so its generated-token count is final and is read alongside the push; `between(pushed)` (the controller's
    verification of the pushed weights) is awaited; then wait_delivered holds until every counted token has
    arrived and both counts have settled. Returns (what the push returned, the count at the ack, the count at the
    cut, whether it settled, the seconds the wait took)."""
    pushed, first = await asyncio.gather(push, read_generated())
    if between is not None:
        await between(pushed)
    t = now()
    final, settled = await wait_delivered(first, read_generated, delivered, now, settle_s=settle_s)
    return pushed, first, final, settled, now() - t
