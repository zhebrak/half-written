"""FakeEngine reproduces vLLM v0.30's pause semantics (PLAN.md §1) and knows, per token, which weights sampled it."""
import asyncio

import pytest

from halfwritten.clock import run_sim
from halfwritten.engines.base import EngineError
from halfwritten.engines.fake import CostModel, FakeEngine, LengthModel
from halfwritten.types import GenRequest

PROMPT = tuple(range(8))
UNIT = CostModel(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)  # every step takes exactly one second


def engine(length=30, **kw):
    return FakeEngine(lengths=LengthModel(mean=length, sigma=0.0), cost=kw.pop("cost", UNIT), **kw)


async def collect(eng, rid, prompt=PROMPT, max_tokens=64, adapter=None):
    chunks = []
    async for c in eng.generate(GenRequest(rid, prompt, max_tokens, adapter=adapter)):
        chunks.append(c)
    return chunks


def tokens(chunks):
    return [t for c in chunks for t in c.token_ids]


def test_answer_runs_to_its_drawn_length():
    async def main():
        eng = engine(10)
        await eng.start()
        chunks = await collect(eng, "s#0")
        await eng.close()
        return chunks, eng.truth("s"), asyncio.get_running_loop().time()

    chunks, truth, t = run_sim(main())
    assert len(tokens(chunks)) == 10 and chunks[-1].finish == "stop"
    assert truth == [(0, False)] * 10
    assert t == pytest.approx(10)  # one token per one-second step


def test_cap_ends_with_length():
    async def main():
        eng = engine(10)
        await eng.start()
        return await collect(eng, "s#0", max_tokens=4)

    chunks = run_sim(main())
    assert len(tokens(chunks)) == 4 and chunks[-1].finish == "length"


def test_wait_drains_running_and_queued_requests_start_on_new_weights():
    async def main():
        eng = engine(20)
        await eng.start()
        first = asyncio.create_task(collect(eng, "a#0"))
        await asyncio.sleep(5.5)
        await eng.pause("wait", True)
        assert first.done() and await eng.is_paused()  # wait returned only after the running answer finished
        second = asyncio.create_task(collect(eng, "b#0"))  # submitted while paused: queued
        await asyncio.sleep(3)
        assert eng._waiting and not eng._running
        eng.apply_weights(1)
        await eng.resume()
        await second
        return eng.truth("a"), eng.truth("b")

    a, b = run_sim(main())
    assert a == [(0, False)] * 20 and b == [(1, False)] * 20


def test_abort_returns_partial_tokens_then_an_empty_abort_chunk():
    async def main():
        eng = engine(50)
        await eng.start()
        task = asyncio.create_task(collect(eng, "a#0"))
        await asyncio.sleep(7.5)
        await eng.pause("abort", True)
        return await task, eng.truth("a")

    chunks, truth = run_sim(main())
    assert chunks[-1].token_ids == () and chunks[-1].finish == "abort"
    assert len(tokens(chunks)) == len(truth) == 8  # the step in flight at t=7.5 completes before the pause


def _swap(clear_cache):
    async def main():
        eng = engine(30)
        await eng.start()
        task = asyncio.create_task(collect(eng, "a#0"))
        await asyncio.sleep(10.5)
        before = eng.prefill_tokens_total
        await eng.pause("keep", clear_cache)
        eng.apply_weights(1)
        await eng.resume()
        chunks = await task
        return chunks, eng.truth("a"), eng.preemptions_total, eng.prefill_tokens_total - before

    return run_sim(main())


def test_keep_without_clear_cache_continues_on_old_kv():
    chunks, truth, preemptions, prefill = _swap(clear_cache=False)  # arm D
    assert len(tokens(chunks)) == 30 and chunks[-1].finish == "stop"
    assert truth == [(0, False)] * 11 + [(1, True)] * 19
    assert preemptions == 0 and prefill == 0


def test_keep_with_clear_cache_preempts_and_reprefills_on_new_weights():
    chunks, truth, preemptions, prefill = _swap(clear_cache=True)  # arm E
    assert truth == [(0, False)] * 11 + [(1, False)] * 19
    assert preemptions == 1
    assert prefill == len(PROMPT) + 11 - 1  # prompt + answer so far, less the input token a decode step computes


def test_continuation_keeps_the_attempts_length():
    """C1: prompt + partial ids resubmitted as a new request finish the answer the attempt drew."""
    async def main():
        eng = engine(30)
        await eng.start()
        first = asyncio.create_task(collect(eng, "a#0"))
        await asyncio.sleep(10.5)
        await eng.pause("abort", True)
        partial = tokens(await first)
        eng.apply_weights(1)
        await eng.resume()
        rest = await collect(eng, "a#0+1", PROMPT + tuple(partial), 64 - len(partial))
        return partial, rest, eng.truth("a")

    partial, rest, truth = run_sim(main())
    assert len(partial) + len(tokens(rest)) == 30 and rest[-1].finish == "stop"
    assert truth == [(0, False)] * len(partial) + [(1, False)] * (30 - len(partial))


def test_kv_pressure_preempts_and_counts():
    async def main():
        eng = engine(40, kv_tokens=70)
        await eng.start()
        await asyncio.gather(collect(eng, "a#0"), collect(eng, "b#0"))
        return eng.preemptions_total

    assert run_sim(main()) >= 1


def test_weights_change_only_on_a_paused_or_idle_engine():
    async def main():
        eng = engine(30)
        await eng.start()
        eng.apply_weights(1)  # idle: allowed
        task = asyncio.create_task(collect(eng, "a#0"))
        await asyncio.sleep(2.5)
        with pytest.raises(EngineError):
            eng.apply_weights(2)
        task.cancel()

    run_sim(main())


def test_one_step_late_delivery_trails_by_a_step():
    async def main():
        eng = engine(3, delivery="one_step_late")
        await eng.start()
        loop, times = asyncio.get_running_loop(), []
        async for c in eng.generate(GenRequest("a#0", PROMPT, 64)):
            times.append(loop.time())
        return times

    assert run_sim(main()) == pytest.approx([2, 3, 4])


def test_lora_in_place_reads_old_kv_and_versioned_names_pin():
    async def main():
        eng = engine(20)
        await eng.start()
        await eng.load_adapter("policy", "/dev/shm/x/v0", inplace=False)
        await eng.load_adapter("policy-v0", "/dev/shm/x/v0", inplace=False)
        inplace = asyncio.create_task(collect(eng, "a#0", adapter="policy"))
        pinned = asyncio.create_task(collect(eng, "b#0", adapter="policy-v0"))
        await asyncio.sleep(5.5)
        await eng.pause("keep", False)
        await eng.load_adapter("policy", "/dev/shm/x/v1", inplace=True)
        await eng.load_adapter("policy-v1", "/dev/shm/x/v1", inplace=False)
        await eng.resume()
        with pytest.raises(EngineError, match="in use"):
            await eng.unload_adapter("policy-v0")
        await asyncio.gather(inplace, pinned)
        await eng.unload_adapter("policy-v0")
        return eng.truth("a"), eng.truth("b")

    a, b = run_sim(main())
    assert a == [(0, False)] * 6 + [(1, True)] * 14
    assert b == [(0, False)] * 20
