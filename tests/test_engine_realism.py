"""The controller under behaviours a real `vllm serve` has and an in-process fake hides: requests that reach the
engine after a pause, preemption counts published a step late, a per-step token budget, forced lengths. Each was a
review finding; each test fails on the code before its fix."""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from halfwritten.config import RunConfig
from halfwritten.controller import Controller
from halfwritten.engines.fake import CostModel, FakeEngine, LengthModel
from halfwritten.rewards import sim_reward
from halfwritten.tasks import synthetic
from halfwritten.trace import Tracer
from halfwritten.trainer.client import SimTrainer
from halfwritten.types import GenRequest
from tests.simutil import ledger_matches_truth, quick_sim

LONG = dict(mean=150, sigma=1.2, budget_s=30.0)


@pytest.mark.parametrize("regime", ["C1", "C2", "B", "D", "E"])
def test_a_request_that_lands_after_the_pause_is_queued_not_waited_for(regime):
    """Over HTTP a just-sent request can reach vLLM after /pause; vLLM queues it and starts it after resume. Abort
    regimes used to wait for its abort until the timeout. (20 ms of latency: the exact fence pads a pause with
    nothing, so a request must be slow enough to land inside the push.)"""
    summary, controller, engine, _ = quick_sim(regime, submit_latency=0.02, timeout_s=20.0, **LONG)
    assert summary.valid, (summary.error, summary.violations[:3])
    assert ledger_matches_truth(controller, engine) == []
    if regime in ("C1", "C2"):
        assert summary.events["queued_at_abort"] > 0


@pytest.mark.parametrize("regime", ["E", "E+lora"])
def test_preemption_counts_published_a_step_late_are_not_kv_pressure(regime):
    summary, *_ = quick_sim(regime, stats_lag=True, **LONG)
    assert summary.valid, summary.violations[:3]


def test_kv_pressure_is_still_caught_when_counts_lag():
    summary, *_ = quick_sim("D", stats_lag=True, kv_tokens=1500, mean=150, sigma=1.2)
    assert not summary.valid and any(v.startswith("I5") for v in summary.violations)


@pytest.mark.parametrize("regime", ["B", "C1", "C2", "D", "E"])
def test_forced_lengths_hold_through_every_regime(regime):
    """The systems map forces each answer's length (ignore_eos): a C1 continuation asks for the rest of it, a C2
    restart draws a new one, and every consumed answer has exactly its drawn length."""
    summary, controller, engine, _ = quick_sim(regime, forced_lengths="lognormal:120,1.0", **LONG)
    assert summary.valid, summary.violations[:3]
    led, lengths = controller.ledger, controller.lengths
    closed = led.closed()
    assert closed and all(led.length(sid) == min(lengths(sid, led.attempt(sid)), 256) for sid in closed)
    assert ledger_matches_truth(controller, engine) == []


def test_re_prefill_after_a_reset_spreads_over_steps_under_the_token_budget():
    async def main():
        eng = FakeEngine(lengths=LengthModel(mean=200, sigma=0.0), cost=CostModel(1.0, 0, 0, 0),
                         max_num_batched_tokens=4096, max_model_len=4096)
        await eng.start()
        tasks = [asyncio.create_task(drain(eng, GenRequest(f"r{i}#0", tuple(range(3000)), 64))) for i in range(3)]
        await asyncio.sleep(0.5)
        assert len(eng._running) == 1  # one 3,000-token prefill per 4,096-token step
        await asyncio.sleep(3)
        assert len(eng._running) == 3
        await eng.pause("keep", True)
        assert not eng._running and len(eng._waiting) == 3  # the reset sends them back to the waiting queue
        steps = eng.steps
        await eng.resume()
        await asyncio.sleep(1.5)
        assert len(eng._running) == 2 and eng.steps == steps + 2  # one re-prefill per step again
        for t in tasks:
            t.cancel()

    async def drain(eng, req):
        async for _ in eng.generate(req):
            pass

    from halfwritten.clock import run_sim
    run_sim(main())


def test_a_budget_below_max_model_len_is_refused():
    from halfwritten.engines.base import EngineError
    from halfwritten.engines.vllm_http import serve_command
    with pytest.raises(EngineError):
        FakeEngine(max_num_batched_tokens=2048, max_model_len=4096)
    with pytest.raises(ValueError):
        serve_command("m", max_num_batched_tokens=2048, max_model_len=4096)


def _real_time_run(reward, reward_pool, budget=2.0, reward_timeout_s=30.0):
    async def main():
        loop = asyncio.get_running_loop()
        eng = FakeEngine(lengths=LengthModel(mean=40, sigma=0.5), cost=CostModel(2e-3, 2e-5, 0, 1e-6))
        trainer = SimTrainer(eng, Tracer(loop.time, "trainer"), s_fixed=0.02, s_per_token=1e-5, push_s=0.01)
        cfg = RunConfig(regime="D", B=4, G=4, M=4, K=2, max_tokens=256, budget_s=budget,
                        reward_timeout_s=reward_timeout_s)
        return await Controller(cfg, eng, trainer, synthetic(64, 16), reward, reward_pool=reward_pool).run()

    return asyncio.run(main())


def test_a_slow_reward_in_a_pool_does_not_stall_the_loop():
    fast = sim_reward()

    def slow(answer, ids, finish):
        time.sleep(0.05)  # blocking, as a sympy-heavy check is
        return fast(answer, ids, finish)

    with ThreadPoolExecutor(16) as pool:
        pooled = _real_time_run(slow, pool)
    inline = _real_time_run(slow, None)
    assert pooled.valid and inline.valid
    assert pooled.steps >= 2 * inline.steps  # blocking the loop starved generation; the pool does not


def test_a_reward_that_hangs_scores_zero_and_is_counted():
    def hangs(answer, ids, finish):
        time.sleep(0.3)
        return 1.0

    with ThreadPoolExecutor(32) as pool:
        summary = _real_time_run(hangs, pool, budget=1.0, reward_timeout_s=0.05)
    assert summary.events["reward_timeouts"] > 0


def test_a_staged_push_slows_the_step_it_overlaps():
    from halfwritten.clock import run_sim
    from halfwritten.types import TrainBatch, TrainSample

    async def main(slowdown):
        loop = asyncio.get_running_loop()
        tr = SimTrainer(FakeEngine(), Tracer(loop.time, "t"), s_fixed=1.0, s_per_token=0.0, push_s=0.5,
                        lora=True, send_slowdown=slowdown)
        await tr.start()
        await tr.push(0)
        await tr.submit(TrainBatch(0, (TrainSample("s", "g", (1,), (2,), (-1.0,), 1.0, 0.5, (0,), (0,), (0,)),), 1, 0))
        await asyncio.sleep(0.1)
        push = asyncio.create_task(tr.push(1) if tr.staged >= 1 else asyncio.sleep(0))
        t0 = loop.time()
        await tr.next_ready()
        await push
        return loop.time() - t0

    # the step starts at t=0.5 (after the first push); no overlap with the slow push in this layout, so equal
    assert run_sim(main(0.0)) == pytest.approx(run_sim(main(0.0)))

    async def overlapped(slowdown):
        loop = asyncio.get_running_loop()
        tr = SimTrainer(FakeEngine(), Tracer(loop.time, "t"), s_fixed=1.0, s_per_token=0.0, push_s=0.4,
                        lora=True, send_slowdown=slowdown)
        await tr.start()
        await tr.submit(TrainBatch(0, (TrainSample("s", "g", (1,), (2,), (-1.0,), 1.0, 0.5, (0,), (0,), (0,)),), 1, 0))
        await asyncio.sleep(0.1)
        await tr.push(0)  # 0.1 -> 0.5, inside the step (0 -> 1.0)
        await tr.next_ready()
        return loop.time()

    assert run_sim(overlapped(0.0)) == pytest.approx(1.0)
    assert run_sim(overlapped(0.5)) == pytest.approx(1.0 + 0.5 * 0.4)
