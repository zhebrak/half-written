"""The small shared helpers that replaced repeated code (review-2026-10-02 plan, section 2.8): snapshot names, arm
names, the budget record, the first crossing of a curve, and collecting an answer from an engine."""
import math

import pytest

from halfwritten.regimes import parse_arm, resolve
from halfwritten.types import BUDGET_MARK, GenRequest, parse_snapshot, snapshot_name


def test_snapshot_names_round_trip():
    assert snapshot_name("t03", 17) == "t03-v17"
    assert parse_snapshot("t03-v17") == ("t03", 17)
    assert parse_snapshot(snapshot_name(BUDGET_MARK, 290)) == (BUDGET_MARK, 290)
    assert parse_snapshot("init") == ("init", 0)
    assert parse_snapshot("step0050-v50") == ("step0050", 50)
    assert parse_snapshot("odd-name") == ("odd-name", None) and parse_snapshot("t-vx") == ("t-vx", None)


def test_arm_names_resolve_to_a_regime_and_its_weights():
    assert parse_arm("D") == ("D", False) and parse_arm("D+lora") == ("D", True) and parse_arm("F") == ("F", False)
    assert resolve("D+lora").weights == "lora_inplace" and resolve("D").weights == "full"
    assert resolve("F", lora=True).weights == "lora_versioned"
    with pytest.raises(ValueError, match="lora=True"):
        resolve("F")  # F's name alone does not make a config a LoRA one
    with pytest.raises(ValueError, match="unknown regime"):
        resolve("Z")


def test_the_budget_record_and_the_first_crossing():
    from bench.stats import _first_crossing, at_budget, final_steps, steps_to, time_to
    evals = [{"tag": "init", "snapshot": "init", "t": 0.0, "accuracy": 0.4},
             {"tag": "t01", "snapshot": "t01-v10", "t": 10.0, "accuracy": 0.5},
             {"tag": BUDGET_MARK, "snapshot": "tend-v20", "t": 20.0, "accuracy": 0.6}]
    assert at_budget(evals)["snapshot"] == "tend-v20" and at_budget(evals[:2]) is None
    assert final_steps(evals) == 20.0
    assert _first_crossing([(0, 0.4), (10, 0.5), (20, 0.6)], 0.55) == pytest.approx(15.0)
    assert _first_crossing([(0, 0.4), (10, 0.5)], 0.9) == math.inf and _first_crossing([], 0.1) == math.inf
    assert time_to(evals, 0.55, T=20.0) == pytest.approx(15.0) and steps_to(evals, 0.55) == pytest.approx(15.0)


def test_collect_streams_a_whole_answer_from_an_engine():
    from halfwritten.clock import run_sim
    from halfwritten.engines.base import collect
    from halfwritten.engines.fake import FakeEngine
    from halfwritten.lengths import LengthModel

    async def main():
        engine = FakeEngine(lengths=LengthModel(mean=12, sigma=0.0), max_num_seqs=4)
        await engine.start()
        seen = []
        got = await collect(engine, GenRequest("c-0#0", (1, 2, 3), 32), on_chunk=seen.append)
        await engine.close()
        return got, seen

    got, seen = run_sim(main())
    assert got.finish is not None and 1 <= len(got.tokens) <= 32
    assert sum(len(c.token_ids) for c in seen) == len(got.tokens)
