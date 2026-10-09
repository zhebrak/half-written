"""The controller's certified behaviour, pinned: every regime under every delivery mode of the fake engine, digested
(ledger, syncs, steps, events, violations, token counters) against tests/fixtures/golden_sim.json. A refactor of the
sync path must leave this unchanged unless it declares an output change, in which case the fixture is regenerated
(HW_UPDATE_GOLDEN=1) and the diff reviewed (the review-2026-10-02 plan, section 1.0)."""
import pytest

from tests.goldens import canonical, check_digests
from tests.simutil import quick_sim

REGIMES = ["A", "B", "C1", "C2", "D", "E", "F", "B+lora", "D+lora", "E+lora"]
DELIVERIES = ["immediate", "one_step_late", "bursty"]
# vLLM 0.30's shape at the pause boundary, which the plain deliveries never reach: tokens still arriving after the
# pause is acknowledged (stats_lag, async_scheduling), a submit latency, aborts that end a waiting request without a
# finish reason; the pausing arms only, since A and F never cut. The queued-at-abort and finishless-abort branches
# stay at zero even here: the controller's admission gate keeps M*G under max_num_seqs, so nothing queues in the
# fake; vLLM reaches them through its own scheduling races, which the H100 certify rows exercised.
VLLM_SHAPED = dict(submit_latency=0.02, stats_lag=True, async_scheduling=True, silent_waiting_aborts=True, M=8,
                   budget_s=15.0)
PAUSING = ["B", "C1", "C2", "D", "E"]


def digest(summary, controller) -> dict:
    """Everything a sync or a cut decides, in one JSON-able record."""
    return canonical({
        "valid": summary.valid, "error": summary.error, "violations": summary.violations,
        "steps": summary.steps, "groups_submitted": summary.groups_submitted, "samples": summary.samples,
        "tokens": summary.tokens, "late_old_tokens": summary.late_old_tokens,
        "versions_skipped": summary.versions_skipped, "events": summary.events,
        "syncs": summary.syncs, "steps_log": summary.steps_log,
        "ledger": controller.ledger.table(), "discards": controller.ledger.discards_table(),
    })


@pytest.fixture(scope="module")
def goldens():
    out = {}
    for regime in REGIMES:
        for delivery in DELIVERIES:
            summary, controller, _, _ = quick_sim(regime, delivery=delivery)
            out[f"{regime}/{delivery}"] = digest(summary, controller)
    for regime in PAUSING:
        for delivery in ("one_step_late", "bursty"):
            summary, controller, _, _ = quick_sim(regime, delivery=delivery, **VLLM_SHAPED)
            out[f"{regime}/{delivery}/vllm"] = digest(summary, controller)
    return out


def test_the_vllm_shaped_cases_have_tokens_arriving_after_the_acknowledgement(goldens):
    """The column exists to pin what the plain deliveries never exercise: under the keep pauses (D, E) old-version
    tokens land after the pause's acknowledgement, so the cut is decided by the fence, not by the ack. The drain
    (B) and the aborts (C1, C2) have none by design: their pauses end or discard every running answer."""
    shaped = {k: d for k, d in goldens.items() if k.endswith("/vllm")}
    assert len(shaped) == len(PAUSING) * 2
    for key, d in shaped.items():
        assert all(s["fence_s"] > 0 for s in d["syncs"]), key
        assert (d["late_old_tokens"] > 0) == key.startswith(("D/", "E/")), key


def test_every_regime_and_delivery_matches_the_golden_digest(goldens):
    check_digests("golden_sim.json", goldens, summary=lambda d: {
        "steps": d["steps"], "syncs": len(d["syncs"]), "delivered": d["tokens"]["delivered"],
        "consumed": d["tokens"]["consumed"], "closed": len(d["ledger"]), "violations": len(d["violations"])})


def test_the_golden_runs_are_valid(goldens):
    for key, d in goldens.items():
        assert d["valid"] and d["violations"] == [] and d["error"] is None, key
