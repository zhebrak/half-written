"""The analysis on synthetic rows with known effects: the confirmatory t analysis keeps its error rates at 3 seeds,
endpoints read the curves right, and rows that would quietly bias the result are refused."""
import json
import math

import numpy as np
import pytest
import yaml

from bench.stats import (
    AnalysisError,
    accuracy_at_T,
    analyse,
    at_steps,
    auc,
    gain_threshold,
    holm,
    iqm,
    item_bootstrap,
    load_rows,
    paired_t,
    sign_flip_p,
    time_to,
    tost,
)


def evals(points, end, correct=None, n=400):
    """points: (t, steps, accuracy) marks; end: the same at the budget."""
    out = [{"snapshot": "init", "tag": "init", "t": 0.0, "steps": 0, "accuracy": points[0][2] if points else 0.3,
            "n": n}]
    out += [{"snapshot": f"t{i:02d}-v{s}", "tag": f"t{i:02d}", "t": t, "steps": s, "accuracy": a, "n": n}
            for i, (t, s, a) in enumerate(points, 1)]
    t, s, a = end
    out.append({"snapshot": f"tend-v{s}", "tag": "tend", "t": t, "steps": s, "accuracy": a, "n": n,
                **({"correct": correct} if correct else {})})
    return out


def test_endpoints_from_a_curve():
    ev = evals([(10, 4, 0.5), (20, 8, 0.6)], end=(40.0001, 16, 0.7))  # tend's mark lands just after T
    ev[0]["accuracy"] = 0.4
    assert accuracy_at_T(ev) == 0.7
    expected = ((0.4 + 0.5) / 2 * 10 + (0.5 + 0.6) / 2 * 10 + (0.6 + 0.7) / 2 * 20) / 40
    assert auc(ev, 40) == pytest.approx(expected)  # tend counted, at exactly T
    assert time_to(ev, 0.55, 40) == pytest.approx(15.0) and time_to(ev, 0.9, 40) == math.inf
    assert at_steps(ev, 12) == pytest.approx(0.65) and at_steps(ev, 17) is None


def test_the_threshold_is_a_share_of_As_gain_over_the_warm_start():
    """From a trained warm start a share of A's accuracy can sit below W0's own (S1: W0 0.594, A at T ~0.64, 90% of
    it 0.576), and every arm would reach it at t = 0; a share of the gain cannot."""
    a = [{"arm": "A", "evals": evals([(10, 4, 0.61)], end=(40, 16, 0.64))} for _ in range(2)]
    for r in a:
        r["evals"][0]["accuracy"] = 0.594
    d = {"arm": "D", "evals": evals([(10, 8, 0.62), (20, 16, 0.66)], end=(40, 32, 0.70))}
    d["evals"][0]["accuracy"] = 0.594
    thr = gain_threshold(a + [d], 0.9)
    assert thr == pytest.approx(0.594 + 0.9 * (0.64 - 0.594)) and thr > 0.594
    assert 10 < time_to(d["evals"], thr, 40) < 20 < time_to(a[0]["evals"], thr, 40) <= 40
    assert gain_threshold([d], 0.9) is None  # no A rows, no target
    for r in a:  # A gained 1 point: a target inside the noise is no target
        r["evals"][-1]["accuracy"] = 0.604
    assert gain_threshold(a + [d], 0.9) is None


def test_the_area_can_be_the_primary_endpoint():
    rows = _rows({"A": 0.0, "D": 0.02, "E": 0.02})
    acc = analyse(rows, family=["D-A"], delta=0.02, primary="accuracy_T")
    area = analyse(rows, family=["D-A"], delta=0.02)  # the area is the default (decision 1, owner 2026-10-02)
    assert acc["primary"] == "accuracy_T" and area["primary"] == "auc"
    assert area["family"]["D-A"]["diff"] != acc["family"]["D-A"]["diff"]
    assert "item_level" not in area["family"]["D-A"]["sensitivity"]
    with pytest.raises(AnalysisError, match="primary"):
        analyse(rows, family=["D-A"], delta=0.02, primary="reward")


def test_paired_t_keeps_its_error_rates_at_three_seeds():
    """The reason the confirmatory analysis is t: bootstrap tests reject a true null far too often at n = 3."""
    rng = np.random.default_rng(0)
    false_pos = np.mean([paired_t(rng.normal(0, 1, 3))["p"] < 0.05 for _ in range(2000)])
    assert false_pos < 0.07
    covered = np.mean([(lambda c: c[0] <= 0.5 <= c[1])(paired_t(rng.normal(0.5, 1, 3))["ci"]) for _ in range(2000)])
    assert covered > 0.93
    false_equiv = np.mean([tost(rng.normal(0.02, 0.01, 5), 0.02)["equivalent"] for _ in range(2000)])
    assert false_equiv < 0.07  # a true difference of exactly delta is called equivalent at most ~5% of the time
    assert sign_flip_p([0.01, 0.012, 0.015]) == pytest.approx(0.25)  # its floor at 3 seeds
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": math.nan})
    assert adj["a"] == pytest.approx(0.04) and adj["d"] == 1.0  # a missing contrast still counts in m
    assert iqm([1, 2, 3, 4, 100, 5, 6, 7]) == pytest.approx(np.mean([3, 4, 5, 6]))


def _rows(effects: dict[str, float], seeds=range(6), steps: dict[str, int] | None = None):
    rng = np.random.default_rng(0)
    rows = []
    for seed in seeds:
        base = 0.6 + rng.normal(0, 0.02)
        for arm, effect in effects.items():
            acc = base + effect + rng.normal(0, 0.003)
            n = (steps or {}).get(arm, 16)
            correct = "".join("1" if rng.random() < acc else "0" for _ in range(400))
            rows.append({"row": f"{arm}{seed}", "label": arm, "arm": arm, "regime": arm, "seed": seed,
                         "budget_s": 40.0, "emulated": None,
                         "evals": evals([(20, n // 2, (0.4 + acc) / 2)], end=(40, n, acc), correct=correct)})
    return rows


def test_the_family_finds_a_difference_an_equivalence_and_a_missing_contrast():
    rows = _rows({"A": 0.0, "B": 0.0, "D": 0.06, "E": 0.06})
    out = analyse(rows, family=["D-E", "D-B", "E-B", "best-A", "best-A2"], delta=0.02, best="D", primary="accuracy_T")
    f = out["family"]
    assert f["D-B"]["ci95"][0] > 0.02 and f["D-B"]["p_holm"] < 0.05 and f["D-B"]["verdict"] == "superior"
    assert not f["D-B"]["tost_descriptive"]["equivalent"]
    assert f["D-E"]["tost_descriptive"]["equivalent"] and abs(f["D-E"]["diff"]) < 0.01
    assert f["D-E"]["verdict"] in ("not significant", "superior", "inferior")  # never "equivalent" outside the sequence
    assert "item_level" in f["D-A"]["sensitivity"] and f["D-A"]["sensitivity"]["sign_flip_p"] <= 2 / 2 ** 6 + 1e-9
    assert "missing" in f["D-A2"] and "emulated.json" in str(f["D-A2"]["missing"])
    assert f["D-B"]["p_holm"] >= f["D-B"]["p"] * 1  # Holm over all five, the missing one included
    with pytest.raises(AnalysisError, match="--best"):
        analyse(rows, family=["best-A"], delta=0.02)
    with pytest.raises(AnalysisError, match="no arm 'C1'"):
        analyse(rows, family=["C1-E"], delta=0.02)


def test_h2_compares_at_the_step_count_both_reached():
    """D took twice the steps of A by T; per sample it matches A (same accuracy at A's step count)."""
    rows = []
    for seed in range(5):
        for arm, n in (("A", 10), ("D", 20)):
            # accuracy rises 0.01 per step for both: per sample they are the same policy improvement
            pts = [(t, s, 0.4 + 0.01 * s) for t, s in ((10, n // 4), (20, n // 2), (30, 3 * n // 4))]
            rows.append({"row": f"{arm}{seed}", "label": arm, "arm": arm, "regime": arm, "seed": seed,
                         "budget_s": 40.0, "emulated": None,
                         "evals": evals(pts, end=(40, n, 0.4 + 0.01 * n + 0.001 * seed))})
    for r in rows:
        r["evals"][0]["accuracy"] = 0.4
    out = analyse(rows, family=["D-A"], delta=0.02, primary="accuracy_T")
    assert out["family"]["D-A"]["diff"] == pytest.approx(0.1)  # D is ahead at T (more steps)...
    h2 = out["h2"]["D-A"]
    assert all(v == 10 for v in h2["steps"].values()) and h2["equivalent"]  # ...but not per sample
    # decision 3's steps-domain secondary says the same: D needs as many optimiser steps as A to reach the threshold
    ratio = out["steps_to_threshold"]["D/A"]
    assert ratio["ratio"] == pytest.approx(1.0, abs=0.05) and 0.95 < ratio["ci95"][0] <= ratio["ci95"][1] < 1.05
    assert not ratio["never"]


def test_a2_reads_s1_at_the_steps_a2_reaches_by_T():
    rows = _rows({"A": 0.0, "D": 0.06}, steps={"A": 16, "D": 24})
    for r in rows:
        if r["arm"] == "A":
            # A2 steps take 3.5 s: 11 steps by T = 40, which A's own curve covers (it reached 16)
            r["emulated"] = {"steps": [{"step": k, "A2_cum": 3.5 * (k + 1)} for k in range(16)]}
    out = analyse(rows, family=["D-A2"], delta=0.02, primary="accuracy_T")
    assert "missing" not in out["family"]["D-A2"] and out["arms"]["A2"]["emulated"]
    # A2 has an accuracy at T, not an area: under the default primary it is described, never contrasted
    area = analyse(rows, family=["D-A2"], delta=0.02)
    assert "missing" in area["family"]["D-A2"] and "accuracy_T" in area["arms"]["A2"]
    for r in rows:
        if r["arm"] == "A":
            # emulate replays A's own 16 steps; at 1 s each A2 is through them by t = 16 s, long before T = 40
            r["emulated"] = {"steps": [{"step": k, "A2_cum": 1.0 * (k + 1)} for k in range(16)]}
    out = analyse(rows, family=["D-A2"], delta=0.02, primary="accuracy_T")
    assert "A2 passes A's last step 16 at t = 16 s < T" in str(out["family"]["D-A2"]["missing"])


def _write_row(root, name, regime, seed, valid=True, n=400, tend=True, tag="", host=0.3, fence=0.04, lag=0.04,
               budget_s=40.0, lr=1e-6, split=None, marks=None, identity=None, scorer=None):
    row = root / name
    row.mkdir(parents=True)
    import re
    sizes = dict(zip("BGMKS", map(int, re.search(r"B(\d+)G(\d+)M(\d+)-K(\d+)-S(\d+)", name).groups())))
    (row / "config.yaml").write_text(yaml.safe_dump({"config": {"regime": regime, "seed": seed, "budget_s": budget_s,
                                                                "lora": False, "tag": tag, "train": {"lr": lr},
                                                                **sizes},
                                                     **({"identity": identity} if identity is not None else {})}))
    if marks is not None:  # (tag, staged, written?)
        (row / "snapshots.json").write_text(json.dumps({
            "marks": [{"tag": t, "staged": v} for t, v, _ in marks],
            "written": [[t, f"work/{t}-v{v}", 1.0] for t, v, w in marks if w]}))
    (row / "metrics.json").write_text(json.dumps({"valid": valid, "fence_s": {"p50": fence},
                                                  **({"host": {"main_thread_cpu_frac": host,
                                                               "loop_lag_s": {"p99": lag}}} if host else {})}))
    ev = evals([(20, 4, 0.5)], end=(40, 8, 0.6), n=n)
    if not tend:
        ev = ev[:-1]
    if split:
        ev = [e | {"split": split} for e in ev]
    if scorer:
        ev = [e | {"scorer": "2026-10-02", "math_verify": scorer} for e in ev]
    (row / "evals.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))


def test_rows_that_would_bias_the_analysis_are_refused(tmp_path):
    _write_row(tmp_path / "a", "D-B8G8M8-K2-S1-s0", "D", 0)
    _write_row(tmp_path / "a", "D-B8G8M8-K2-S1-s1", "D", 1, valid=False)
    rows, excluded = load_rows(tmp_path / "a")
    assert [r["seed"] for r in rows] == [0] and excluded[0]["reason"] == "invalid: not valid"
    _write_row(tmp_path / "b", "x/D-B8G8M8-K2-S1-s0", "D", 0)
    _write_row(tmp_path / "b", "y/D-B8G8M8-K2-S1-s0", "D", 0)
    with pytest.raises(AnalysisError, match="two rows of arm D seed 0"):
        load_rows(tmp_path / "b")
    _write_row(tmp_path / "c", "D-B8G8M8-K2-S1-s0", "D", 0)
    _write_row(tmp_path / "c", "D-B8G8M16-K2-S1-s1", "D", 1)
    with pytest.raises(AnalysisError, match="differ in M"):
        load_rows(tmp_path / "c")
    _write_row(tmp_path / "d", "D-B8G8M8-K2-S1-s0", "D", 0)
    _write_row(tmp_path / "d", "E-B8G8M8-K2-S1-s0", "E", 0, n=50)
    with pytest.raises(AnalysisError, match="item count"):
        load_rows(tmp_path / "d")
    _write_row(tmp_path / "e", "D-B8G8M8-K2-S1-s0", "D", 0, tend=False)
    with pytest.raises(AnalysisError, match="tend"):
        load_rows(tmp_path / "e")
    # rows whose timing measured the harness (review 8): a saturated controller, a late event loop, a slow fence, or
    # no record of either. A busy controller that did no harm (2026-10-02's W0 rows: 0.55 of a core, loop p99 51 ms)
    # stays in
    _write_row(tmp_path / "f", "D-B8G8M8-K2-S1-s0", "D", 0, host=0.55, lag=0.051)
    _write_row(tmp_path / "f", "D-B8G8M8-K2-S1-s1", "D", 1, host=0.9)
    _write_row(tmp_path / "f", "D-B8G8M8-K2-S1-s2", "D", 2, fence=0.47)
    _write_row(tmp_path / "f", "D-B8G8M8-K2-S1-s3", "D", 3, host=None)
    _write_row(tmp_path / "f", "D-B8G8M8-K2-S1-s4", "D", 4, lag=0.3)
    rows, excluded = load_rows(tmp_path / "f")
    assert [r["seed"] for r in rows] == [0]
    assert [e["reason"].removeprefix("harness: ").split(" ")[0] for e in excluded] == ["controller", "fence", "no",
                                                                                         "event"]
    assert len(load_rows(tmp_path / "f", harness=False)[0]) == 5
    # the amendment's limits (decision 7, fixed at the tag) replace the defaults: a wider fence limit keeps seed 2
    rows, excluded = load_rows(tmp_path / "f", limits={"max_fence_p50_s": 0.5})
    assert [r["seed"] for r in rows] == [0, 2] and all(r["task"] is None for r in rows)
    # a MATH scorer that timed out labelled answers 0 unscored: the row is flagged by the same rule
    _write_row(tmp_path / "g", "D-B8G8M8-K2-S1-s0", "D", 0)
    m = tmp_path / "g" / "D-B8G8M8-K2-S1-s0" / "metrics.json"
    m.write_text(json.dumps(json.loads(m.read_text()) | {"events": {"reward_timeouts": 3}}))
    assert load_rows(tmp_path / "g")[1][0]["reason"] == "harness: 3 reward timeout(s): answers scored 0 unscored"


def test_item_bootstrap_on_identical_arms_centres_on_zero():
    rng = np.random.default_rng(3)
    c = {s: "".join("1" if rng.random() < 0.6 else "0" for _ in range(300)) for s in range(4)}
    est, lo, hi = item_bootstrap(c, c, n_boot=500)
    assert est == 0 and lo == 0 and hi == 0



def test_the_rows_compared_must_be_one_experiment(tmp_path):
    """External review F03: two seeds of one arm at budgets 10/20 s and learning rates 0.001/0.004 entered one
    comparison. Across every row compared, only the regime and its implications, the seed and ports may differ: a
    per-arm budget (A run longer) or another learning rate is refused, and so is another warm start or filter."""
    _write_row(tmp_path / "a", "D-B8G8M8-K2-S1-s0", "D", 0)
    _write_row(tmp_path / "a", "E-B8G8M8-K2-S1-s0", "E", 0, lr=4e-6)
    with pytest.raises(AnalysisError, match="train.lr"):
        load_rows(tmp_path / "a")
    _write_row(tmp_path / "b", "D-B8G8M8-K2-S1-s0", "D", 0)
    _write_row(tmp_path / "b", "A-B8G8M8-K0-S1-s0", "A", 0, budget_s=80.0)
    with pytest.raises(AnalysisError, match="budget_s"):
        load_rows(tmp_path / "b")
    _write_row(tmp_path / "c", "D-B8G8M8-K2-S1-s0", "D", 0, identity={"init_from_sha256": "w0"})
    _write_row(tmp_path / "c", "E-B8G8M8-K2-S1-s0", "E", 0, identity={"init_from_sha256": "another"})
    with pytest.raises(AnalysisError, match="warm start"):
        load_rows(tmp_path / "c")


def test_the_analysis_reads_one_split_and_complete_curves(tmp_path):
    """Decisions are read on dev and the confirmatory grid on test; a row's curve must have every scheduled mark, from
    the version staged at it, and is never bridged across a gap (methodologist, review of 2026-10-02)."""
    _write_row(tmp_path / "s", "D-B8G8M8-K2-S1-s0", "D", 0, split="dev")
    rows, excluded = load_rows(tmp_path / "s")
    assert rows == [] and excluded[0]["reason"] == "no evaluation on test"
    assert len(load_rows(tmp_path / "s", split="dev")[0]) == 1
    ok = [("t01", 4, True), ("tend", 8, True)]
    _write_row(tmp_path / "m", "D-B8G8M8-K2-S1-s0", "D", 0, marks=ok)
    _write_row(tmp_path / "m", "D-B8G8M8-K2-S1-s1", "D", 1, marks=[("t01", 4, True), ("t02", 6, False),
                                                                   ("tend", 8, True)])
    _write_row(tmp_path / "m", "D-B8G8M8-K2-S1-s2", "D", 2, marks=[("t01", 3, True), ("tend", 8, True)])
    rows, excluded = load_rows(tmp_path / "m")
    assert [r["seed"] for r in rows] == [0]
    assert "no snapshot at mark t02" in excluded[0]["reason"] and "not the version staged" in excluded[1]["reason"]
    _write_row(tmp_path / "e", "D-B8G8M8-K2-S1-s0", "D", 0, marks=[("t01", 4, True), ("t02", 6, True),
                                                                   ("tend", 8, True)])
    with pytest.raises(AnalysisError, match="t02 has no evaluation"):
        load_rows(tmp_path / "e")


def test_verdicts_separate_equivalent_from_inconclusive():
    from bench.stats import verdict
    assert verdict(0.03, [0.025, 0.035], 0.01, delta=0.02) == "superior"
    assert verdict(-0.03, [-0.035, -0.025], 0.01, delta=0.02) == "inferior"
    assert verdict(0.005, [0.002, 0.008], 0.01, delta=0.02) == "different but within delta"
    assert verdict(0.0, [-0.01, 0.01], 0.6, delta=0.02) == "equivalent"
    assert verdict(0.0, [-0.03, 0.03], 0.9, delta=0.02) == "inconclusive"


def test_equivalence_is_claimed_in_a_fixed_sequence_and_h4_needs_stale_kv():
    """The equivalence claims are the pre-registered ones in order (H4's D-E, then H2's B-A, D-A, E-A), each at the
    full alpha until the first that fails; D-E enters only if D generated enough over stale KV (methodologist)."""
    rows = _rows({"A": 0.0, "B": 0.0, "D": 0.0, "E": 0.0}, steps={"A": 16, "B": 16, "D": 16, "E": 16})
    for r in rows:
        r["stale_kv_share"] = 0.13 if r["arm"] == "D" else 0.0
    out = analyse(rows, family=["D-E"], delta=0.02)
    assert [c["result"] for c in out["equivalence"]] == ["equivalent"] * 4
    for r in rows:
        r["stale_kv_share"] = 0.01 if r["arm"] == "D" else 0.0
    out = analyse(rows, family=["D-E"], delta=0.02)
    assert out["equivalence"][0]["result"].startswith("uninformative")
    assert [c["result"] for c in out["equivalence"][1:]] == ["equivalent"] * 3
    # membership fixed at the tag overrides the rows' own exposure (decision 4)
    assert analyse(rows, family=["D-E"], delta=0.02, h4="in")["equivalence"][0]["result"] == "equivalent"
    assert analyse(rows, family=["D-E"], delta=0.02, h4="out", min_exposure=0.0)["equivalence"][0][
        "result"].startswith("uninformative")
    rows = _rows({"A": 0.0, "B": -0.06, "D": 0.0, "E": 0.0}, steps={"A": 16, "B": 16, "D": 16, "E": 16})
    for r in rows:
        r["stale_kv_share"] = 0.13 if r["arm"] == "D" else 0.0
    out = analyse(rows, family=["D-E"], delta=0.02)
    assert [c["result"] for c in out["equivalence"]] == ["equivalent", "not shown equivalent",
                                                         "not tested: the sequence stopped earlier",
                                                         "not tested: the sequence stopped earlier"]
    # the five verdicts live in the sequence only; six seeds with little noise can make a 0.3-point gap significant
    # yet inside the margin, and the claim says so
    assert out["equivalence"][0]["verdict"] in ("equivalent", "different but within delta")
    assert out["family"]["D-E"]["verdict"] in ("not significant", "superior", "inferior")


def test_h4_and_h2_are_separate_families_so_neither_blocks_the_other():
    """Decision 12 (owner, 2026-10-02): H4's D-E alone, H2's B-A, D-A, E-A as their own sequence. A D-E that fails
    leaves H2 tested; within H2 the sequence still stops at its first failure."""
    rows = _rows({"A": 0.0, "B": 0.0, "D": 0.0, "E": -0.06}, steps={"A": 16, "B": 16, "D": 16, "E": 16})
    for r in rows:
        r["stale_kv_share"] = 0.13 if r["arm"] == "D" else 0.0
    out = analyse(rows, family=["D-E"], delta=0.02)
    claims = {(c["family"], c["contrast"]): c["result"] for c in out["equivalence"]}
    assert claims == {(1, "D-E"): "not shown equivalent", (2, "B-A"): "equivalent", (2, "D-A"): "equivalent",
                      (2, "E-A"): "not shown equivalent"}
    # one flat sequence (the earlier draft): the same D-E failure stops everything after it
    flat = analyse(rows, family=["D-E"], delta=0.02, equivalence=["D-E", "B-A", "D-A", "E-A"])
    assert [c["result"] for c in flat["equivalence"]][1:] == ["not tested: the sequence stopped earlier"] * 3


def test_rows_set_aside_are_not_read(tmp_path):
    """A row moved aside (.invalid-, .legacy-, .signalled-) is out of its label's way: beside its rerun it must not
    count as a second row of the same arm and seed (code review of 2026-10-02)."""
    _write_row(tmp_path, "D-B8G8M8-K2-S1-s0", "D", 0)
    _write_row(tmp_path, "D-B8G8M8-K2-S1-s0.legacy-1790000000123", "D", 0)
    _write_row(tmp_path, "D-B8G8M8-K2-S1-s0.invalid-1790000000456", "D", 0, valid=False)
    rows, excluded = load_rows(tmp_path)
    assert [r["row"].rsplit("/", 1)[1] for r in rows] == ["D-B8G8M8-K2-S1-s0"] and excluded == []


def test_the_seed_count_corrects_dev_item_noise_to_test_size():
    """Decision 13 on exact cases. Two runs at equal accuracy that disagree on half of 4 items differ only by item
    noise: a pair's item variance is discordance / n = 0.125 (a pair's, not one run's), there is no seed part, and on
    a test set of n items it scales by n_dev / n. A real spread between runs adds a seed part, raised to its upper
    bound; more test items shrink only the item part (code review of 2026-10-02)."""
    from bench.stats import seed_count
    a = seed_count(["1100", "1010"], n_test=4, delta=1.0)
    assert a["v_item_dev"] == pytest.approx(0.125) and a["seed_part"] == 0.0
    assert a["sigma_diff_test"] == pytest.approx(0.125 ** 0.5)
    assert seed_count(["1100", "1010"], n_test=400, delta=1.0)["sigma_diff_test"] == pytest.approx((0.125 / 100) ** 0.5)
    b = seed_count(["1100", "1110", "1111", "1000"], n_test=4, delta=1.0)
    assert b["seed_part"] > 0 and b["seed_part_ucb"] > b["seed_part"]
    assert b["sigma_diff_test"] ** 2 == pytest.approx(b["seed_part_ucb"] + b["v_item_dev"])
    assert seed_count(["1100", "1010"], n_test=4, delta=0.05)["seeds"] is None  # beyond 50 seeds: say so
    with pytest.raises(AnalysisError):
        seed_count(["1100"], n_test=4, delta=1.0)
    # the cap (decision 13): past it the arm runs the cap, and its equivalence claims become intervals only
    capped = seed_count(["1100", "1110", "1111", "1000"], n_test=4, delta=1.0, cap=1)
    assert capped["run"] == 1 and capped["intervals_only"]
    free = seed_count(["1100", "1110", "1111", "1000"], n_test=4, delta=1.0, cap=50)
    assert free["run"] == free["seeds"] and not free["intervals_only"]


def test_one_scorer_across_the_rows_compared(tmp_path):
    """math-verify is MATH's reward and its scorer: evaluations scored by two versions are not one experiment (code
    review of 2026-10-02)."""
    _write_row(tmp_path, "D-B8G8M8-K2-S1-s0", "D", 0, scorer="0.9.0")
    _write_row(tmp_path, "E-B8G8M8-K2-S1-s0", "E", 0, scorer="0.10.0")
    with pytest.raises(AnalysisError, match="different scorers"):
        load_rows(tmp_path)


def test_the_analysis_matches_its_golden_output():
    """The pre-registered analysis, pinned before any refactor: the full output of analyse() on a synthetic tree
    (tests/fixtures/golden_analysis.json; HW_UPDATE_GOLDEN=1 regenerates it, with the diff reviewed)."""
    from tests.goldens import check_golden
    rows = _rows({"A": 0.0, "B": 0.0, "D": 0.06, "E": 0.06}, steps={"A": 16, "B": 16, "D": 24, "E": 24})
    for r in rows:
        r["stale_kv_share"] = 0.13 if r["arm"] == "D" else 0.0
    out = analyse(rows, family=["D-E", "D-B", "E-B", "best-A"], delta=0.02, best="D")
    check_golden("golden_analysis.json", out)


def test_the_command_reads_delta_best_and_h4_from_the_amendment(tmp_path, capsys):
    from bench.stats import main
    for arm in ("A", "B", "D", "E"):
        for seed in (0, 1):
            _write_row(tmp_path / "rows", f"{arm}-B8G8M8-K2-S1-s{seed}", arm, seed, split="test")
    (tmp_path / "amend.yaml").write_text(yaml.safe_dump({"delta": 0.02, "best_async": "D", "h4": "out"}))
    main([str(tmp_path / "rows"), "--amendment", str(tmp_path / "amend.yaml"), "--out", str(tmp_path / "a.json")])
    out = json.loads((tmp_path / "a.json").read_text())
    assert out["delta"] == 0.02 and out["best"] == "D"
    assert out["equivalence"][0]["result"].startswith("uninformative")  # h4 out, from the amendment


def test_intervals_only_reports_every_contrast_without_a_verdict():
    """Decision 13: past the seed cap the analysis reports intervals and claims nothing, in both families."""
    rows = _rows({"A": 0.0, "B": -0.06, "D": 0.0, "E": 0.0}, steps={"A": 16, "B": 16, "D": 16, "E": 16})
    for r in rows:
        r["stale_kv_share"] = 0.13 if r["arm"] == "D" else 0.0
    out = analyse(rows, family=["D-E", "B-A"], delta=0.02, intervals_only=True)
    assert out["intervals_only"] is True
    assert {t["verdict"] for t in out["family"].values()} == {"interval only"}
    assert all("ci95" in t and "p_holm" in t for t in out["family"].values())
    assert [c["result"] for c in out["equivalence"]] == ["interval only"] * 4  # B-A's failure stops no sequence
    assert all("ci90" in c and "verdict" not in c for c in out["equivalence"])


def test_the_command_reads_equivalence_and_intervals_only_from_the_amendment(tmp_path):
    from bench.stats import main
    for arm in ("A", "B", "D", "E"):
        for seed in (0, 1):
            _write_row(tmp_path / "rows", f"{arm}-B8G8M8-K2-S1-s{seed}", arm, seed, split="test")
    (tmp_path / "amend.yaml").write_text(yaml.safe_dump({"delta": 0.03, "best_async": "D", "h4": "in",
                                                          "equivalence": "D-E", "intervals_only": True}))
    main([str(tmp_path / "rows"), "--amendment", str(tmp_path / "amend.yaml"), "--out", str(tmp_path / "a.json")])
    out = json.loads((tmp_path / "a.json").read_text())
    assert out["intervals_only"] is True and [c["contrast"] for c in out["equivalence"]] == ["D-E"]
    assert out["equivalence"][0]["result"] == "interval only"
    assert {t["verdict"] for t in out["family"].values() if "verdict" in t} == {"interval only"}
    # the command line wins over the amendment's equivalence key
    main([str(tmp_path / "rows"), "--amendment", str(tmp_path / "amend.yaml"), "--equivalence", "B-A,D-A",
          "--out", str(tmp_path / "b.json")])
    assert [c["contrast"] for c in json.loads((tmp_path / "b.json").read_text())["equivalence"]] == ["B-A", "D-A"]
