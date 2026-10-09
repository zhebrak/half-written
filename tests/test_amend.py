"""The amendment (bench.amend): T from the pooled D curves, the seeds at the snapshot A reaches by T, the best async
arm from the systems map's W0 cell, H4's membership, the rules' recorded fallbacks, and the stops left."""
import json
import math

import numpy as np
import pytest
import yaml

from bench.amend import AmendmentStop, amend, fit_gain, main, t_fraction


def _row(root, label, regime, tag, *, budget_s=600.0, steps_per_hour=1700.0, max_steps=None, task="gsm8k",
         evals=None, commit="abc1234"):
    row = root / label
    row.mkdir(parents=True)
    (row / "config.yaml").write_text(yaml.safe_dump({
        "config": {"regime": regime, "tag": tag, "task": task, "budget_s": budget_s, "max_steps": max_steps,
                   "seed": 0, "B": 32, "G": 8, "M": 32, "K": 4, "S": 1},
        "environment": {"commit": commit},
        "identity": {"init_from_sha256": "w0" * 32, "prompt_filter_sha256": "f" * 64, "dev_sha256": "d" * 64}}))
    (row / "metrics.json").write_text(json.dumps({"valid": True, "steps_per_hour": steps_per_hour,
                                                  "host": {"main_thread_cpu_frac": 0.4}}))
    if evals is not None:
        (row / "evals.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evals))
    return row


def _curve(start, gain, tau, budget_s, n=16, split="dev", noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = [{"tag": "init", "snapshot": "init", "t": 0.0, "accuracy": start, "split": split, "n": 500}]
    for k in range(1, n):
        t = budget_s * k / n
        out.append({"tag": f"t{k:02d}", "snapshot": f"t{k:02d}-v{k * 10}", "t": t, "split": split, "n": 500,
                    "accuracy": start + gain * (1 - math.exp(-t / tau)) + rng.normal(0, noise)})
    out.append({"tag": "tend", "snapshot": f"tend-v{n * 10}", "t": budget_s, "split": split, "n": 500,
                "accuracy": start + gain * (1 - math.exp(-budget_s / tau))})
    return out


def _one_gpu(root, k=6, max_steps=150, acc=0.65):
    rng = np.random.default_rng(1)
    for s in range(k):
        evals = [{"tag": "init", "snapshot": "init", "t": 0.0, "accuracy": 0.59, "split": "dev", "n": 500}]
        for step in range(10, max_steps + 1, 10):
            p = acc + 0.004 * s + 0.0003 * step
            correct = "".join("1" if rng.random() < p else "0" for _ in range(500))
            evals.append({"tag": f"step{step:04d}", "snapshot": f"step{step:04d}-v{step}", "steps": step,
                          "accuracy": correct.count("1") / 500, "correct": correct, "split": "dev", "n": 500})
        _row(root, f"A-B32G8M32-K0-S1-pilot-1gpu-s{100 + s}", "A", "pilot-1gpu", budget_s=3600.0,
             steps_per_hour=570.0, max_steps=max_steps, evals=evals)


def _pilot(root, tau=180.0, gain=0.12, budget_s=600.0, a_rate=864.0):
    for s in (100, 101):
        _row(root, f"D-B32G8M32-K4-S1-pilot-s{s}", "D", "pilot", budget_s=budget_s,
             evals=_curve(0.59, gain, tau, budget_s, noise=0.004, seed=s))
    _row(root, "A-B32G8M32-K0-S1-pilot-s100", "A", "pilot", budget_s=budget_s, steps_per_hour=a_rate,
         evals=_curve(0.59, 0.09, 420.0, budget_s))


def _map(root, rates=(("B", 1420.0), ("D", 1740.0), ("E", 1608.0)), tag="gsm8k-w0"):
    for regime, rate in rates:
        _row(root, f"{regime}-B32G8M32-K4-S1-{tag}-s0", regime, tag, task="sim", budget_s=300.0,
             steps_per_hour=rate)


def test_the_fit_recovers_the_curve_and_t80():
    pts = [(t, 0.59 + 0.12 * (1 - math.exp(-t / 180.0))) for t in range(0, 601, 30)]
    fit = fit_gain(pts, 0.59)
    assert fit["gain"] == pytest.approx(0.12, abs=1e-3) and fit["tau_s"] == pytest.approx(180.0, rel=0.02)
    assert fit["t80_s"] == pytest.approx(180.0 * math.log(5), rel=0.02) and fit["rmse"] < 1e-3
    assert t_fraction({"tau_s": 100.0}, 0.5) == pytest.approx(100.0 * math.log(2))
    with pytest.raises(AmendmentStop, match="too few"):
        fit_gain(pts[:2], 0.59)


def test_the_amendment_applies_every_rule(tmp_path):
    _pilot(tmp_path / "gsm8k")
    _one_gpu(tmp_path / "one")
    _map(tmp_path / "sim")
    out = amend("gsm8k", tmp_path / "gsm8k", tmp_path / "one", tmp_path / "sim", delta=0.02, n_test=1319)
    assert out["T_s"] == 300.0 and out["snapshot_every_s"] == 18.75  # t80 = 180 ln 5 = 290 s, rounded up
    assert out["fit"]["points"] == 34 and 0.11 < out["fit"]["gain"] < 0.13
    assert out["step_T"] == pytest.approx(72.0) and set(out["seed_snapshot_steps"]) == {70}
    assert out["seeds"]["C1"] == 2 and out["seeds"]["A"] == out["seeds"]["D"] == out["seed_count"]["run"] >= 2
    assert out["seed_count"]["run"] <= 12
    assert out["best_async"] == "D" and out["map_steps_per_hour"]["E"] == 1608.0
    assert out["h4"] == "out" and out["delta"] == 0.02 and out["identity"]["dev_sha256"] == "d" * 64
    assert out["commit"] == "abc1234" and out["planned_rows"] == sum(out["seeds"].values())
    assert len(out["from_rows"]) == 3 + 6 + 3
    assert out["equivalence"] == "D-E;B-A,D-A,E-A" and out["intervals_only"] is False
    assert out["snapshots"] == 16 and out["time_cap"] is None
    assert set(out["harness"]) >= {"D-B32G8M32-K4-S1-pilot-s100", "A-B32G8M32-K0-S1-pilot-s100"}
    # a time-bounded run (2026-10-03): T/8 marks, and a cap in box-hours: the rows that fit after a reserve set the
    # seeds per arm (verdicts stand at a fixed n; intervals only is decision 13's cap alone), never under two
    capped = amend("gsm8k", tmp_path / "gsm8k", tmp_path / "one", tmp_path / "sim", delta=0.02, n_test=1319,
                   snapshots=8, time_cap_h=3.0, row_overhead_s=360.0, reserve_rows=2)
    assert capped["snapshot_every_s"] == 37.5 and capped["snapshots"] == 8
    fit_rows = int(3 * 3600 // (300 + 360)) - 2  # 14 rows: C1's two, then three seeds an arm
    assert capped["time_cap"]["rows_that_fit"] == fit_rows and capped["time_cap"]["seeds_run"] == 3
    assert capped["time_cap"]["seeds_by_rule"] == out["seed_count"]["run"]
    assert 0 <= capped["time_cap"]["tost_power_at_seeds_run"] <= 1
    assert capped["seeds"] == {"A": 3, "B": 3, "D": 3, "E": 3, "C1": 2} and capped["intervals_only"] is False
    assert any("time cap 3.0 h" in w and "verdicts stand" in w for w in capped["warnings"])
    with pytest.raises(AmendmentStop, match="fewer than two seeds"):
        amend("gsm8k", tmp_path / "gsm8k", tmp_path / "one", tmp_path / "sim", delta=0.02, n_test=1319,
              time_cap_h=1.0, row_overhead_s=360.0)
    roomy = amend("gsm8k", tmp_path / "gsm8k", tmp_path / "one", tmp_path / "sim", delta=0.02, n_test=1319,
                  time_cap_h=100.0)
    assert roomy["seeds"]["A"] == 12 and roomy["intervals_only"] is False  # the window is used, up to the cap
    # without the one-GPU stage the cap alone sets the seeds (recorded as such); without a cap it is a stop
    alone = amend("gsm8k", tmp_path / "gsm8k", tmp_path / "none", tmp_path / "sim", delta=0.02, n_test=1319,
                  time_cap_h=3.0, row_overhead_s=360.0)
    assert alone["seed_count"] is None and alone["seeds"]["D"] == 3 and alone["time_cap"]["seeds_by_rule"] is None
    assert any("decision 6's seed rule is not applied" in w for w in alone["warnings"])
    with pytest.raises(AmendmentStop, match="two or more"):
        amend("gsm8k", tmp_path / "gsm8k", tmp_path / "none", tmp_path / "sim", delta=0.02, n_test=1319)


def test_math_under_a_time_cap_gives_every_arm_the_caps_seeds_and_needs_no_pilot_a_row(tmp_path):
    """The 2026-10-03 MATH chain: D x 2 and E in the pilot, no A row and no one-GPU rows; under the cap A, B, D and E
    take the seeds that fit (decision 12's larger A and B counts are a budget rule, and the cap is the budget)."""
    _pilot(tmp_path / "math", tau=600.0, budget_s=2400.0)
    import shutil
    shutil.rmtree(tmp_path / "math" / "A-B32G8M32-K0-S1-pilot-s100")
    _map(tmp_path / "sim", tag="math-w0")
    with pytest.raises(AmendmentStop, match="0 A rows"):
        amend("math", tmp_path / "math", tmp_path / "none", tmp_path / "sim", delta=0.03, n_test=500)
    out = amend("math", tmp_path / "math", tmp_path / "none", tmp_path / "sim", delta=0.03, n_test=500,
                snapshots=8, time_cap_h=9.0, row_overhead_s=900.0, reserve_rows=1)
    rows = int(9 * 3600 // (out["T_s"] + 900)) - 1
    assert out["seeds"] == {arm: min(8, rows // 4) for arm in ("A", "B", "D", "E")} and out["step_T"] is None
    assert out["a_steps_per_hour"] is None and out["seed_count"] is None and out["snapshot_every_s"] == out["T_s"] / 8
    assert any("no pilot A row" in w for w in out["warnings"]) and out["time_cap"]["seeds_by_rule"] is None


def test_math_gives_a_and_b_five_seeds_only_under_a_short_T(tmp_path):
    _pilot(tmp_path / "math", tau=600.0, budget_s=2400.0)  # t80 = 966 s: T = 17 min
    _one_gpu(tmp_path / "one", max_steps=400)
    _map(tmp_path / "sim", tag="math-w0")
    short = amend("math", tmp_path / "math", tmp_path / "one", tmp_path / "sim", delta=0.03, n_test=500)
    assert 900 <= short["T_s"] <= 1080 and short["T_s"] % 60 == 0  # t80 = 600 ln 5 = 966 s, as fitted on noise
    assert short["seeds"]["A"] == short["seeds"]["B"] == 5
    _pilot(tmp_path / "math2", tau=1500.0, budget_s=6000.0)  # t80 = 2414 s: T = 41 min
    _one_gpu(tmp_path / "one2", max_steps=1000)
    long = amend("math", tmp_path / "math2", tmp_path / "one2", tmp_path / "sim", delta=0.03, n_test=500)
    assert 2280 <= long["T_s"] <= 2580 and long["seeds"]["A"] == long["seeds"]["B"] == 3  # t80 = 2414 s
    assert short["h4"] == "out"  # no stale-KV share recorded in these synthetic rows: out
    # decision 12: past 30 minutes D-E is MATH's only equivalence claim; under it, both families
    assert short["equivalence"] == "D-E;B-A,D-A,E-A" and long["equivalence"] == "D-E"


def test_the_seed_cap_marks_the_claims_intervals_only(tmp_path):
    _pilot(tmp_path / "gsm8k")
    _one_gpu(tmp_path / "one", acc=0.65)
    _map(tmp_path / "sim")
    # a spread so wide the rule asks for more seeds than the cap: the cap runs, the claims become intervals
    import bench.amend as am
    over = {"seeds": 40, "run": 12, "intervals_only": True, "sigma_diff_test": 0.05}
    original = am.seed_count
    am.seed_count = lambda *a, **k: over
    try:
        out = amend("gsm8k", tmp_path / "gsm8k", tmp_path / "one", tmp_path / "sim", delta=0.02, n_test=1319)
    finally:
        am.seed_count = original
    assert out["seeds"]["D"] == 12 and out["intervals_only"] is True


def test_the_rules_fallbacks_are_applied_and_recorded_not_stops(tmp_path):
    """The owner's policy (2026-10-02): a rule the pilot cannot support is recorded under `warnings` with its
    stated fallback, and the chain goes on; only inputs nothing can be computed from stop it."""
    _one_gpu(tmp_path / "one")
    _map(tmp_path / "sim")
    run = lambda pilot, **kw: amend("gsm8k", tmp_path / pilot, tmp_path / "one", tmp_path / "sim", delta=0.02,  # noqa
                                    n_test=1319, **kw)
    _pilot(tmp_path / "flat", gain=0.01)
    flat = run("flat")
    assert flat["T_s"] == 600.0 and any("MIN_GAIN" in w and "horizon" in w for w in flat["warnings"])
    _pilot(tmp_path / "short", tau=180.0, budget_s=300.0)  # t80 = 290 s > 0.8 x 300: T = the horizon
    short = run("short")
    assert short["T_s"] == 300.0 and any("beyond 0.8 x" in w for w in short["warnings"])
    _pilot(tmp_path / "fast", a_rate=3000.0)  # step_T = 250 > 150: read at the last common step
    fast = run("fast")
    assert set(fast["seed_snapshot_steps"]) == {150} and any("last common evaluated step 150" in w
                                                             for w in fast["warnings"])
    _pilot(tmp_path / "ok")
    over = run("ok", budget_h=0.1)
    assert over["seeds"] == run("ok")["seeds"] and any("exceeds the stage's 0.1 h" in w for w in over["warnings"])
    assert run("ok")["warnings"] == []
    with pytest.raises(AmendmentStop, match="no valid systems-map cell"):
        amend("gsm8k", tmp_path / "ok", tmp_path / "one", tmp_path / "nomap", delta=0.02, n_test=1319)
    with pytest.raises(AmendmentStop, match="has 0 D rows"):
        amend("gsm8k", tmp_path / "none", tmp_path / "one", tmp_path / "sim", delta=0.02, n_test=1319)
    # a pilot row flagged for its harness is read, and named
    flagged = tmp_path / "flagged"
    _pilot(flagged)
    m = flagged / "D-B32G8M32-K4-S1-pilot-s100" / "metrics.json"
    m.write_text(json.dumps(json.loads(m.read_text()) | {"fence_s": {"p50": 0.132}}))
    out = run("flagged")
    assert out["fit"]["points"] == 34 and any("fence p50 132 ms" in w for w in out["warnings"])
    # decision 7: each pilot row's record and the limits go into the amendment, for the owner to fix at the tag
    assert out["harness"]["D-B32G8M32-K4-S1-pilot-s100"]["fence_p50_s"] == 0.132
    # the limits by the stated rule: max(the code's, 1.5 x the highest pre-tag value of any arm)
    assert out["harness_limits"] == {"max_host_cpu": 0.75, "max_loop_lag_p99_s": 0.1, "max_fence_p50_s": 0.198}
    assert "1.5 x" in out["harness_limit_rule"]


def test_the_command_writes_the_file_and_signals_a_stop(tmp_path, capsys):
    _pilot(tmp_path / "gsm8k")
    _one_gpu(tmp_path / "one")
    _map(tmp_path / "sim")
    out = tmp_path / "amendments" / "gsm8k.yaml"
    rc = main(["gsm8k", "--pilot", str(tmp_path / "gsm8k"), "--one-gpu", str(tmp_path / "one"), "--map",
               str(tmp_path / "sim"), "--out", str(out)])
    assert rc == 0 and yaml.safe_load(out.read_text())["T_s"] == 300.0 and "best_async: D" in capsys.readouterr().out
    rc = main(["gsm8k", "--pilot", str(tmp_path / "gsm8k"), "--one-gpu", str(tmp_path / "one"), "--map",
               str(tmp_path / "none"), "--out", str(tmp_path / "x.yaml")])
    assert rc == 2 and "HARD STOP" in capsys.readouterr().err and not (tmp_path / "x.yaml").exists()
