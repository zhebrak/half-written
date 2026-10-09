"""The findings (bench.findings): each renders from the page's data without a NaN or an empty number, names its
sources, says when a row ran outside the harness rule, and disappears when its data is absent; the grid's results
render from an analysis."""
import json
import math
import re

from bench.findings import f_pilot, findings, markdown

SESSION = "two_gpu-2026-10-02"


def pilot_curve(arm, tau, task="gsm8k", split="test", session=SESSION, horizon=600.0, n=1319):
    label = f"{arm}-B32G8M32-K4-S1-pilot-s100" + ("" if split == "test" else f" ({split})")
    return {"label": label, "row": f"{arm}-B32G8M32-K4-S1-pilot-s100", "file": "evals.jsonl", "split": split,
            "task": task, "tag": "pilot", "regime": arm, "lora": False, "session": session,
            "source": f"data/{session}/{task}/{arm}",
            "points": [{"step": k * 18, "acc": 0.59 + 0.12 * (1 - math.exp(-horizon / 16 * k / tau)), "n": n,
                        "t": horizon / 16 * k} for k in range(17)]}


def row(regime, tag, rate, task="gsm8k", session=SESSION, budget_s=600.0, from_w0=True, harness=None, **kw):
    return {"label": f"{regime}-B32G8M32-K4-S1-{tag}-s0", "regime": regime, "tag": tag, "task": task,
            "session": session, "steps_per_hour": rate, "lora": False, "api_servers": 4, "K": 4,
            "budget_s": budget_s, "from_w0": from_w0, "harness": harness,
            "source": f"data/{session}/{task}/{regime}-{tag}", "stale_kv_share": None, "fence_p50": 0.05,
            "cuts": {"n": 100, "exact": 100, "recorded": True}, **kw}


def _data():
    table = [{"lag": k, "R": 0.5, "R_ci95": "0.4-0.6", "R_0_8": 0.85, "R_32_999": 0.4, "removed_l14": 0.1,
              "removed_l21": 0.9, "drift_median": 4.4e-7 if k == 1 else 1e-5, "drift_mean": 4.85e-5 if k == 1 else 1e-3}
             for k in (1, 2, 4, 8, 16, 32)]
    bf16 = [t | {"drift_median": 1.6e-4 if t["lag"] <= 4 else 1.4e-3, "drift_mean": 7.1e-4} for t in table]
    return {
        "rows": [row("A", "pilot", 864.0), row("D", "pilot", 1740.0, stale_kv_share=0.051,
                                                harness="fence p50 132 ms (limit 100 ms)"),
                 row("E", "pilot", 1608.0, harness="fence p50 131 ms (limit 100 ms)"),
                 row("B", "validate-exact", 1420.0, budget_s=180.0, from_w0=False),
                 row("D", "certify-exact", 1692.0, budget_s=300.0, from_w0=False),
                 row("E", "certify-exact", 1650.0, budget_s=300.0, from_w0=False),
                 row("C1", "validate-fixed", 1680.0, budget_s=300.0, from_w0=False,
                     cuts={"n": 84, "exact": 0, "recorded": False}),  # cut by the old window fence
                 row("D", "validate-api1", 1120.0, api_servers=1, stale_kv_share=0.194,
                     harness="fence p50 250 ms (limit 100 ms)"),
                 row("D", "validate", 320.0, lora=True, api_servers=1), row("F", "validate", 900.0, api_servers=1)],
        "curves": [pilot_curve("A", 428.0), pilot_curve("D", 178.0), pilot_curve("E", 303.0),
                   {"label": "A-B32G8M32-K0-S1-s1-s0", "row": "A-B32G8M32-K0-S1-s1-s0", "file": "evals.jsonl",
                    "split": "test", "task": "gsm8k", "tag": "s1", "regime": "A", "lora": False,
                    "session": "h100nvl-2026-10-01", "source": "data/h100nvl-2026-10-01/s1",
                    "points": [{"step": float(s), "acc": 0.46 + 0.26 * (1 - math.exp(-s / 90)), "n": 1319}
                               for s in range(0, 500, 25)]}],
        "canaries": [{"regime": "D", "loaded": True, "load": 255, "passed_now": True, "decisive": 294,
                      "source": f"data/{SESSION}/c"},
                     {"regime": "E", "loaded": True, "load": 255, "passed_now": True, "decisive": 57,
                      "source": f"data/{SESSION}/c"}],
        "phase1": [{"compute": "fp32", "gpu": "H100 NVL", "table": table, "source": "data/h100nvl-2026-10-01/p32"},
                   {"compute": "bf16", "gpu": "H100 NVL", "table": bf16, "source": "data/h100nvl-2026-10-01/p16"}],
        "floors": [{"engine_head": "fp32", "condition": "steady", "k3": 7.13e-4, "tokens": 24239,
                    "source": "data/h100nvl-2026-10-01/floor"},
                   {"engine_head": "bf16", "condition": "steady", "k3": 9.1e-4, "tokens": 26651,
                    "source": "data/h100nvl-2026-10-01/floor"}],
        "timelines": {}}


def test_every_finding_renders_from_the_data_and_names_its_sources():
    out = findings(_data())
    assert [f["id"] for f in out] == ["canary", "exact-cut", "throughput", "pilot", "exposure", "s1", "phase1", "bf16",
                                      "lora"]
    for f in out:
        assert f["title"] and f["text"] and f["state"] and f["kind"] in ("good", "old", "warn", "bad"), f["id"]
        assert "nan" not in f["text"].lower() and "None" not in f["text"] and "??" not in f["text"], f["text"]
        assert not re.search(r"\b\d+\.\d{4,}", f["text"]), f["text"]  # numbers are rounded for a reader
        assert f["sources"] and all(s.startswith("data/") for s in f["sources"]), f["id"]
    by = {f["id"]: f for f in out}
    assert "255 answers alongside" in by["canary"]["text"] and "294 positions" in by["canary"]["text"]
    assert by["throughput"]["title"] == ("On GSM8K at K = 4, one 10-minute row each from W0: D made 2.0x A's "
                                         "optimiser steps an hour, E 1.9x.")
    assert "1,420 an hour in a 3-minute row from the base model" in by["throughput"]["text"]
    assert by["pilot"]["title"] == ("In the first pilot the fitted t80 was D's 4.8 minutes; E's 8.1 and A's 11.5 lie "
                                    "past what a 10-minute run supports.")
    assert "the dev rerun sets T and the seeds" in by["pilot"]["text"] and "sets the study" not in by["pilot"]["text"]
    assert "5% of its tokens" in by["exposure"]["title"] and "19%" in by["exposure"]["text"]
    assert "D and E cannot differ much" in by["exposure"]["text"] and by["exposure"]["title"].startswith("On GSM8K")
    # rows outside the harness rule are reported with their diagnostic wherever a finding rests on them
    for fid in ("throughput", "pilot", "exposure"):
        assert "D's pilot row: fence p50 132 ms (limit 100 ms)" in by[fid]["text"], fid
        assert "outside the harness rule as pre-registered (decision 7)" in by[fid]["text"], fid
    assert "E's pilot row: fence p50 131 ms" in by["throughput"]["text"]
    assert "D's validate-api1 row: fence p50 250 ms" in by["exposure"]["text"] and "harness" not in by["s1"]["text"]
    # only rows whose syncs record the fence's count are certification cuts; the window-fence row is not one
    assert by["exact-cut"]["title"] == "Every cut is exact." and "200 of 200 cuts of the 2 certification" in \
        by["exact-cut"]["text"]
    assert "to step 475." in by["s1"]["text"] and "475.0" not in by["s1"]["text"]
    assert "every lag from 1 to 32 steps" in by["phase1"]["text"]
    assert "1.6e-4 nats (median)" in by["bf16"]["text"] and "4.4e-7" in by["bf16"]["text"]
    assert "1.4e-3 at lag 32" in by["bf16"]["text"]
    assert "mean over 24,239 tokens) is 7.1e-4, about 15 times the mean one-step drift in fp32 (4.8e-5)" in \
        by["bf16"]["text"]
    assert "320" in by["lora"]["text"] and "1,120" in by["lora"]["text"] and "900" in by["lora"]["text"]


def test_findings_without_their_data_are_absent_and_the_grid_adds_results():
    data = _data()
    assert f_pilot({"curves": []}, None) is None
    data["canaries"] = []
    data["phase1"] = []
    ids = [f["id"] for f in findings(data)]
    assert "canary" not in ids and "phase1" not in ids and "bf16" not in ids and "throughput" in ids
    analysis = {"primary": "auc", "delta": 0.02,
                "family": {"D-E": {"diff": 0.003, "ci95": [-0.01, 0.016], "p_holm": 0.6, "verdict": "not significant"},
                           "best-A": {"missing": "0 paired seeds"}},
                "equivalence": [{"contrast": "D-E", "family": 1, "result": "equivalent", "verdict": "equivalent"},
                                {"contrast": "B-A", "family": 2, "result": "missing"}],
                "steps_to_threshold": {"D/A": {"ratio": 1.02, "ci95": [0.9, 1.15]}}}
    out = findings(data, analysis)
    assert [f["id"] for f in out[:3]] == ["grid-family", "grid-equivalence", "grid-steps"]
    assert "D-E: +0.3 points" in out[0]["text"] and "not computed" in out[0]["text"]
    assert "D-E (family 1, the primary endpoint over wall clock, H4): equivalent, equivalent" in out[1]["text"]
    assert "1.02" in out[2]["text"]
    md = markdown(out)
    assert md.startswith("- **") and md.count("\n") == len(out) - 1 and json.dumps(out)
    # decision 13: past the seed cap the grid's findings carry intervals and no claim
    only = analysis | {"intervals_only": True,
                       "family": {"D-E": analysis["family"]["D-E"] | {"verdict": "interval only"}},
                       "equivalence": [{"contrast": "D-E", "family": 1, "result": "interval only",
                                        "ci90": [-0.008, 0.014]}]}
    grid = findings(data, only)[:2]
    assert all("intervals only" in f["title"] for f in grid) and grid[0]["kind"] == "old"
    assert "interval only, 90% CI [-0.8, +1.4]" in grid[1]["text"]
    assert not any(w in grid[1]["text"] for w in ("equivalent", "superior", "inferior"))


def test_each_analysis_renders_its_own_grid_block_with_its_own_delta():
    """Several experiments' files (GSM8K's grid, MATH's, MATH's factor studies) each get their blocks, named and
    reading their own delta: nothing merges by key, so MATH's results are never shown as GSM8K's."""
    data = _data()
    gsm8k = {"task": "gsm8k", "tag": "", "primary": "auc", "delta": 0.02,
             "family": {"D-E": {"diff": 0.003, "ci95": [-0.01, 0.016], "p_holm": 0.6, "verdict": "not significant"}},
             "equivalence": [{"contrast": "D-E", "family": 1, "result": "equivalent", "verdict": "equivalent"}]}
    math = {"task": "math", "tag": "", "primary": "auc", "delta": 0.03,
            "family": {"D-E": {"diff": -0.021, "ci95": [-0.04, -0.002], "p_holm": 0.03, "verdict": "inferior"}},
            "equivalence": [{"contrast": "D-E", "family": 1, "result": "not shown equivalent"}]}
    depth = math | {"tag": "depth", "equivalence": []}
    out = findings(data, [gsm8k, math, depth])
    assert [f["id"] for f in out if f["id"].startswith("grid-")] == [
        "grid-family-gsm8k", "grid-equivalence-gsm8k", "grid-family-math", "grid-equivalence-math",
        "grid-family-math-depth"]
    by = {f["id"]: f for f in out}
    assert by["grid-family-gsm8k"]["title"].startswith("GSM8K: the superiority") and "+0.3 points" in by[
        "grid-family-gsm8k"]["text"]
    assert by["grid-family-math"]["title"].startswith("MATH: ") and "-2.1 points" in by["grid-family-math"]["text"]
    assert "delta = 2 points" in by["grid-equivalence-gsm8k"]["title"]
    assert "delta = 3 points" in by["grid-equivalence-math"]["title"]
    assert by["grid-family-math-depth"]["title"].startswith("MATH depth stress: ")
    assert findings(data, gsm8k)[0]["id"] == "grid-family-gsm8k"  # one dict is one experiment


def test_each_finding_reads_the_gsm8k_test_pilot_of_the_newest_session_among_three_pilots():
    """A dev rerun, MATH's pilot and an older session's pilot sit beside the GSM8K test pilot: every finding reads
    the one it names, by the curve's and row's own fields and the session order, never by a label."""
    data = _data()
    data["sessions"] = {"two_gpu-2026-10-01": "2026-10-01/two_gpu-2026-10-01", SESSION: f"2026-10-02/{SESSION}",
                        "box-2026-10-03": "2026-10-03/box-2026-10-03"}
    data["curves"] += [pilot_curve("D", 50.0, split="dev", session="box-2026-10-03"),  # the dev rerun: faster
                       pilot_curve("D", 900.0, task="math", session="box-2026-10-03"),  # MATH's pilot: slower
                       pilot_curve("D", 900.0, session="two_gpu-2026-10-01")]  # an older session's
    data["rows"] += [row("D", "pilot", 99.0, task="math", session="box-2026-10-03"),
                     row("D", "pilot", 1.0, session="two_gpu-2026-10-01"),
                     row("D", "certify", 1.0, session="two_gpu-2026-10-01", cuts={"n": 9, "exact": 3,
                                                                                  "recorded": True})]
    by = {f["id"]: f for f in findings(data)}
    assert "D's 4.8 minutes" in by["pilot"]["title"]  # the GSM8K test pilot's fit, not dev's, MATH's or the old one
    assert "D made 2.0x" in by["throughput"]["title"] and by["exposure"]["title"].endswith("5% of its tokens.")
    assert "200 of 200 cuts of the 2 certification rows of two_gpu-2026-10-02" in by["exact-cut"]["text"]
    # a newer session with its own pilot takes over
    data["curves"] += [pilot_curve(arm, tau, session="box-2026-10-03") for arm, tau in (("A", 400.0), ("D", 100.0),
                                                                                       ("E", 300.0))]
    data["rows"] += [row(arm, "pilot", rate, session="box-2026-10-03") for arm, rate in (("A", 900.0), ("D", 2700.0),
                                                                                         ("E", 1700.0))]
    by = {f["id"]: f for f in findings(data)}
    assert "D made 3.0x" in by["throughput"]["title"] and "D's 2.7 minutes" in by["pilot"]["title"]


def test_the_readme_block_is_rewritten_between_its_markers(tmp_path):
    from bench.dashboard import write_readme_block
    readme = tmp_path / "README.md"
    readme.write_text("# x\n\nbefore\n\n<!-- findings -->\nold line\n<!-- /findings -->\n\nafter\n")
    write_readme_block(readme, findings(_data()))
    text = readme.read_text()
    assert "old line" not in text and text.endswith("after\n")
    assert text.startswith("# x\n\nbefore\n\n<!-- findings -->\n")
    assert "- **Every cut is exact.**" in text
