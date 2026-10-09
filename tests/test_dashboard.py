"""The lab page's builder (bench.dashboard): rows, curves and timelines gathered from result directories, filtered
to the current harness (any task; the newest session's where two hold the same row), and rendered into the
template; the figures draw from the same data."""
import json
from pathlib import Path

from halfwritten.config import RunConfig, SimSettings
from halfwritten.results import row_dir, write_row
from halfwritten.simulate import configure, simulate


def _row(root, regime, budget_s=3.0, host=0.4, evals=True, tag="pilot", task="sim", seed=0, split="dev"):
    cfg = configure(RunConfig(B=4, G=4, M=4, K=2, budget_s=budget_s, task=task, tag=tag, seed=seed,
                              sim=SimSettings(prompt_len=16, length_mean=60)), regime)
    r = simulate(cfg, n_prompts=64)
    r.summary.host = {"main_thread_cpu_frac": host, "loop_lag_s": {"p99": 0.01}}
    path = write_row(row_dir(cfg, "NVIDIA H100 80GB HBM3", root), cfg, r.summary, r.controller.ledger, r.events,
                     env={"gpus": ["NVIDIA H100 80GB HBM3"]})
    if evals:
        (path / "evals.jsonl").write_text("".join(json.dumps(e) + "\n" for e in [
            {"tag": "init", "snapshot": "init", "t": 0.0, "accuracy": 0.5, "n": 100, "split": split},
            {"tag": "t01", "snapshot": f"t01-v{max(r.summary.steps // 2, 1)}", "t": budget_s / 2, "accuracy": 0.55,
             "n": 100, "split": split},
            {"tag": "tend", "snapshot": f"tend-v{r.summary.steps}", "t": budget_s, "accuracy": 0.6, "n": 100,
             "split": split}]))
    return path


def _session(root: Path, name: str, archived: str) -> Path:
    s = root / name
    s.mkdir(parents=True, exist_ok=True)
    (s / "MANIFEST.json").write_text(json.dumps({"archived": archived, "files": []}))
    return s


def test_the_page_gathers_rows_curves_and_timelines_and_renders(tmp_path):
    from bench import dashboard
    root = tmp_path / "data"
    s = _session(root, "box-2026-10-02", "2026-10-02")
    _row(s, "A")
    _row(s, "D")
    data = dashboard.gather([root])
    assert {r["regime"] for r in data["rows"]} == {"A", "D"} and all(r["gpu"] == "H100 SXM" for r in data["rows"])
    assert all(r["session"] == "box-2026-10-02" for r in data["rows"]) and data["sessions"] == {
        "box-2026-10-02": "2026-10-02/box-2026-10-02"}
    # a curve says what it is: its row, regime, tag, task and split, so nothing selects it by its label
    assert {(c["regime"], c["tag"], c["task"], c["split"]) for c in data["curves"]} == {("A", "pilot", "sim", "dev"),
                                                                                       ("D", "pilot", "sim", "dev")}
    assert all(c["row"] == c["label"].split(" ")[0] and c["session"] == "box-2026-10-02" for c in data["curves"])
    assert all("stale_kv_share" in r and "harness" in r and "cuts" in r for r in data["rows"])
    assert not any(k in data["rows"][0] for k in ("reward_curve", "k3", "profile"))  # nothing reads them
    assert all(k in data["rows"][0] for k in ("sync_p50", "pause_p50", "syncs", "tokens_per_s"))  # the throughput strip
    tl = dashboard.timeline(root.parent / data["rows"][0]["source"], window=(0.0, 2.0))
    assert set(tl["lanes"]) == {"engine", "trainer", "sender"} and tl["lanes"]["engine"]
    blob = json.dumps(data, default=float)
    page = (Path(dashboard.__file__).parent / "dashboard.html").read_text().replace("/*DATA*/", blob)
    assert "<title>halfwritten</title>" in page and blob[:20] in page


def test_the_current_filter_keeps_rows_on_the_current_harness_from_the_newest_session(tmp_path):
    from bench import dashboard
    root = tmp_path / "data"
    old, new = _session(root, "box-2026-10-01", "2026-10-02"), _session(root, "box-2026-10-02", "2026-10-02")
    _row(old, "D", budget_s=3.0)
    data = dashboard.current(dashboard.gather([root]))
    assert data["rows"] == []  # a 3 s row is not a measurement (the page wants 180 s or more)
    empty = {"curves": [], "canaries": [], "floors": [], "phase1": [], "timelines": {}}
    rows = dashboard.gather([root])["rows"]
    rows[0].update(colocated=False, budget_s=300.0)
    assert dashboard.current({"rows": rows, **empty})["rows"]  # any task: the systems map's cells are task "sim"
    rows[0]["host_cpu"] = None  # no harness record: not current
    assert not dashboard.current({"rows": rows, **empty})["rows"]
    # the same row run again in a newer session (a copy of one row across sessions is one row; a rerun differs in
    # its wall time): the newer session's is the current one
    _row(new, "D", budget_s=3.5)
    rows = dashboard.gather([root])["rows"]
    assert len(rows) == 2 and {r["session"] for r in rows} == {"box-2026-10-01", "box-2026-10-02"}
    for r in rows:
        r.update(colocated=False, budget_s=300.0)
    cur = dashboard.current({"rows": rows, "sessions": dashboard.sessions([root]), **empty})["rows"]
    assert [r["session"] for r in cur] == ["box-2026-10-02"]


def test_grid_curves_survive_the_filter_and_draw(tmp_path):
    from bench import dashboard, figures
    root = tmp_path / "data"
    s = _session(root, "box-2026-10-02", "2026-10-02")
    for regime in ("A", "D"):
        for seed in (0, 1):
            _row(s, regime, budget_s=4.0, tag="", seed=seed, split="test", task="gsm8k")  # a grid's rows
            _row(s, regime, budget_s=4.0, tag="", seed=seed, split="test", task="math")  # the other task's grid
    _row(s, "A", budget_s=4.0, tag="pilot", split="test", task="gsm8k")
    _row(s, "D", budget_s=4.0, tag="depth", split="test", task="math", seed=5)  # a factor study, not the grid
    data = dashboard.gather([root])
    for r in data["rows"]:
        r.update(colocated=False, budget_s=300.0)
    cur = dashboard.current(data)
    assert len(cur["rows"]) == 10 and {c["tag"] for c in cur["curves"]} == {"", "pilot", "depth"}
    assert sum(1 for c in cur["curves"] if c["tag"] == "") == 8
    written = figures.accuracy_vs_time(cur, tmp_path)
    assert [p.name for p in written] == ["accuracy_vs_time-gsm8k.svg", "accuracy_vs_time-math.svg"]
    for p in written:  # one line per arm per task: never the other task's seeds, nor a tagged row's, pooled in
        svg = p.read_text()
        assert svg.startswith("<?xml") and "A (n = 2)" in svg and "D (n = 2)" in svg and "n = 4" not in svg
    assert figures.pilot(cur, tmp_path).exists()


def test_the_page_renders_every_analysis_it_is_given(tmp_path):
    """Several experiments' analyses (GSM8K's grid, MATH's) each reach the page as their own block with their own
    delta: nothing merges them by key."""
    from bench import dashboard
    root = tmp_path / "data"
    _row(_session(root, "box-2026-10-02", "2026-10-02"), "A")
    files = []
    for task, delta in (("gsm8k", 0.02), ("math", 0.03)):
        f = tmp_path / f"{task}.json"
        f.write_text(json.dumps({
            "task": task, "tag": "", "primary": "auc", "delta": delta,
            "family": {"D-E": {"diff": 0.01, "ci95": [0.0, 0.02], "p_holm": 0.04, "verdict": "superior"}},
            "equivalence": [{"contrast": "D-E", "family": 1, "result": "equivalent", "verdict": "equivalent"}]}))
        files.append(str(f))
    out = tmp_path / "index.html"
    dashboard.main(["--roots", str(root), "--out", str(out), "--analysis", *files])
    page = out.read_text()
    assert "grid-family-gsm8k" in page and "grid-family-math" in page
    assert "delta = 2 points" in page and "delta = 3 points" in page


def test_a_canary_judged_natively_under_the_decisive_gate_reaches_the_page(tmp_path):
    from bench import dashboard
    root = tmp_path / "data"
    s = _session(root, "box-2026-10-02", "2026-10-02")
    canary = lambda post: {"post_old": 0.0, "post_fresh": post["fresh"], "post_stale": post["stale"],  # noqa: E731
                           "boundary": 40, "total": 120, "load_streams": 255, "misses": 0, "decisive_stale_fresh": 50,
                           "post_stale_on_decisive": post["stale"], "post_fresh_on_decisive": post["fresh"]}
    for name, failures in (("canary-1790944919", []), ("canary-1790941313", ["D: fresh"])):
        d = s / "_queue" / name
        d.mkdir(parents=True)
        (d / "report.json").write_text(json.dumps({
            "gpu": "H100 SXM", "canary_D": canary({"fresh": 0.0, "stale": 1.0}),
            "canary_E": canary({"fresh": 1.0, "stale": 0.0}), "canary_failures": failures}))
    data = dashboard.gather([root])
    assert {(c["source"].split("/")[-2], c["passed_now"]) for c in data["canaries"]} == {
        ("canary-1790944919", True), ("canary-1790941313", False)}
    cur = dashboard.current(data)["canaries"]
    assert [c["source"].split("/")[-2] for c in cur] == ["canary-1790944919"] * 2  # the failed one is not current
    assert cur[0]["at"] == 1790944919 and dashboard.canary_time("canary-exact-115416", "two_gpu-2026-10-02") > 0
