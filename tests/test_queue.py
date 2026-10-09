"""The run queue: seed blocks, a matrix's rows and variants, invalid rows and the session gate."""
import pytest
import yaml

from bench.queue import plan


def test_queue_runs_whole_seed_blocks_in_shuffled_order():
    arms = ["A", "B", "D", "E", "C1"]
    order = plan(arms, seeds=[0, 1, 2])
    blocks = [order[i:i + 5] for i in range(0, 15, 5)]
    assert all(sorted(a for a, _ in b) == sorted(arms) and {s for _, s in b} == {i} for i, b in enumerate(blocks))
    assert len({tuple(a for a, _ in b) for b in blocks}) > 1
    assert plan(arms, [0, 1, 2]) == order


def test_an_arm_with_fewer_seeds_sits_out_the_later_blocks():
    order = plan(["A", "C1", "D"], {"A": [0, 1, 2], "C1": [0, 1], "D": [0, 1, 2]})
    assert [s for _, s in order] == [0, 0, 0, 1, 1, 1, 2, 2]
    assert sorted(a for a, s in order if s == 2) == ["A", "D"]


def test_queue_matrix_rows_and_a_dry_run(tmp_path):
    import yaml

    from bench.queue import row_overrides, run_matrix
    matrix = {"base": "configs/cpu/smoke.yaml", "arms": ["A", "D", "F"], "seeds": [0, 1], "set": ["budget_s=3"],
              "per_arm": {"D": ["snapshot_every_s=1"]}}
    assert row_overrides(matrix, "A", 1) == ["budget_s=3", "regime=A", "seed=1", "K=0", "S=1"]
    assert row_overrides(matrix, "D", 0)[-3:] == ["snapshot_every_s=1", "regime=D", "seed=0"]
    assert "lora=true" in row_overrides(matrix, "F", 0)
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(matrix))
    report = run_matrix(str(path), str(tmp_path / "out"), dry_run=True)
    assert [r["status"] for r in report] == ["planned"] * 6
    assert len({r["row"] for r in report}) == 6


def test_queue_variants_become_tagged_rows():
    from bench.queue import cells, row_overrides
    m = {"arms": ["B", "D"], "seeds": [0, 1],
         "variants": {"short": ["forced_lengths=fixed:64"], "long": ["forced_lengths=fixed:512"]}}
    c = cells(m)
    assert len(c) == 8 and {(a, v) for a, v, s in c if s == 0} == {(a, v) for a in "BD" for v in ("short", "long")}
    assert [s for _, _, s in c] == [0] * 4 + [1] * 4  # seed blocks
    assert row_overrides(m, "D", 0, "long")[-4:] == ["forced_lengths=fixed:512", "tag=long", "regime=D", "seed=0"]


def test_every_committed_matrix_plans_valid_distinct_rows(tmp_path):
    from pathlib import Path

    import yaml

    from bench.queue import run_matrix
    root = Path(__file__).resolve().parents[1]
    amendment = tmp_path / "amend.yaml"
    amendment.write_text(yaml.safe_dump({"T_s": 300.0, "snapshot_every_s": 18.75,
                                         "seeds": {"A": 3, "B": 3, "C1": 2, "D": 3, "E": 3}}))
    for m in sorted((root / "configs" / "matrix").glob("*.yaml")):
        # the confirmatory grids and the tier-2 matrices plan only under an amendment: a synthetic one here
        amend = str(amendment) if yaml.safe_load(m.read_text()).get("requires_amendment") else None
        report = run_matrix(str(m), str(tmp_path / m.stem), dry_run=True, log=lambda *_: None, amendment=amend)
        assert report and all(r["status"] == "planned" for r in report), m.name
        assert len({r["row"] for r in report}) == len(report), m.name


def test_queue_moves_an_invalid_row_aside_and_runs_it_once_more(tmp_path, monkeypatch):
    import json
    import subprocess

    import yaml

    import bench.queue as q
    import halfwritten.gpu
    matrix = {"base": "configs/cpu/smoke.yaml", "arms": ["D"], "seeds": [0]}
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(matrix))
    calls = []

    def fake_run(cmd, **kw):  # the first attempt breaks an invariant, the second is valid
        from halfwritten.config import load, to_dict, with_overrides
        from halfwritten.launch import finished, row_dir_on_host
        sets = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--set"]
        cfg = with_overrides(load(matrix["base"]), sets)
        row = row_dir_on_host(cfg, tmp_path / "out")
        finished(row, cfg)  # as run_row begins: an invalid row is moved aside here, once
        row.mkdir(parents=True, exist_ok=True)
        (row / "config.yaml").write_text(yaml.safe_dump({"config": to_dict(cfg), "identity": {}}))
        valid = len(calls) > 0
        calls.append(row)
        (row / "metrics.json").write_text(json.dumps({"valid": valid}))
        return subprocess.CompletedProcess(cmd, 0 if valid else 1, "", "")

    monkeypatch.setattr(q, "run_command", fake_run)
    monkeypatch.setattr(halfwritten.gpu, "wait_for_gpu", lambda **kw: True)
    report = q.run_matrix(str(path), str(tmp_path / "out"), log=lambda *_: None)
    assert [r["status"] for r in report] == ["invalid, rerun", "valid"]
    assert len(calls) == 2 and len(list(calls[0].parent.glob("*.invalid-*"))) == 1
    # a later session finds the valid row and runs nothing; a dry run moves nothing
    calls.clear()
    assert [r["status"] for r in q.run_matrix(str(path), str(tmp_path / "out"), log=lambda *_: None)] == ["done"]
    assert calls == []


def test_an_invalid_row_found_at_start_is_rerun_once_not_twice(tmp_path, monkeypatch):
    """Code review of 2026-10-02: the queue and finished() each set an invalid row aside, so a row found invalid at
    the start of a session was run twice more. Now finished() alone counts the reruns."""
    import json
    import subprocess

    import yaml

    import bench.queue as q
    import halfwritten.gpu
    from halfwritten.config import load, to_dict, with_overrides
    from halfwritten.launch import finished, row_dir_on_host
    matrix = {"base": "configs/cpu/smoke.yaml", "arms": ["D"], "seeds": [0]}
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(matrix))
    cfg = with_overrides(load(matrix["base"]), q.row_overrides(matrix, "D", 0))
    row = row_dir_on_host(cfg, tmp_path / "out")
    row.mkdir(parents=True)  # left invalid by a direct `halfwritten run`
    (row / "config.yaml").write_text(yaml.safe_dump({"config": to_dict(cfg), "identity": {}}))
    (row / "metrics.json").write_text(json.dumps({"valid": False}))
    assert q.run_matrix(str(path), str(tmp_path / "out"), dry_run=True, log=lambda *_: None)[0]["status"] == "planned"
    assert row.exists() and not list(row.parent.glob("*.invalid-*"))  # a dry run moves nothing
    runs = []

    def fake_run(cmd, **kw):  # every attempt is invalid
        finished(row, cfg)
        row.mkdir(parents=True, exist_ok=True)
        (row / "config.yaml").write_text(yaml.safe_dump({"config": to_dict(cfg), "identity": {}}))
        (row / "metrics.json").write_text(json.dumps({"valid": False}))
        runs.append(1)
        return subprocess.CompletedProcess(cmd, 1, "", "")
    monkeypatch.setattr(q, "run_command", fake_run)
    monkeypatch.setattr(halfwritten.gpu, "wait_for_gpu", lambda **kw: True)
    report = q.run_matrix(str(path), str(tmp_path / "out"), log=lambda *_: None)
    assert len(runs) == 1 and report[-1]["status"] == "invalid"
    assert len(list(row.parent.glob("*.invalid-*"))) == 1


def test_queue_gate_certifies_the_launch_the_rows_use(tmp_path, monkeypatch):
    import subprocess

    import yaml

    import bench.queue as q
    import halfwritten.gpu
    matrix = {"base": "configs/cpu/smoke.yaml", "arms": ["D"], "seeds": [0], "canary": True,
              "set": ["launch.head_fp32=true", "budget_s=3"]}
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(matrix))
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "canary: D failed", "")
    monkeypatch.setattr(q, "run_command", fake_run)
    monkeypatch.setattr(halfwritten.gpu, "wait_for_gpu", lambda **kw: True)
    report = q.run_matrix(str(path), str(tmp_path / "out"), log=lambda *_: None)
    assert [e["status"] for e in report] == ["canary failed"] and report[0]["arm"] is None
    cmd = " ".join(seen[0])
    assert "--config configs/cpu/smoke.yaml" in cmd and "--set launch.head_fp32=true" in cmd and "budget_s" not in cmd
    assert "--background-answers 7" in cmd  # under the runs' load: M*G - 1 other answers (the smoke config's M=2, G=4)


def test_a_run_can_take_some_of_a_matrixs_variants(tmp_path):
    from bench.queue import run_matrix
    m = {"base": "configs/cpu/smoke.yaml", "arms": ["D"], "seeds": [0],
         "variants": {"short": ["forced_lengths=fixed:64"], "long": ["forced_lengths=fixed:512"]}}
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(m))
    rows = run_matrix(str(path), str(tmp_path / "out"), dry_run=True, log=lambda *_: None, variants=["long"])
    assert [r["variant"] for r in rows] == ["long"]
    with pytest.raises(ValueError, match="no variants"):
        run_matrix(str(path), str(tmp_path / "out"), dry_run=True, log=lambda *_: None, variants=["medium"])


def test_the_gate_certifies_each_load_and_launch_its_rows_use(tmp_path, monkeypatch):
    """External review F04: a variant at another M (the systems map's M/B = 2) ran under a canary made at the base
    config's load. One canary now runs per distinct certificate, before any row, with that load's background answers;
    a row whose certificate failed is skipped, the others run in their planned order and record theirs."""
    import json
    import subprocess

    import yaml

    import bench.queue as q
    import halfwritten.gpu
    matrix = {"base": "configs/cpu/smoke.yaml", "arms": ["D", "E"], "seeds": [0], "canary": True,
              "variants": {"base": [], "wide": ["M=4"]}}
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(matrix))
    canaries, runs = [], []

    def fake_run(cmd, **kw):
        if "bench.canary" in cmd:
            canaries.append(cmd)
            wide = cmd[cmd.index("--background-answers") + 1] == "15"  # M=4 x G=4 - 1
            return subprocess.CompletedProcess(cmd, 1 if wide else 0, "", "")
        from halfwritten.config import load, with_overrides
        from halfwritten.launch import row_dir_on_host
        sets = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--set"]
        row = row_dir_on_host(with_overrides(load(matrix["base"]), sets), tmp_path / "out")
        row.mkdir(parents=True, exist_ok=True)
        (row / "metrics.json").write_text(json.dumps({"valid": True}))
        runs.append(row)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(q, "run_command", fake_run)
    monkeypatch.setattr(halfwritten.gpu, "wait_for_gpu", lambda **kw: True)
    report = q.run_matrix(str(path), str(tmp_path / "out"), log=lambda *_: None)
    assert sorted(c[c.index("--background-answers") + 1] for c in canaries) == ["15", "7"]
    assert sorted(next(x for x in c if x.startswith("M=")) for c in canaries) == ["M=2", "M=4"]
    status = {(r["variant"], r["arm"]): r["status"] for r in report}
    assert status == {("base", "D"): "valid", ("base", "E"): "valid", ("wide", "D"): "canary failed",
                      ("wide", "E"): "canary failed"}
    assert all(json.loads((r / "certificate.json").read_text())["passed"] for r in runs)


def _matrix(tmp_path, **extra):
    import yaml
    matrix = {"base": "configs/cpu/smoke.yaml", "arms": ["D"], "seeds": [0], **extra}
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(matrix))
    return matrix, path


def _row_of(matrix, tmp_path):
    import bench.queue as q
    from halfwritten.config import load, to_dict, with_overrides
    from halfwritten.launch import row_dir_on_host
    cfg = with_overrides(load(matrix["base"]), q.row_overrides(matrix, "D", 0))
    return cfg, row_dir_on_host(cfg, tmp_path / "out"), {"config": to_dict(cfg), "identity": {}}


def test_the_command_exits_nonzero_when_the_queue_could_not_run(monkeypatch, capsys):
    """Staff engineer's review, 2026-10-02: the early returns had no `arm`, so main() crashed with a KeyError, and
    its exit status was 0 whatever happened, so a slot's step saw success."""
    import bench.queue as q
    monkeypatch.setattr(q, "run_matrix", lambda *a, **k: [
        {"status": "gpu never free", "arm": None, "variant": None, "seed": None, "row": None}])
    assert q.main(["run", "m.yaml"]) == 1
    assert "gpu never free" in capsys.readouterr().out
    monkeypatch.setattr(q, "run_matrix", lambda *a, **k: [{"status": "valid", "arm": "D", "seed": 0, "row": "r"},
                                                         {"status": "done", "arm": "E", "seed": 0, "row": "r2"}])
    assert q.main(["run", "m.yaml"]) == 0
    monkeypatch.setattr(q, "run_matrix", lambda *a, **k: [{"status": "valid", "arm": "D", "seed": 0, "row": "r"},
                                                         {"status": "failed", "arm": "E", "seed": 0, "row": "r2"}])
    assert q.main(["run", "m.yaml"]) == 1


def test_a_rows_status_is_its_metrics_not_its_exit_code(tmp_path, monkeypatch):
    """A valid row whose process exits non-zero in its teardown is valid; a process that exits 0 after writing an
    invalid row is invalid (and rerun once)."""
    import json
    import subprocess

    import yaml

    import bench.queue as q
    import halfwritten.gpu
    from halfwritten.launch import finished
    matrix, path = _matrix(tmp_path)
    cfg, row, recorded = _row_of(matrix, tmp_path)
    calls = []

    def fake_run(cmd, **kw):
        finished(row, cfg)
        row.mkdir(parents=True, exist_ok=True)
        (row / "config.yaml").write_text(yaml.safe_dump(recorded))
        calls.append(cmd)
        (row / "metrics.json").write_text(json.dumps({"valid": True, "host": {"main_thread_cpu_frac": 0.3}}))
        return subprocess.CompletedProcess(cmd, 1, "teardown error", "")
    monkeypatch.setattr(q, "run_command", fake_run)
    monkeypatch.setattr(halfwritten.gpu, "wait_for_gpu", lambda **kw: True)
    assert [r["status"] for r in q.run_matrix(str(path), str(tmp_path / "out"), log=lambda *_: None)] == ["valid"]
    assert len(calls) == 1

    def fake_run_invalid(cmd, **kw):
        finished(row, cfg)
        row.mkdir(parents=True, exist_ok=True)
        (row / "config.yaml").write_text(yaml.safe_dump(recorded))
        calls.append(cmd)
        (row / "metrics.json").write_text(json.dumps({"valid": False, "violations": ["I9: late"]}))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    calls.clear()
    import shutil
    shutil.rmtree(tmp_path / "out")
    monkeypatch.setattr(q, "run_command", fake_run_invalid)
    report = q.run_matrix(str(path), str(tmp_path / "out"), log=lambda *_: None)
    assert [r["status"] for r in report] == ["invalid, rerun", "invalid"] and len(calls) == 2
    assert report[-1]["problem"].startswith("invalid: I9")


def test_a_twice_invalid_row_stands_and_is_not_evaluated(tmp_path, monkeypatch):
    """A row whose rerun was invalid too is reported as standing invalid, not "done", and never evaluated."""
    import json

    import yaml

    import bench.queue as q
    import halfwritten.gpu
    matrix, path = _matrix(tmp_path, evaluate=True)
    cfg, row, recorded = _row_of(matrix, tmp_path)
    row.mkdir(parents=True)
    (row / "config.yaml").write_text(yaml.safe_dump(recorded))
    (row / "metrics.json").write_text(json.dumps({"valid": False, "error": "ControllerError: I9"}))
    (row.parent / f"{row.name}.invalid-1").mkdir()
    calls = []
    monkeypatch.setattr(q, "run_command", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(halfwritten.gpu, "wait_for_gpu", lambda **kw: True)
    report = q.run_matrix(str(path), str(tmp_path / "out"), log=lambda *_: None)
    assert [r["status"] for r in report] == ["invalid (stands)"] and "I9" in report[0]["problem"]
    assert calls == [] and row.exists()


def test_a_harness_flag_is_recorded_not_rerun(tmp_path, monkeypatch):
    """A valid row whose harness record trips decision 7's limits (the controller near a full core) stands and is
    flagged in the report; the analysis applies the rule (the owner's policy, 2026-10-02: run, then decide on the
    limits). Earlier the queue reran such rows, which fall almost only on D and E."""
    import json
    import subprocess

    import yaml

    import bench.queue as q
    import halfwritten.gpu
    from halfwritten.launch import finished
    matrix, path = _matrix(tmp_path)
    cfg, row, recorded = _row_of(matrix, tmp_path)
    calls = []

    def fake_run(cmd, **kw):
        finished(row, cfg)
        row.mkdir(parents=True, exist_ok=True)
        (row / "config.yaml").write_text(yaml.safe_dump(recorded))
        calls.append(cmd)
        (row / "metrics.json").write_text(json.dumps({"valid": True, "host": {"main_thread_cpu_frac": 0.95}}))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(q, "run_command", fake_run)
    monkeypatch.setattr(halfwritten.gpu, "wait_for_gpu", lambda **kw: True)
    report = q.run_matrix(str(path), str(tmp_path / "out"), log=lambda *_: None)
    assert [r["status"] for r in report] == ["valid"] and len(calls) == 1
    assert report[0]["harness"].startswith("controller at 95%")
    assert not list(row.parent.glob(f"{row.name}.*-*")) and finished(row, cfg)  # it stands
    report = q.run_matrix(str(path), str(tmp_path / "out"), log=lambda *_: None)
    assert [r["status"] for r in report] == ["done"] and len(calls) == 1


def test_rows_that_stand_invalid_give_the_queue_its_own_exit_status(monkeypatch, capsys):
    """Exit 3: every row ran, some stand invalid after their rerun; the chain notes it and goes on (a twice-invalid
    row is a reported outcome, not a reason to idle the box)."""
    import bench.queue as q
    monkeypatch.setattr(q, "run_matrix", lambda *a, **k: [
        {"status": "valid", "arm": "D", "seed": 0, "row": "r", "harness": "fence p50 132 ms (limit 100 ms)"},
        {"status": "invalid (stands)", "arm": "E", "seed": 0, "row": "r2", "problem": "invalid: I9"}])
    assert q.main(["run", "m.yaml"]) == 3
    out = capsys.readouterr().out
    assert "harness: fence p50 132 ms" in out and "invalid: I9" in out
    monkeypatch.setattr(q, "run_matrix", lambda *a, **k: [
        {"status": "invalid", "arm": "E", "seed": 0, "row": "r2", "problem": "invalid: I9"},
        {"status": "failed", "arm": "D", "seed": 1, "row": "r3"}])
    assert q.main(["run", "m.yaml"]) == 1


def test_run_command_ends_a_hung_command_and_its_process_group():
    """A row that hangs is ended as a slot step is, TERM then KILL, within the grace: subprocess.run's timeout would
    KILL the launcher at once and leave its engine, in its own session, holding the GPU."""
    import sys
    import time

    import bench.queue as q
    t0 = time.monotonic()
    p = q.run_command([sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                                              "print('up', flush=True); time.sleep(60)"], timeout_s=0.5, grace_s=0.5)
    assert p.returncode == -1 and p.stdout.startswith("timeout after") and "up" in p.stdout
    assert time.monotonic() - t0 < 10
    ok = q.run_command([sys.executable, "-c", "print('fine')"], timeout_s=10)
    assert ok.returncode == 0 and ok.stdout.strip() == "fine"


def _amendment(tmp_path, seeds=None):
    import yaml
    path = tmp_path / "amend.yaml"
    path.write_text(yaml.safe_dump({"T_s": 300.0, "snapshot_every_s": 18.75,
                                    "seeds": seeds or {"A": 3, "B": 3, "C1": 2, "D": 3, "E": 3}}))
    return path


def test_a_confirmatory_grid_plans_from_its_amendment_and_refuses_without_one(tmp_path):
    """The grid's T, snapshot interval and seeds are the pilot's (bench.amend), never the base config's placeholders:
    a matrix that requires an amendment does not plan without the file."""
    import yaml

    import bench.queue as q
    matrix = {"base": "configs/cpu/smoke.yaml", "arms": ["A", "D", "C1"], "seeds": "amendment",
              "requires_amendment": True, "amendment": str(tmp_path / "missing.yaml")}
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(matrix))
    with pytest.raises(ValueError, match="runs only under an amendment"):
        q.run_matrix(str(path), str(tmp_path / "out"), dry_run=True, log=lambda *_: None)
    report = q.run_matrix(str(path), str(tmp_path / "out"), dry_run=True, log=lambda *_: None,
                          amendment=str(_amendment(tmp_path)))
    assert sorted((r["arm"], r["seed"]) for r in report) == [("A", 0), ("A", 1), ("A", 2), ("C1", 0), ("C1", 1),
                                                            ("D", 0), ("D", 1), ("D", 2)]
    assert all("budget_s=300.0" in r["set"] and "snapshot_every_s=18.75" in r["set"] for r in report)
    with pytest.raises(ValueError, match="names no seeds for"):
        q.run_matrix(str(path), str(tmp_path / "out"), dry_run=True, log=lambda *_: None,
                     amendment=str(_amendment(tmp_path, {"A": 1})))


def test_the_committed_grids_plan_under_a_synthetic_amendment(tmp_path):
    from pathlib import Path

    from bench.queue import run_matrix
    root = Path(__file__).resolve().parents[1]
    for m in ("gsm8k_main.yaml", "math_main.yaml"):
        report = run_matrix(str(root / "configs" / "matrix" / m), str(tmp_path / m), dry_run=True,
                            log=lambda *_: None, amendment=str(_amendment(tmp_path)))
        assert report and all(r["status"] == "planned" for r in report), m
        assert len({r["row"] for r in report}) == len(report), m


def test_snapshots_are_pruned_only_after_a_complete_evaluation(tmp_path, monkeypatch):
    """With `prune: true`, a row's snapshot weights go once every mark is evaluated on the row's split, the budget's
    (tend) kept: a grid's rows would otherwise fill the box (about 20 GB a row)."""
    import json
    import subprocess

    import yaml

    import bench.queue as q
    import halfwritten.gpu
    from halfwritten.launch import finished
    matrix, path = _matrix(tmp_path, evaluate=True, prune=True)
    cfg, row, recorded = _row_of(matrix, tmp_path)

    def fake_run(cmd, **kw):
        if "bench.evaluate" in cmd:  # the evaluation: every mark on dev
            (row / "evals.jsonl").write_text("".join(json.dumps(e) + "\n" for e in [
                {"tag": "init", "snapshot": "init", "split": "dev", "accuracy": 0.5},
                {"tag": "t01", "snapshot": "t01-v4", "split": "dev", "accuracy": 0.55},
                {"tag": "tend", "snapshot": "tend-v9", "split": "dev", "accuracy": 0.6}]))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        finished(row, cfg)
        for tag in ("t01-v4", "tend-v9"):
            (row / "work" / tag).mkdir(parents=True, exist_ok=True)
            (row / "work" / tag / "model.safetensors").write_bytes(b"w" * 64)
        (row / "config.yaml").write_text(yaml.safe_dump(recorded))
        (row / "snapshots.json").write_text(json.dumps({"marks": [{"tag": "t01", "staged": 4},
                                                                  {"tag": "tend", "staged": 9}]}))
        (row / "metrics.json").write_text(json.dumps({"valid": True, "host": {"main_thread_cpu_frac": 0.3}}))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(q, "run_command", fake_run)
    monkeypatch.setattr(halfwritten.gpu, "wait_for_gpu", lambda **kw: True)
    report = q.run_matrix(str(path), str(tmp_path / "out"), log=lambda *_: None)
    assert report[0]["status"] == "valid" and report[0]["evaluated"] and report[0]["pruned"] == 1
    assert not (row / "work" / "t01-v4" / "model.safetensors").exists()
    assert json.loads((row / "work" / "t01-v4" / "pruned.json").read_text())["bytes"] == 64
    assert (row / "work" / "tend-v9" / "model.safetensors").exists()


def test_the_grid_is_cut_only_at_a_seed_blocks_boundary_by_the_stop_file_or_the_deadline(tmp_path):
    """RUN.md §3: touching <out>/_queue/STOP, or the queue's --deadline-h, stops the grid at the next seed block;
    the rows not run are reported as such, nothing fails, and the exit status is 0."""
    import yaml

    from bench.queue import FAILED, STANDS, run_matrix
    matrix = tmp_path / "m.yaml"
    matrix.write_text(yaml.safe_dump({"base": "configs/cpu/smoke.yaml", "arms": ["D", "E"], "seeds": [0, 1]}))
    out = tmp_path / "out"
    planned = run_matrix(str(matrix), str(out), dry_run=True, log=lambda *_: None)
    assert [r["status"] for r in planned] == ["planned"] * 4
    (out / "_queue" / "STOP").touch()
    cut = run_matrix(str(matrix), str(out), dry_run=True, log=lambda *_: None)
    assert [r["status"] for r in cut] == ["not run (cut)"] * 4 and all("STOP" in r["why"] for r in cut)
    assert not any(r["status"] in FAILED or r["status"] in STANDS for r in cut)
    (out / "_queue" / "STOP").unlink()
    late = run_matrix(str(matrix), str(out), dry_run=True, log=lambda *_: None, deadline_h=0.0)
    assert [r["status"] for r in late] == ["not run (cut)"] * 4 and "deadline" in late[0]["why"]
