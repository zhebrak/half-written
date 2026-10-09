"""Row: the one reader of a result row's files (review-2026-10-02 plan, section 2.4)."""
import json

from halfwritten.config import RunConfig, SimSettings
from halfwritten.results import Row, row_dir, row_problem, rows, set_aside, write_row
from halfwritten.simulate import configure, simulate


def _sim_row(root, regime="D", tag=""):
    cfg = configure(RunConfig(B=4, G=4, M=4, K=2, budget_s=3.0, tag=tag,
                              sim=SimSettings(prompt_len=16, length_mean=60)), regime)
    r = simulate(cfg, n_prompts=64)
    return write_row(row_dir(cfg, "sim", root), cfg, r.summary, r.controller.ledger, r.events, env={"gpus": []})


def test_a_row_reads_every_file_once_and_only_when_asked(tmp_path):
    path = _sim_row(tmp_path)
    row = Row(path)
    assert row.finished and row.valid and row.problem(harness=False) is None
    assert row.config["regime"] == "D" and row.config["K"] == 2
    assert "commit" in row.environment or row.environment == {"gpus": []}
    assert isinstance(row.identity, dict)
    assert row.metrics["steps"] >= 1 and row.steps and row.syncs and row.ledger and row.trace
    assert row.evals() == [] and row.snapshots == {} and row.emulated is None
    assert row.states() and row.trainer_spans() and all(s > 0 for s in row.trainer_spans())
    assert row.stale_kv_share(after_s=0.0) is not None
    assert row.gpu  # the directory's name, with no recorded GPU
    (path / "evals.jsonl").write_text(json.dumps({"tag": "init", "accuracy": 0.5}) + "\n"
                                      + json.dumps({"tag": "tend", "accuracy": 0.6, "split": "dev"}) + "\n")
    assert len(Row(path).evals()) == 2 and [e["tag"] for e in Row(path).evals("test")] == ["init"]
    assert [e["tag"] for e in Row(path).evals("dev")] == ["tend"]
    # a problem row: not valid, then excluded for its harness only when it recorded one
    m = row.metrics | {"valid": False, "error": "ControllerError: I9"}
    (path / "metrics.json").write_text(json.dumps(m))
    assert Row(path).problem().startswith("invalid: ControllerError")
    m = row.metrics | {"valid": True, "host": {"main_thread_cpu_frac": 0.9}}
    (path / "metrics.json").write_text(json.dumps(m))
    assert Row(path).problem().startswith("harness: controller at 90%") and Row(path).problem(harness=False) is None
    assert row_problem(Row(path).metrics) is None  # the queue keeps it: the analysis applies the harness rule
    m = row.metrics | {"valid": True}
    (path / "metrics.json").write_text(json.dumps(m))
    assert Row(path).problem(harness=False) is None and "no harness record" in Row(path).problem()
    assert row_problem(Row(path).metrics) is None  # no record: a rerun could not mend it, so the queue keeps it


def test_rows_skips_sessions_prep_and_rows_set_aside(tmp_path):
    a = _sim_row(tmp_path, "D")
    b = _sim_row(tmp_path, "E")
    aside = _sim_row(tmp_path, "D", tag="x")
    set_aside(aside, "invalid")
    (tmp_path / "_sessions" / "x").mkdir(parents=True)
    (tmp_path / "_sessions" / "x" / "metrics.json").write_text("{}")
    (tmp_path / "prep" / "gsm8k").mkdir(parents=True)
    (tmp_path / "prep" / "gsm8k" / "metrics.json").write_text("{}")
    assert [r.path for r in rows(tmp_path)] == sorted([a, b])
