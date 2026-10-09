"""Result rows: labels and paths, the files a row holds, launch commands per layout, the report and GPU labels."""
import json

import pyarrow.parquet as pq
import pytest
import yaml

from halfwritten.config import RunConfig, SimSettings
from halfwritten.results import gpu_slug, row_dir, row_label, write_row
from halfwritten.simulate import configure, simulate


def test_labels_and_paths():
    cfg = RunConfig(regime="D", lora=True, K=4, S=1, seed=2, task="gsm8k", model="Qwen/Qwen3-0.6B-Base")
    assert row_label(cfg) == "D+lora-B32G8M32-K4-S1-s2"
    assert str(row_dir(cfg, "NVIDIA A100-SXM4-80GB", root="r")) == \
        "r/gsm8k/Qwen3-0.6B-Base/A100-SXM4-80GB/D+lora-B32G8M32-K4-S1-s2"
    # rows that differ only in concurrency or batch size, or in a tagged setting, get their own directories
    labels = {row_label(RunConfig(regime="D", M=m, B=b, tag=t)) for m, b, t in [(8, 8, ""), (16, 8, ""), (8, 16, ""),
                                                                                 (8, 8, "lognormal-1.1")]}
    assert len(labels) == 4
    assert gpu_slug(None) == "cpu"


def test_a_row_holds_every_file(tmp_path):
    cfg = configure(RunConfig(B=4, G=4, M=4, K=2, budget_s=5.0,
                              sim=SimSettings(prompt_len=16, length_mean=80, length_sigma=1.0)), "D")
    r = simulate(cfg, n_prompts=64)
    path = write_row(tmp_path / "row", cfg, r.summary, r.controller.ledger, r.events, env={"commit": "test"})
    names = {p.name for p in path.iterdir()}
    assert {"config.yaml", "metrics.json", "steps.parquet", "syncs.parquet", "ledger.parquet", "trace.json",
            "trace.parquet"} <= names
    m = json.loads((path / "metrics.json").read_text())
    assert m["valid"] and m["lanes"]["trainer"]["fwd_bwd"] > 0
    assert yaml.safe_load((path / "config.yaml").read_text())["config"]["regime"] == "D"
    assert pq.read_table(path / "steps.parquet").num_rows == len(r.summary.steps_log)
    assert json.loads((path / "trace.json").read_text())["traceEvents"]


def test_a_row_is_never_overwritten_by_a_different_config(tmp_path):
    from halfwritten.results import RowExists
    cfg = configure(RunConfig(B=4, G=4, M=4, K=2, budget_s=2.0, sim=SimSettings(prompt_len=16, length_mean=40)), "D")
    r = simulate(cfg, n_prompts=64)
    write_row(tmp_path / "row", cfg, r.summary, r.controller.ledger, r.events, env={})
    write_row(tmp_path / "row", cfg, r.summary, r.controller.ledger, r.events, env={})  # same config: a rerun
    with pytest.raises(RowExists):
        write_row(tmp_path / "row", configure(cfg, "E"), r.summary, r.controller.ledger, r.events, env={})


def test_launch_builds_the_engine_and_trainer_for_each_layout(tmp_path):
    from halfwritten.config import from_dict
    from halfwritten.launch import engine_command, trainer_client
    colocated = from_dict({"regime": "D", "model": "Qwen/Qwen3-0.6B-Base", "M": 8, "G": 8,
                           "launch": {"engine_gpu": "0", "trainer_gpu": "0", "weight_backend": "ipc"}})
    argv, env, served = engine_command(colocated)
    assert argv[0].endswith("vllm") and argv[1] == "serve" and served == "Qwen/Qwen3-0.6B-Base"
    assert '{"backend": "ipc"}' in argv and env["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and argv[argv.index("--max-num-seqs") + 1] == "64"
    t = trainer_client(colocated, "http://x", tmp_path / "row" / "work")
    assert t.sink == {"kind": "vllm", "url": "http://x", "backend": "ipc", "master_port": 29555, "world_size": 2}
    assert t.env["CUDA_VISIBLE_DEVICES"] == "0"
    split = from_dict({"regime": "E", "launch": {"engine_gpu": "0", "trainer_gpu": "1"}})
    assert '{"backend": "nccl"}' in engine_command(split)[0]
    assert trainer_client(split, "http://x", tmp_path / "w").env["CUDA_VISIBLE_DEVICES"] == "1"
    # the NCCL layout's environment on both ends (NVLS failed in the study's containers), and only there
    assert engine_command(split)[1]["NCCL_NVLS_ENABLE"] == "0"
    assert trainer_client(split, "http://x", tmp_path / "w").env["NCCL_NVLS_ENABLE"] == "0"
    assert "NCCL_NVLS_ENABLE" not in env and "NCCL_NVLS_ENABLE" not in t.env
    lora = from_dict({"regime": "F", "lora": True, "launch": {"engine_gpu": "0", "trainer_gpu": "0"}})
    adapters = trainer_client(lora, "http://x", tmp_path / "r" / "work", str(tmp_path / "adapters"))
    assert adapters.sink["kind"] == "adapter"
    assert "--enable-lora" in engine_command(lora)[0]
    with pytest.raises(ValueError, match="ipc"):
        from_dict({"regime": "D", "launch": {"engine_gpu": "0", "trainer_gpu": "0", "weight_backend": "nccl"}})



def test_gpu_labels_from_slugs_smi_names_and_sessions():
    from halfwritten.results import gpu_label
    assert gpu_label("H100-80GB-HBM3") == gpu_label("NVIDIA H100 80GB HBM3") == gpu_label("h100x2-1") == "H100 SXM"
    assert gpu_label("NVIDIA H100 NVL") == gpu_label("results/_sessions/h100-1/x") == "H100 NVL"
    assert gpu_label("NVIDIA A100-SXM4-40GB") == gpu_label("a100-1") == "A100 40GB"


def test_a_finished_row_of_another_layout_is_not_taken_for_this_one():
    """The one-GPU pilot's A seed 100 had the two-GPU pilot's A row's label; run_row returned that row as done and
    the seed never ran (2026-10-02). config.differences is the one definition of "the same measurement": ports are
    not it, GPU indices count only as the layout, a key older code did not record ran at its default, numbers
    compare by value (code review: three copies had disagreed on exactly these)."""
    from halfwritten.config import LaunchSettings, RunConfig, differences, to_dict
    two_gpu = to_dict(RunConfig(regime="A", K=0, launch=LaunchSettings(engine_gpu="0", trainer_gpu="1")))
    one_gpu = to_dict(RunConfig(regime="A", K=0, launch=LaunchSettings(engine_gpu="1", trainer_gpu="1",
                                                                       weight_backend="ipc", port=8102)))
    assert differences(two_gpu, one_gpu) == ["launch.colocated", "launch.weight_backend"]
    assert differences(two_gpu, to_dict(RunConfig(regime="A", K=0, launch=LaunchSettings(port=9000)))) == []
    swapped = to_dict(RunConfig(regime="A", K=0, launch=LaunchSettings(engine_gpu="1", trainer_gpu="0")))
    assert differences(two_gpu, swapped) == []  # which GPU index is not the measurement; the layout is
    older = dict(two_gpu)
    older["launch"] = {k: v for k, v in two_gpu["launch"].items() if k != "lora_inplace_fix"}
    del older["eval_split"]  # recorded by older code: ran at the defaults
    assert differences(older, two_gpu) == []
    assert differences(two_gpu | {"budget_s": 300}, two_gpu | {"budget_s": 300.0}) == []
    assert differences(older, two_gpu | {"eval_split": "test"}) == ["eval_split"]


def test_metrics_json_is_written_last(tmp_path, monkeypatch):
    """metrics.json marks a finished row (run_row and the queue skip one that has it), so a crash while the row is
    written must not leave it beside missing files (external review F09): it is written last, every file atomically."""
    from pathlib import Path

    from halfwritten import results
    from tests.simutil import quick_sim
    summary, controller, _, events = quick_sim("D", budget_s=2.0)
    order = []
    real = Path.replace

    def spy(self, target):
        order.append(Path(target).name)
        return real(self, target)
    monkeypatch.setattr(Path, "replace", spy)
    results.write_row(tmp_path / "row", controller.cfg, summary, controller.ledger, events, env={},
                      snapshots={"marks": [], "written": []})
    assert order[-1] == "metrics.json" and {"ledger.parquet", "trace.parquet", "snapshots.json"} <= set(order)
    assert not list((tmp_path / "row").glob(".*.tmp"))


def _finished_row(root, cfg, valid=True, identity=None):
    import yaml

    from halfwritten.config import to_dict
    from halfwritten.launch import row_dir_on_host
    path = row_dir_on_host(cfg, root, "cpu")
    path.mkdir(parents=True)
    rec = {"config": to_dict(cfg)} | ({"identity": identity} if identity is not None else {})
    (path / "config.yaml").write_text(yaml.safe_dump(rec))
    (path / "metrics.json").write_text(json.dumps({"valid": valid}))
    return path


def test_a_finished_row_counts_only_for_its_own_measurement(tmp_path):
    """External review F01: a saved row at budget 10 s and lr 0.001 was returned for a request at 20 s and 0.004, and
    an invalid row counted as done. Now: another measurement under the same label is an error naming the settings;
    an invalid row is set aside and run again once; a row from before identities is set aside."""
    from halfwritten.config import load, with_overrides
    from halfwritten.launch import finished
    a = load("configs/cpu/smoke.yaml")
    path = _finished_row(tmp_path / "x", a, identity={})
    assert finished(path, a)
    assert finished(path, with_overrides(a, ["launch.port=9999"]))  # a port is not the measurement
    with pytest.raises(ValueError, match="budget_s, train.lr"):
        finished(path, with_overrides(a, ["budget_s=20", "train.lr=0.004"]))
    bad = _finished_row(tmp_path / "y", a, valid=False, identity={})
    assert not finished(bad, a) and not bad.exists() and list(bad.parent.glob(f"{bad.name}.invalid-*"))
    again = _finished_row(tmp_path / "y", a, valid=False, identity={})
    assert finished(again, a)  # its rerun was invalid too: it stands, reported
    old = _finished_row(tmp_path / "z", a)  # no identity: from before this check
    assert not finished(old, a) and list(old.parent.glob(f"{old.name}.legacy-*"))


def test_a_changed_warm_start_behind_the_same_path_is_another_measurement(tmp_path):
    from halfwritten.config import load, with_overrides
    from halfwritten.launch import finished
    from halfwritten.results import file_sha256, row_identity
    w0 = tmp_path / "w0"
    w0.mkdir()
    (w0 / "model.safetensors").write_bytes(b"W0 weights")
    cfg = with_overrides(load("configs/cpu/smoke.yaml"), [f"init_from={w0}"])
    path = _finished_row(tmp_path / "r", cfg, identity=row_identity(cfg))
    assert finished(path, cfg) and file_sha256(w0 / "model.safetensors") == row_identity(cfg)["init_from_sha256"]
    (w0 / "model.safetensors").write_bytes(b"other weights")
    with pytest.raises(ValueError, match="init_from_sha256"):
        finished(path, cfg)
