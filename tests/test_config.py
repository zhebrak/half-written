"""RunConfig: validation, the YAML round trip, where a run's row would go, and launch limits."""
import pytest

from halfwritten.config import RunConfig, from_dict, load, to_dict


def test_synchronous_arm_needs_K0_and_async_arms_need_a_reachable_push():
    RunConfig(regime="A", K=0, S=1).validate()
    with pytest.raises(ValueError, match="synchronous"):
        RunConfig(regime="A", K=2).validate()
    with pytest.raises(ValueError, match="K >= 1"):
        RunConfig(regime="D", K=0).validate()
    with pytest.raises(ValueError, match="S <= K\\+1"):
        RunConfig(regime="D", K=1, S=3).validate()
    with pytest.raises(ValueError, match="lora=True"):
        RunConfig(regime="F").validate()


def test_yaml_roundtrip_and_unknown_keys(tmp_path):
    import yaml
    cfg = from_dict({"regime": "E", "K": 4, "sim": {"length_mean": 300}})
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(to_dict(cfg)))
    assert load(path) == cfg
    with pytest.raises(ValueError, match="unknown config keys"):
        from_dict({"regime": "E", "batch": 3})
    with pytest.raises(ValueError, match="unknown sim keys"):
        from_dict({"regime": "E", "sim": {"x": 1}})


def test_where_prints_the_row_a_run_would_write(capsys):
    from halfwritten.__main__ import main
    assert main(["where", "configs/cpu/smoke.yaml", "--set", "tag=fp32", "--set", "seed=3", "--out", "r"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("r/sim/fake/") and out.endswith("D-B2G4M2-K2-S1-fp32-s3")


def test_lora_arms_refuse_several_api_servers():
    from halfwritten.config import RunConfig, with_overrides
    with pytest.raises(ValueError, match="api_servers"):
        with_overrides(RunConfig(regime="F", lora=True, K=2), ["launch.api_servers=4"])
    assert with_overrides(RunConfig(regime="D"), ["launch.api_servers=4"]).launch.api_servers == 4


def test_renamed_and_dropped_keys_still_load_from_old_rows():
    """An archived row's config loads as today's code reads it: renamed keys renamed; a key since dropped accepted at
    the value the code now assumes (or at any value where it had no effect), refused by name at another; and such a
    row compares equal to today's config (differences), so a rerun does not take it for another measurement."""
    from halfwritten.config import differences, to_dict
    cfg = from_dict({"regime": "D", "K": 2, "quiet_s": 0.2, "fence_min_s": 0.1, "lengths": "fixed:5",
                     "sim": {"reward_p": 0.3, "seed": 0, "adapter_push_s": 0.02}, "fence_exact": True,
                     "reset_counts_as_preemption": True, "stop": [], "launch": {"master_port": 29555}})
    assert (cfg.forced_lengths, cfg.sim.p_correct) == ("fixed:5", 0.3)
    recorded = to_dict(cfg) | {"fence_exact": True, "fence_quiet_s": 0.05, "reset_counts_as_preemption": True}
    assert differences(recorded, to_dict(cfg)) == []
    assert differences(recorded | {"fence_exact": False}, to_dict(cfg)) == ["fence_exact"]
    with pytest.raises(ValueError, match="fence_exact"):
        from_dict({"regime": "D", "K": 2, "fence_exact": False})
    with pytest.raises(ValueError, match="stop"):
        from_dict({"regime": "D", "K": 2, "stop": ["Question:"]})


def test_in_place_lora_arms_are_refused_on_vllm():
    """vLLM 0.30 applies an in-flight request's older adapter to the whole batch after an in-place reload (the probe of
    2026-10-02): no in-place LoRA arm launches on stock vLLM; with launch.lora_inplace_fix the server clears the flag.
    Simulation (the fake engine, which reloads correctly) is unaffected."""
    from halfwritten.config import LaunchSettings
    from halfwritten.launch import engine_command
    with pytest.raises(ValueError, match="in-place LoRA reloads are wrong on stock vLLM 0.30"):
        engine_command(RunConfig(regime="D", lora=True, launch=LaunchSettings(engine="vllm")).validate())
    engine_command(RunConfig(regime="F", lora=True, launch=LaunchSettings(engine="vllm")).validate())
    fixed = RunConfig(regime="D", lora=True, launch=LaunchSettings(engine="vllm", lora_inplace_fix=True)).validate()
    assert engine_command(fixed)[0][1:3] == ["-m", "halfwritten.engines.vllm_serve"]
    engine_command(RunConfig(regime="D", lora=True, launch=LaunchSettings(engine="fake")).validate())


def test_a_lora_run_from_a_warm_start_serves_the_warm_start(tmp_path, monkeypatch):
    """External review F05: the trainer loaded W0 and wrapped it in an adapter, while the engine served the original
    base plus the adapter: two policies. The engine now serves a directory holding W0's weights and the base's config
    and tokenizer, under the model's name."""
    import huggingface_hub

    from halfwritten.config import LaunchSettings
    from halfwritten.launch import engine_command
    base, w0 = tmp_path / "snapshot", tmp_path / "w0"
    base.mkdir()
    w0.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (base / name).write_text(name)
    (w0 / "model.safetensors").write_text("W0")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: str(base))
    cfg = RunConfig(regime="F", lora=True, model="Qwen/Qwen3-0.6B-Base", init_from=str(w0),
                    launch=LaunchSettings(engine="vllm")).validate()
    argv, _, served = engine_command(cfg, base_dir=tmp_path / "row" / "engine-base")
    where = tmp_path / "row" / "engine-base"
    assert argv[argv.index("serve") + 1] == str(where) and argv[argv.index("--served-model-name") + 1] == served
    assert (where / "model.safetensors").read_text() == "W0" and (where / "config.json").read_text() == "config.json"
    assert "--revision" not in argv  # a local directory, made of pinned files
    with pytest.raises(ValueError, match="base_dir"):
        engine_command(cfg)
