"""The simulator's tools: length specs, the regime calculator and its CLI, A2's timing emulation, replay of a row."""
import pytest
import yaml

from halfwritten.__main__ import main
from halfwritten.config import RunConfig, SimSettings
from halfwritten.engines.fake import CostModel
from halfwritten.lengths import sim_fields
from halfwritten.results import row_dir, write_row
from halfwritten.simulate import configure, simulate


def test_lengths_spec():
    assert sim_fields("lognormal:300,1.1") == {"length_mean": 300.0, "length_sigma": 1.1}
    with pytest.raises(ValueError):
        sim_fields("uniform:3")


def test_configure_makes_A_synchronous_and_F_lora():
    base = RunConfig(K=4, S=2)
    assert (configure(base, "A").K, configure(base, "A").S) == (0, 1)
    assert configure(base, "F").lora and configure(base, "E+lora").lora and not configure(base, "E").lora


def test_sim_cli_writes_a_valid_row_per_regime(tmp_path, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"B": 4, "G": 4, "M": 4, "K": 2, "max_tokens": 256, "budget_s": 5.0,
                                   "sim": {"prompt_len": 16, "length_mean": 60}}))
    assert main(["sim", str(cfg), "--regimes", "A,D,F", "--out", str(tmp_path / "results")]) == 0
    out = capsys.readouterr().out
    assert "regime" in out and "True" in out
    rows = sorted(p.parent.name for p in (tmp_path / "results").rglob("metrics.json"))
    assert rows == ["A-B4G4M4-K0-S1-s0", "D-B4G4M4-K2-S1-s0", "F-B4G4M4-K2-S1-s0"]


def test_emulated_A_reproduces_a_simulated_A_row_and_A2_is_faster(tmp_path):
    """On a row whose engine is the fake with a known cost model, re-timing A from its own answers must land close to
    its measured wall clock; A2 (two engines, halved training) must come out faster per step."""
    from dataclasses import asdict

    from bench.emulate import emulate
    cost = CostModel(alpha=2e-3, beta=2e-5, gamma=1e-8, delta=5e-6)
    cfg = configure(RunConfig(B=4, G=4, M=4, K=2, budget_s=8.0, max_tokens=256,
                              sim=SimSettings(prompt_len=16, length_mean=80, length_sigma=1.0, cost=asdict(cost),
                                              train_s_per_token=1e-4)), "A")
    r = simulate(cfg, n_prompts=64)
    path = write_row(tmp_path / "A", cfg, r.summary, r.controller.ledger, r.events, env={})
    e = emulate(path, cost, allreduce_s=0.0, kv_tokens_per_engine=1000)
    # the last batch may be consumed with its step cut by the budget: it is timed up to the budget
    assert len(e["steps"]) in (r.summary.steps, r.summary.steps + 1)
    assert abs(e["emulated_A_wall_s"] - e["measured_wall_s"]) / e["measured_wall_s"] < 0.05  # PLAN.md's +-5%
    assert all(x["A2"] < x["A"] for x in e["steps"])
    assert e["a2_assumptions"]["kv_ok"] is False  # 8 answers x 4096 tokens cannot fit in 1,000 KV tokens


@pytest.mark.parametrize("regime", ["A", "D"])
def test_replay_reproduces_a_simulated_row(regime, tmp_path):
    """A row written by the simulator, replayed from what the row recorded (lengths, trainer spans, pushes) on the
    same engine cost model, must come back within a few percent: the replay's own error, before any real engine."""
    from bench.replay import replay
    from halfwritten.config import RunConfig, SimSettings
    from halfwritten.results import write_row
    from halfwritten.simulate import configure, simulate
    cost = {"alpha": 4e-3, "beta": 2e-5, "gamma": 2e-8, "delta": 3e-6}
    cfg = configure(RunConfig(regime="D", B=8, G=4, M=8, K=2, max_tokens=256, budget_s=120.0,
                              sim=SimSettings(prompt_len=48, length_mean=90, length_sigma=0.6, cost=cost,
                                              train_s_fixed=0.4, train_s_per_token=1e-4, push_s=0.15)), regime)
    result = simulate(cfg)
    row = row_dir(cfg, "sim", tmp_path)
    write_row(row, cfg, result.summary, result.controller.ledger, result.events)
    r = replay(row, cost)
    assert r["valid"]
    assert abs(r["fitted"]["train_s_fixed"] - 0.4) < 0.02 and abs(r["fitted"]["train_s_per_token"] - 1e-4) < 5e-6
    for name in ("steps/h", "gen tok/s"):
        assert abs(r["comparison"][name]["rel_error"]) < 0.08, (name, r["comparison"][name])
