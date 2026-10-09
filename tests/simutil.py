"""Short simulated runs for the tests: 4 groups of 4 answers, K=2, a 5 s budget, unless overridden. Keyword
arguments naming a SimSettings field (stats_lag, submit_latency, ...) configure the fake engine and trainer; the rest
go to RunConfig."""
from dataclasses import asdict, fields

from halfwritten.config import RunConfig, SimSettings
from halfwritten.engines.fake import CostModel, FakeEngine
from halfwritten.simulate import simulate

_SIM = {f.name for f in fields(SimSettings)}


def quick_sim(regime: str, *, lora: bool = False, mean: float = 60, sigma: float = 0.9, max_num_seqs: int = 64,
             engine_cls=FakeEngine, s_per_token: float = 1e-4, cost: CostModel = CostModel(), journal=None, **kw):
    """Returns (summary, controller, engine, trainer events)."""
    synchronous = regime.removesuffix("+lora") == "A"
    lora = lora or regime == "F" or regime.endswith("+lora")
    sim = SimSettings(**{"prompt_len": 16, "length_mean": mean, "length_sigma": sigma, "max_num_seqs": max_num_seqs,
                         "cost": asdict(cost), "train_s_per_token": s_per_token,
                         **{k: v for k, v in kw.items() if k in _SIM}})
    base = dict(B=4, G=4, M=4, K=0 if synchronous else 2, S=1, max_tokens=256, budget_s=5.0)
    cfg = {k: v for k, v in kw.items() if k not in _SIM}
    config = RunConfig(regime=regime.removesuffix("+lora"), lora=lora, sim=sim, **{**base, **cfg})
    r = simulate(config, engine_cls=engine_cls, n_prompts=64, journal=journal)
    return r.summary, r.controller, r.engine, [e for e in r.events if e.proc == "trainer"]


def ledger_matches_truth(controller, engine) -> list[str]:
    """Every closed answer's ledger versions and stale flags against the fake engine's ground truth."""
    from bench.canary import compare
    report = compare(controller.ledger, engine.truth)
    return [] if report["ok"] else report["examples"] or [str(report)]
