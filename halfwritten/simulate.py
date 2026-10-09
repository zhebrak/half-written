"""One simulated run: the real controller, FakeEngine and SimTrainer configured from RunConfig.sim, in virtual time.

Used by `python -m halfwritten sim`, the regime calculator (bench/calculator.py) and the tests. The engine's cost
model and the trainer's step time are the calibration a GPU session measures (bench/calibrate.py), so with them the
same code predicts a regime's throughput, idle time and staleness for any length distribution. configure() names a
regime's config, table_row() and table() summarise a simulated run, for both entry points.
"""
import asyncio
from dataclasses import dataclass, replace

import numpy as np

from .clock import run_sim
from .config import RunConfig
from .controller import Controller, RunSummary
from .engines.fake import CostModel, FakeEngine
from .lengths import LengthModel
from .regimes import parse_arm, resolve
from .rewards import sim_reward
from .tasks import synthetic
from .trace import Event, Tracer, lane_fractions, merge
from .trainer.client import SimTrainer


@dataclass
class SimResult:
    summary: RunSummary
    controller: Controller
    engine: FakeEngine
    events: list[Event]  # controller and trainer lanes, merged


def simulate(cfg: RunConfig, engine_cls=FakeEngine, n_prompts: int = 1024, journal=None) -> SimResult:
    cfg.validate()
    sim = cfg.sim
    lora = resolve(cfg.regime, cfg.lora).weights != "full"

    async def main():
        loop = asyncio.get_running_loop()
        lengths = LengthModel(mean=sim.length_mean, sigma=sim.length_sigma, seed=cfg.seed,
                              empirical=tuple(sim.empirical_lengths) if sim.empirical_lengths else None)
        engine = engine_cls(lengths=lengths, cost=CostModel(**sim.cost), max_num_seqs=sim.max_num_seqs,
                            kv_tokens=sim.kv_tokens, max_model_len=cfg.max_model_len,
                            max_num_batched_tokens=sim.max_num_batched_tokens, delivery=sim.delivery,
                            submit_latency=sim.submit_latency, stats_lag=sim.stats_lag,
                            async_scheduling=sim.async_scheduling, silent_waiting_aborts=sim.silent_waiting_aborts)
        trainer_tracer = Tracer(loop.time, "trainer")
        trainer = SimTrainer(engine, trainer_tracer, s_fixed=sim.train_s_fixed, s_per_token=sim.train_s_per_token,
                             push_s=sim.push_s, lora=lora,
                             publication=cfg.publication, eps_max=cfg.eps_max, send_slowdown=sim.send_slowdown,
                             seed=cfg.seed)
        controller = Controller(cfg, engine, trainer, synthetic(n_prompts, sim.prompt_len, seed=cfg.seed),
                                sim_reward(sim.p_correct, cfg.seed, forced_lengths=bool(cfg.forced_lengths)),
                                journal=journal)
        summary = await controller.run()
        trainer_events = trainer_tracer.close()
        summary.lanes.update({lane: lane_fractions(trainer_events, lane) for lane in ("trainer", "sender")})
        return SimResult(summary, controller, engine, merge(controller.tracer.events, trainer_events))

    return run_sim(main())


def configure(base: RunConfig, regime: str) -> RunConfig:
    """The base config for one regime: A runs synchronously (K=0, S=1); F and +lora names run LoRA."""
    name, lora = parse_arm(regime)
    lora = lora or name == "F"
    if name == "A":
        return replace(base, regime=name, lora=lora, K=0, S=1).validate()
    return replace(base, regime=name, lora=lora).validate()


def table_row(regime: str, result) -> dict:
    s, m = result.summary, result.summary.metrics()
    lags = [x["lag_mean"] for x in s.steps_log if x.get("lag_mean") is not None]
    engine = s.lanes["engine"]
    return {"regime": regime, "valid": s.valid, "steps": s.steps, "steps/h": round(m["steps_per_hour"]),
            "goodput tok/s": round(m["goodput_tokens_per_s"]), "gen tok/s": round(m["tokens_per_s"]),
            "in flight": round(s.inflight_mean, 1),
            "draining %": round(100 * engine.get("draining", 0)),
            "paused %": round(100 * sum(engine.get(k, 0) for k in ("pausing", "paused", "receiving"))),
            "trainer busy %": round(100 * s.lanes["trainer"].get("fwd_bwd", 0)),
            "pause p50 ms": round(1000 * m["pause_s"].get("p50", 0), 1),
            "lag mean": round(float(np.mean(lags)), 2) if lags else None, "lag max": m["lag_max"],
            "depth max": m["depth_max"], "wasted": s.tokens["discarded"], "re-prefilled": s.tokens["reprefilled"]}


def table(rows: list[dict]) -> str:
    keys = list(rows[0])
    width = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    lines = [" ".join(k.rjust(width[k]) for k in keys)]
    lines += [" ".join(str(r[k]).rjust(width[k]) for k in keys) for r in rows]
    return "\n".join(lines)
