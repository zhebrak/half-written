"""Replay a real row in the simulator and report how well the simulator predicts it (PLAN.md §6, W2: within 10-15%).

    python -m bench.replay ROW --calibration calibration.json [--out replay.json]

The simulated run uses the row's own configuration (regime, B, G, M, K, S, budget or steps) and what the row measured
about everything except the engine:
    answer lengths   drawn from the row's ledger (empirical)
    prompt length    the ledger's mean
    trainer          seconds per step = a + b * tokens, fitted to the row's fwd_bwd spans against its steps' tokens
    push             the median measured push
The engine is the calibrated FakeEngine (CostModel from bench.calibrate). The comparison is in steps per hour,
generated tokens per second, mean lag, the engine's generating fraction and the trainer's busy fraction.

On one GPU with engine and trainer colocated (configs/one_gpu), the simulator has the two on separate devices, so a
synchronous row (A: they alternate) should replay closely while an asynchronous one (they overlap) shows the
contention as error. That gap is a measurement of colocation, not a failure of the replay.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from halfwritten.config import from_dict
from halfwritten.results import Row
from halfwritten.simulate import simulate


def trainer_model(row: Path) -> tuple[float, float]:
    """(seconds fixed, seconds per token) per step, fitted to the row's fwd_bwd spans against its steps' tokens."""
    r = Row(row)
    spans = r.trainer_spans()
    steps = r.steps
    n = min(len(spans), len(steps)) - 1  # the last span is cut short by the end of the run
    x = np.array([s["tokens"] for s in steps[:n]], dtype=float)
    y = np.array(spans[:n])
    if n < 3 or np.ptp(x) == 0:
        return float(y.mean()) if n > 0 else 0.2, 0.0
    # tokens per step cluster far from 0, so the intercept is an extrapolation an outlier moves a lot: fit, drop
    # residuals beyond 3 MADs (a warm-up step, a stalled one), fit again
    b, a = np.polyfit(x, y, 1)
    r = y - (a + b * x)
    mad = np.median(np.abs(r - np.median(r))) or 1e-12
    keep = np.abs(r - np.median(r)) <= 3 * 1.4826 * mad
    if keep.sum() >= 3:
        b, a = np.polyfit(x[keep], y[keep], 1)
    return float(max(a, 0.0)), float(max(b, 0.0))


def replay_config(row: Path, cost: dict):
    r = Row(row)
    cfg = from_dict(r.config)
    ledger, syncs = r.ledger, r.syncs
    a, b = trainer_model(row)
    sim = replace(cfg.sim, empirical_lengths=[int(r["tokens"]) for r in ledger if r["tokens"]],
                  prompt_len=int(round(np.mean([r["prompt_len"] for r in ledger if r.get("prompt_len")] or [64]))),
                  max_num_seqs=cfg.M * cfg.G, train_s_fixed=a, train_s_per_token=b,
                  push_s=float(np.median([s["push_s"] for s in syncs])) if syncs else cfg.sim.push_s,
                  cost=dict(cost))
    return replace(cfg, sim=sim, snapshot_every_s=None, snapshot_every_steps=None), {"train_s_fixed": a,
                                                                                     "train_s_per_token": b}


def compare(measured: dict, predicted: dict) -> dict:
    keys = {"steps/h": "steps_per_hour", "gen tok/s": "tokens_per_s", "lag mean": "lag_mean"}
    out = {}
    for name, k in keys.items():
        m, p = measured.get(k), predicted.get(k)
        out[name] = {"measured": m, "predicted": p,
                     "rel_error": None if not m or p is None else (p - m) / m}
    for lane, state in (("engine", "generating"), ("trainer", "fwd_bwd")):
        m = (measured.get("lanes") or {}).get(lane, {}).get(state)
        p = (predicted.get("lanes") or {}).get(lane, {}).get(state)
        out[f"{lane} {state}"] = {"measured": m, "predicted": p,
                                  "rel_error": None if not m or p is None else (p - m) / m}
    return out


def replay(row: Path, cost: dict) -> dict:
    cfg, fitted = replay_config(row, cost)
    result = simulate(cfg)
    measured = Row(row).metrics
    predicted = result.summary.metrics() | {"lanes": result.summary.lanes}
    return {"row": str(row), "valid": result.summary.valid, "fitted": fitted, "cost": cost,
            "comparison": compare(measured, predicted)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("row")
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = replay(Path(a.row), json.loads(Path(a.calibration).read_text())["cost"])
    Path(a.out or Path(a.row) / "replay.json").write_text(json.dumps(r, indent=1, default=float))
    for name, c in r["comparison"].items():
        err = "" if c["rel_error"] is None else f"{100 * c['rel_error']:+.1f}%"
        print(f"{name:>18}  measured {c['measured']}  predicted {c['predicted']}  {err}")


if __name__ == "__main__":
    main()
