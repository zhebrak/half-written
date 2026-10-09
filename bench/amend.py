"""The amendment: what a task's pilot sets for its confirmatory grid, machine-written under the pre-registered rules
(HYPOTHESES.md decisions 2, 4, 9, 12 and 13; "Amendments"), for the owner to commit at the tag.

    python -m bench.amend gsm8k --pilot results/gsm8k --one-gpu results/gsm8k_one_gpu --map results/sim \\
        --out configs/amendments/gsm8k.yaml [--delta 0.02] [--n-test 1319] [--budget-h 9]

Read: the two-GPU pilot rows (tag "pilot": D x 2, whose curves pooled give T; A, whose steps/h place step_T), the
one-GPU A x 6 (tag "pilot-1gpu": the within-arm spread at the snapshot nearest step_T sets the seed count), and the
systems map's cell on the task's W0 lengths (the best async arm among B, D and E). Written: T (the time at which D
reaches 80% of its fitted gain, rounded up to the minute), T/16 snapshots, the seeds per arm, h4's membership
(GSM8K: out, decision 4; MATH: by D's stale-KV exposure), the best async arm, delta, the equivalence families the
analysis may claim (decision 12: MATH past 30 minutes keeps D-E only) and whether the seed cap makes the claims
intervals only (decision 13), and the inputs' digests.

The pre-registered rules have stated fallbacks, applied and recorded under `warnings` rather than stopping the box
(the owner's policy, 2026-10-02: run, then decide): a fitted t80 beyond 0.8 x the pilot's horizon or a gain below
MIN_GAIN sets T to the horizon; step_T past the one-GPU runs' last evaluated step reads their last common
snapshot; more box time than the stage was given is recorded, never traded for seeds; a pilot row flagged for its
harness is read and named. A hard stop (AmendmentStop, exit status 2) is left for inputs nothing can be computed
from: no pilot rows or evaluations, fewer than two one-GPU runs, no cell to rank the arms on.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

from bench.stats import (
    EQUIVALENCE,
    MIN_EXPOSURE,
    MIN_GAIN,
    SEED_CAP,
    at_budget,
    curve,
    seed_count,
    snapshot_steps,
    start_accuracy,
    tost_power,
)
from halfwritten.results import HARNESS_LIMITS, Row, harness_record, rows

# beyond T, per row: engine start, warm-up, the row's write, its evaluation. Measured on 2026-10-02's pilot: a
# 600 s row took 653 s in the queue and its 17 evaluations 176 s, about 4 minutes in all (the 10 first assumed
# would have had the budget check fire inside the seed range the rules allow)
ROW_OVERHEAD_S = 4 * 60
GSM8K_FIXED = {"C1": 2}  # the consistency check's seeds (PLAN.md §4.5)
HARNESS_LIMIT_FACTOR = 1.5  # decision 7's limits at the tag: max(the code's, this x the highest pre-tag value)
HARNESS_LIMIT_RULE = "each limit = max(the code's limit, 1.5 x the highest pre-tag value of any arm)"


class AmendmentStop(RuntimeError):
    """Inputs nothing can be computed from (no rows, no evaluations): the chain records it and skips the grid."""


def fit_gain(points: list[tuple[float, float]], start: float) -> dict:
    """a(t) = start + gain (1 - e^(-t/tau)) fitted to (t seconds, accuracy) points: the plateau, the gain, tau and
    t80, the time at which 80% of the gain is reached (decision 2's rule)."""
    from scipy.optimize import curve_fit
    t = np.array([p[0] for p in points], dtype=float)
    a = np.array([p[1] for p in points], dtype=float)
    if len(points) < 3 or t.max() <= 0:
        raise AmendmentStop(f"too few points to fit a curve ({len(points)})")

    def f(x, gain, tau):
        return start + gain * (1 - np.exp(-x / tau))

    (gain, tau), _ = curve_fit(f, t, a, p0=[max(a.max() - start, 1e-3), max(t.max() / 3, 1.0)],
                               bounds=([0.0, 1e-3], [1.0, 1e6]), maxfev=20000)
    rmse = float(np.sqrt(np.mean((f(t, gain, tau) - a) ** 2)))
    return {"start": float(start), "gain": float(gain), "plateau": float(start + gain), "tau_s": float(tau),
            "t80_s": t_fraction({"tau_s": tau}, 0.8), "rmse": rmse, "points": len(points)}


def t_fraction(fit: dict, frac: float) -> float:
    """The time at which the fitted curve has made `frac` of its gain."""
    return float(-fit["tau_s"] * math.log(1 - frac))


def _pilot_rows(root: Path, tag: str, regime: str) -> list[Row]:
    return [r for r in rows(root) if r.config.get("tag") == tag and r.config.get("regime") == regime
            and r.problem(harness=False) is None]


def amend(task: str, pilot: Path, one_gpu: Path, systems_map: Path, *, delta: float, n_test: int,
          budget_h: float | None = None, split: str = "dev", snapshots: int = 16, time_cap_h: float | None = None,
          row_overhead_s: float = ROW_OVERHEAD_S, reserve_rows: int = 2) -> dict:
    """`snapshots`: marks per row (T / snapshots apart; 16 as pre-registered, 8 for a time-bounded run, decided at
    the tag). `time_cap_h`: the most box-hours the grid may take (a time-bounded run, 2026-10-03): the rows that fit
    at T + `row_overhead_s` each bound the seeds per arm, below the rule's count the claims become intervals only
    (decision 13's consequence of a cap), never below two; applied before the tag and recorded under `time_cap`."""
    d_rows = _pilot_rows(pilot, "pilot", "D")
    a_rows = _pilot_rows(pilot, "pilot", "A")
    if not d_rows or (not a_rows and time_cap_h is None):
        raise AmendmentStop(f"the pilot under {pilot} has {len(d_rows)} D rows and {len(a_rows)} A rows")
    # T: the pooled D curves, each snapshot at its wall-clock mark (the budget's snapshot at exactly T)
    points, starts, horizon = [], [], 0.0
    for r in d_rows:
        ev = r.evals(split)
        if not ev or at_budget(ev) is None:
            raise AmendmentStop(f"{r.path}: no complete evaluation on {split}")
        points += curve(ev, r.config["budget_s"])
        starts.append(start_accuracy(ev))
        horizon = max(horizon, float(r.config["budget_s"]))
    fit = fit_gain(points, float(np.mean(starts)))
    e_rows = _pilot_rows(pilot, "pilot", "E")
    warnings = [f"{r.path.name}: {why}" for r in [*d_rows, *a_rows, *e_rows] if (why := r.problem(harness=True))]
    if fit["gain"] < MIN_GAIN:
        warnings.append(f"D's fitted gain {fit['gain']:.3f} is below MIN_GAIN {MIN_GAIN}: no curve to set T on; "
                        f"T = the pilot's horizon {horizon:.0f} s")
        T_s = float(math.ceil(horizon / 60) * 60)
    elif fit["t80_s"] > 0.8 * horizon:
        warnings.append(f"t80 {fit['t80_s']:.0f} s lies beyond 0.8 x the pilot's horizon {horizon:.0f} s: T would "
                        f"be an extrapolation; T = the horizon")
        T_s = float(math.ceil(horizon / 60) * 60)
    else:
        T_s = float(math.ceil(fit["t80_s"] / 60) * 60)
    # the seed count, read at the one-GPU A runs' snapshot nearest the step A reaches by T (decision 13)
    a_rate = float(np.mean([r.metrics["steps_per_hour"] for r in a_rows])) if a_rows else None
    step_T = a_rate * T_s / 3600 if a_rate is not None else None
    if not a_rows:  # a time-bounded pilot without A (MATH, 2026-10-03): step_T and the seed rule need it
        warnings.append(f"no pilot A row under {pilot}: step_T is not placed; the seeds come from the time cap")
    one = _pilot_rows(one_gpu, "pilot-1gpu", "A")
    seeds, chosen = None, []
    if len(one) < 2 and time_cap_h is None:
        raise AmendmentStop(f"{len(one)} one-GPU A rows under {one_gpu}: the seed count needs two or more")
    if len(one) >= 2 and step_T is not None:
        evals = {}
        for r in one:
            ev = [e for e in r.evals(split) if snapshot_steps(e) is not None and e.get("correct")]
            if not ev:
                raise AmendmentStop(f"{r.path}: no per-item evaluation on {split}")
            evals[r] = ev
        # the snapshot read is the one nearest step_T that every run evaluated; past the runs' last common step,
        # their last common snapshot (recorded: the runs were too short for this T)
        common = set.intersection(*(set(snapshot_steps(e) for e in ev) for ev in evals.values()))
        if not common:
            raise AmendmentStop("the one-GPU runs share no evaluated snapshot step")
        read_at = min(common, key=lambda s: abs(s - step_T))
        if step_T > max(common):
            warnings.append(f"step_T {step_T:.0f} is past the one-GPU runs' last common evaluated step "
                            f"{max(common)}: the seed count is read there (run them longer next time)")
            read_at = max(common)
        correct = [next(e["correct"] for e in ev if snapshot_steps(e) == read_at) for ev in evals.values()]
        chosen = [read_at] * len(correct)
        seeds = seed_count(correct, n_test=n_test, delta=delta, cap=SEED_CAP[task])
    else:  # a time-bounded run without the one-GPU stage (2026-10-03): the cap alone sets the seeds, recorded
        warnings.append(f"{len(one)} one-GPU A rows under {one_gpu}: decision 6's seed rule is not applied; the "
                        "seeds per arm come from the time cap alone, up to decision 13's cap")
    n_rule = seeds["run"] if seeds else None
    n = n_rule if n_rule is not None else SEED_CAP[task]
    equivalence = ";".join(",".join(f) for f in EQUIVALENCE)  # both families: H4's D-E, H2's B-A, D-A, E-A
    if task == "gsm8k":
        per_arm = {"A": n, "B": n, "D": n, "E": n, **GSM8K_FIXED}
    else:  # decision 12: MATH's A and B get 5 seeds when its T is 30 minutes or less, else D-E is its only claim
        ab = 5 if T_s <= 1800 else 3
        per_arm = {"A": ab, "B": ab, "D": n, "E": n}
        if T_s > 1800:
            equivalence = "D-E"  # H2's sequence is descriptive at 3 seeds (2 df)
    # H4's membership: GSM8K is out (decision 4); MATH by D's stale-KV exposure, read before any accuracy
    shares = [s for r in d_rows if (s := r.stale_kv_share()) is not None]
    exposure = float(np.mean(shares)) if shares else None
    h4 = "out" if task == "gsm8k" else ("in" if exposure is not None and exposure >= MIN_EXPOSURE else "out")
    # the best async arm: steps/h on the systems map's cell at the task's W0 lengths, among B, D and E (decision 9)
    cells = [r for r in rows(systems_map) if r.config.get("tag") == f"{task}-w0"
             and r.config.get("regime") in ("B", "D", "E")
             and r.config.get("publication", "staged") == "staged" and r.valid]
    if not cells:
        raise AmendmentStop(f"no valid systems-map cell tagged {task}-w0 under {systems_map}: nothing ranks the arms")
    for arm in ("B", "D", "E"):
        candidates = [r for r in cells if r.config["regime"] == arm]
        if len(candidates) > 1:
            raise AmendmentStop(f"ambiguous systems-map candidates for {arm}: {[str(r.path) for r in candidates]}")
    best = max(cells, key=lambda r: r.metrics["steps_per_hour"])
    # decision 7's records and limits: every pre-tag row of every arm (the pilot's A, D and E, the cell's A, B, D
    # and E) under `harness`, and the limits the analysis applies by the rule stated at the tag (HYPOTHESES.md, the
    # 2026-10-03 addendum): each limit = max(the code's, HARNESS_LIMIT_FACTOR x the highest pre-tag value), so a
    # harness that is uniformly slower on this box excludes no arm by that alone; a revision is a recorded amendment
    recorded = [*d_rows, *a_rows, *e_rows, *cells]
    harness = {r.path.name: harness_record(r.metrics) for r in recorded}
    observed = {"max_host_cpu": "main_thread_cpu_frac", "max_loop_lag_p99_s": "loop_lag_p99_s",
                "max_fence_p50_s": "fence_p50_s"}
    harness_limits = {k: round(max(v, HARNESS_LIMIT_FACTOR * max((h[f] for h in harness.values()
                                                                   if h.get(f) is not None), default=0.0)), 4)
                      for k, v in HARNESS_LIMITS.items() for f in [observed[k]]}
    # decision 13's cap turns the equivalence claims into intervals; a time cap (below) never does: it fixes n
    # before the tag, and verdicts at a fixed n keep their error rate
    intervals_only = bool(seeds and seeds["intervals_only"])
    time_cap = None
    if time_cap_h is not None:  # a time-bounded run: the rows that fit, after a reserve for the queue's reruns,
        # set the seeds of the arms at n (the rule's, or the cap's), up to decision 13's cap, never under two
        fit_rows = int(time_cap_h * 3600 // (T_s + row_overhead_s)) - reserve_rows
        # under a cap every arm but the fixed ones (GSM8K's C1) takes the cap's seeds: decision 12's larger A and B
        # counts for MATH are a budget rule, and the cap is the budget
        at_n = [arm for arm in per_arm if arm not in GSM8K_FIXED]
        fixed_rows = sum(k for arm, k in per_arm.items() if arm not in at_n)
        run = min(SEED_CAP[task], max(0, (fit_rows - fixed_rows) // max(1, len(at_n))))
        if run < 2:
            raise AmendmentStop(f"time cap {time_cap_h} h fits {fit_rows} rows at T {T_s:.0f} s + "
                                f"{row_overhead_s:.0f} s each after a reserve of {reserve_rows}: fewer than two "
                                "seeds an arm")
        power = tost_power(run, seeds["sigma_diff_test"], delta) if seeds else None
        time_cap = {"hours": time_cap_h, "row_overhead_s": row_overhead_s, "reserve_rows": reserve_rows,
                    "rows_that_fit": fit_rows, "seeds_by_rule": n_rule, "seeds_run": run,
                    "tost_power_at_seeds_run": power}
        if n_rule is None or run != n_rule:
            warnings.append(f"time cap {time_cap_h} h fits {fit_rows} rows at T {T_s:.0f} s + {row_overhead_s:.0f} s "
                            f"each (reserve {reserve_rows}): {run} seeds an arm"
                            + (f" against the rule's {n_rule} (TOST power {power:.2f} at the rule's sigma)"
                               if n_rule is not None else " (no seed rule: the one-GPU stage did not run)")
                            + "; the verdicts stand at this n, fixed before the tag")
        per_arm = {arm: (run if arm in at_n else k) for arm, k in per_arm.items()}
    planned_rows = sum(per_arm.values())
    hours = planned_rows * (T_s + row_overhead_s) / 3600
    if budget_h is not None and hours > budget_h:
        warnings.append(f"{planned_rows} rows x (T {T_s:.0f} s + overhead) = {hours:.1f} h exceeds the stage's "
                        f"{budget_h} h: the grid runs in full (the seeds are the rule's, never cut for time)")
    return {
        "task": task, "written": __import__("datetime").date.today().isoformat(),
        "from_rows": sorted(str(r.path) for r in [*d_rows, *a_rows, *one, *cells]),
        "commit": next((r.environment.get("commit") for r in d_rows if r.environment.get("commit")), None),
        "warnings": warnings,
        "harness": harness, "harness_limits": harness_limits, "harness_limit_rule": HARNESS_LIMIT_RULE,
        "fit": fit, "T_s": T_s, "snapshot_every_s": T_s / snapshots, "snapshots": snapshots, "time_cap": time_cap,
        "a_steps_per_hour": a_rate, "step_T": step_T, "seed_snapshot_steps": chosen,
        "seeds": per_arm, "seed_count": seeds, "delta": delta, "n_test": n_test,
        # what bench.stats --amendment applies beside delta, best_async and h4: the equivalence families in its
        # --equivalence syntax (decision 12's MATH rule) and whether the seed cap turns the claims into intervals
        # (decision 13)
        "equivalence": equivalence, "intervals_only": intervals_only,
        "h4": h4, "d_stale_kv_share": exposure, "min_exposure": MIN_EXPOSURE,
        "best_async": best.config["regime"],
        "map_steps_per_hour": {r.config["regime"]: r.metrics["steps_per_hour"] for r in cells},
        "planned_rows": planned_rows, "estimated_hours": hours,
        "identity": d_rows[0].identity,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task", choices=("gsm8k", "math"))
    ap.add_argument("--pilot", required=True, help="the two-GPU pilot rows' root")
    ap.add_argument("--one-gpu", required=True, help="the one-GPU A x 6 rows' root")
    ap.add_argument("--map", required=True, help="the systems map's rows' root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=None, help="default: 0.02 for gsm8k, 0.03 for math")
    ap.add_argument("--n-test", type=int, default=None, help="default: 1319 for gsm8k, 500 for math")
    ap.add_argument("--budget-h", type=float, default=None, help="the stage's estimate: a warning if the grid "
                                                                  "would take longer")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--snapshots", type=int, default=16, help="marks per row (T / snapshots apart); 16 as "
                                                                "pre-registered, 8 for a time-bounded run")
    ap.add_argument("--time-cap-h", type=float, default=None,
                    help="the most box-hours the grid may take: bounds the seeds per arm (intervals only below it)")
    ap.add_argument("--row-overhead-s", type=float, default=ROW_OVERHEAD_S,
                    help="seconds a row costs beyond T (engine start, write, evaluation, teardown)")
    ap.add_argument("--reserve-rows", type=int, default=2, help="rows held back from the time cap for reruns")
    a = ap.parse_args(argv)
    delta = a.delta if a.delta is not None else {"gsm8k": 0.02, "math": 0.03}[a.task]
    n_test = a.n_test if a.n_test is not None else {"gsm8k": 1319, "math": 500}[a.task]
    try:
        out = amend(a.task, Path(a.pilot), Path(a.one_gpu), Path(a.map), delta=delta, n_test=n_test,
                    budget_h=a.budget_h, split=a.split, snapshots=a.snapshots, time_cap_h=a.time_cap_h,
                    row_overhead_s=a.row_overhead_s, reserve_rows=a.reserve_rows)
    except AmendmentStop as e:
        print(f"HARD STOP: {e}", file=sys.stderr)
        return 2
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(json.loads(json.dumps(out, default=float)), sort_keys=False))
    print(yaml.safe_dump({k: out[k] for k in ("T_s", "seeds", "best_async", "h4", "estimated_hours", "warnings")},
                         sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
