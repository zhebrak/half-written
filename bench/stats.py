"""The pre-registered analysis (PLAN.md §4.6, HYPOTHESES.md).

    python -m bench.stats results/gsm8k --delta 0.02 --best D [--family D-E,D-B,E-B,best-A,best-A2]

Rows: the selected experiment under the root (--tag, default untagged). Invalid or unevaluated rows are listed and
left out. An arm is named by its regime ("D", "E", ..., "+lora" for adapter arms; F remains F). It is an error
if two rows share an arm and seed, if a name matches two configurations, if a row has no snapshot at the budget,
or if evaluations differ in item count.

Endpoints, per row:
    auc          area under accuracy against wall clock over [0, T], divided by T (tend placed at exactly T): the
                 primary endpoint (HYPOTHESES.md decision 1)
    accuracy_T   held-out accuracy of the snapshot at the budget ("tend"): the first secondary
    time_to      wall clock at which accuracy first reaches W0 + 90% of the gain arm A's runs made by T (linear
                 between marks; gain_threshold)
    at_steps(n)  accuracy after n optimiser steps, linear between snapshots by the version each one holds

Confirmatory inference is paired over seed blocks, with t intervals on n - 1 degrees of freedom:
    family       D-E, D-B, E-B and best-A, where "best" is named on the command line. It must be chosen from the
                 systems map's steps/hour ranking (H1) before any accuracy is seen. best-A2 can be named too; it is
                 descriptive in the pre-registration (A2's timing is emulated).
    estimate     The mean paired difference in the primary endpoint, its 95% t interval, and the two-sided paired-t
                 p-value, Holm-adjusted over the whole family. A contrast that cannot be computed is reported as
                 missing and still counts in Holm's m, with p = 1.
    equivalence  TOST at delta: the 90% t interval lies inside (-delta, +delta).
    H2           Per sample. For B, D and E against A, compare accuracy at the step count both arms reached in that
                 seed block (the smaller final count), with TOST at delta.
    best-A2      A2 is A re-timed (bench.emulate writes emulated.json into each A row). A2's accuracy at T is S1's
                 (A's) accuracy at the step count A2 reaches by T. It is missing if A's run stopped short of that
                 count; A rows can be run longer to reach it (per_arm budget_s), at a cost in GPU hours.
Sensitivity analyses are reported but not tested:
    - a BCa 95% interval;
    - the exact sign-flip p-value over all 2^n flips (its floor is 2/2^n: 0.25 at 3 seeds, 0.0625 at 5);
    - an item-level bootstrap over seeds and shared items;
    - IQM per arm.
With 3 to 5 seeds, bootstrap intervals and p-values are anti-conservative: in simulation, BCa coverage was 0.74 at
n = 3 and the bootstrap p-value's false-positive rate was 0.29. This is why the t analysis is confirmatory.
"""
import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from halfwritten.config import NOT_THE_MEASUREMENT, REGIME_KEYS, differences
from halfwritten.results import Row, rows
from halfwritten.types import BUDGET_MARK, parse_snapshot, snapshot_name

FAMILY = ["D-E", "D-B", "E-B", "best-A"]  # A2 is descriptive (HYPOTHESES.md); name "best-A2" to compute it


class AnalysisError(ValueError):
    pass


# -- endpoints

def snapshot_steps(e: dict) -> float | None:
    """Optimiser steps in the evaluated weights: the version in the snapshot's name ("t03-v17" holds 17), else the
    mark's step count; the run's starting weights are 0."""
    if e.get("tag") == "init":
        return 0.0
    _, version = parse_snapshot(e.get("snapshot", ""))
    if version is not None:
        return float(version)
    return float(e["steps"]) if e.get("steps") is not None else None


def at_budget(evals: list[dict]) -> dict | None:
    """The record of the snapshot at the budget (BUDGET_MARK), the last if several; None without one."""
    end = [e for e in evals if e.get("tag") == BUDGET_MARK]
    return end[-1] if end else None


def curve(evals: list[dict], T: float | None = None) -> list[tuple[float, float]]:
    """(t, accuracy) by wall clock; with T, the snapshot at the budget is placed at exactly T (its mark is taken a
    fraction of a millisecond after the budget fires)."""
    pts = [((T if (T is not None and e.get("tag") == BUDGET_MARK) else e["t"]), e["accuracy"]) for e in evals
           if e.get("t") is not None]
    return sorted(pts)


def accuracy_at_T(evals: list[dict]) -> float | None:
    end = at_budget(evals)
    return end["accuracy"] if end else None


def auc(evals: list[dict], T: float) -> float | None:
    pts = [p for p in curve(evals, T) if p[0] <= T * (1 + 1e-6)]
    if len(pts) < 2 or T <= 0:
        return None
    t, a = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    return float(np.trapezoid(a, t) / T)


def _first_crossing(pts: list[tuple[float, float]], threshold: float) -> float:
    """The x at which a (x, y) curve first reaches `threshold`, linear between points; inf if it never does."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 >= threshold:
            return x0
        if y1 >= threshold:
            return x0 + (threshold - y0) / (y1 - y0) * (x1 - x0)
    return pts[-1][0] if pts and pts[-1][1] >= threshold else math.inf


def time_to(evals: list[dict], threshold: float, T: float | None = None) -> float:
    """Wall clock at which accuracy first reaches `threshold` (the time-to-threshold secondary)."""
    return _first_crossing(curve(evals, T), threshold)


def start_accuracy(evals: list[dict]) -> float | None:
    """The accuracy of the run's starting weights (W0, the "init" record bench.evaluate writes first)."""
    init = [e for e in evals if e.get("snapshot") == "init" or e.get("tag") == "init"]
    return init[-1]["accuracy"] if init else None


MIN_GAIN = 0.03  # about 3 paired item SEs on GSM8K test (S1): a smaller gain makes the target a coin toss


def gain_threshold(rows: list[dict], frac: float, min_gain: float = MIN_GAIN) -> float | None:
    """The time-to-threshold target: W0's accuracy plus `frac` of the gain A's runs made by T; None without A rows,
    or when A gained less than min_gain (a target set by noise would rank arms by noise). A fraction of A's accuracy
    (the first draft's 90%) would fall below W0's own once W0 is a trained warm start (S1: W0 at step 50 scores
    0.594), and every arm would "reach" it at t = 0."""
    a = [r for r in rows if r["arm"] == "A"]
    ref = [x for r in a if (x := accuracy_at_T(r["evals"])) is not None]
    w0 = [x for r in a if (x := start_accuracy(r["evals"])) is not None]
    if not ref or not w0:
        return None
    start, gain = float(np.mean(w0)), float(np.mean(ref)) - float(np.mean(w0))
    return start + frac * gain if gain >= min_gain else None


def final_steps(evals: list[dict]) -> float | None:
    end = at_budget(evals)
    return snapshot_steps(end) if end else None


def at_steps(evals: list[dict], n: float) -> float | None:
    """Accuracy after n optimiser steps, linear between snapshots; None beyond the last one."""
    pts = sorted({(v, e["accuracy"]) for e in evals if (v := snapshot_steps(e)) is not None})
    if not pts or n > pts[-1][0] + 1e-9 or n < pts[0][0] - 1e-9:
        return None
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    return float(np.interp(n, xs, ys))


def steps_to(evals: list[dict], threshold: float) -> float:
    """Optimiser steps at which accuracy first reaches `threshold`, linear between snapshots; inf if it never does
    (decision 3's steps-domain secondary: how much training, not time, each arm needs)."""
    return _first_crossing(sorted({(v, e["accuracy"]) for e in evals if (v := snapshot_steps(e)) is not None}),
                           threshold)


# -- rows

def config_mismatch(configs: dict[str, dict], ignore: tuple[str, ...]) -> list[str]:
    """The settings, outside `ignore`, on which any row's config differs from the first's (config.differences: the
    same definition reruns use)."""
    first, *rest = list(configs.values()) or [{}]
    return sorted({k for c in rest for k in differences(first, c, ignore)})


def mark_problem(row: Path | Row, evals: list[dict]) -> str | None:
    """Why a row's curve is incomplete, or None. Every scheduled mark (init, each wall-clock mark, tend) must be
    evaluated, each from the version staged at its mark (external review F02); a mark with no snapshot invalidates
    the row (run it again), one with a snapshot but no evaluation is an AnalysisError (evaluate it). The area is
    never bridged across a gap."""
    if not any(e.get("tag") == "init" for e in evals):
        raise AnalysisError(f"{row}: no evaluation of its starting weights (init): run bench.evaluate")
    snaps = (row if isinstance(row, Row) else Row(row)).snapshots
    written = {str(p).rsplit("/", 1)[-1] for _, p, *_ in snaps.get("written", []) if p}
    by_tag = {e.get("tag"): e for e in evals}
    for mark in snaps.get("marks", []):
        tag, name = mark["tag"], snapshot_name(mark["tag"], mark["staged"])
        if tag not in by_tag:
            if not any(w.startswith(f"{tag}-v") for w in written):
                return f"no snapshot at mark {tag}: the row is incomplete, run it again"
            raise AnalysisError(f"{row}: snapshot {tag} has no evaluation: run bench.evaluate")
        if by_tag[tag]["snapshot"] != name:
            return f"snapshot {by_tag[tag]['snapshot']} is not the version staged at mark {tag} ({name})"
    from bench.evaluate import evaluation_problem
    path = row.path if isinstance(row, Row) else row
    if why := evaluation_problem(path, evals[0].get("split", "test"), records=evals):
        raise AnalysisError(f"{row}: {why}: run bench.evaluate")
    return None


SEED_CAP = {"gsm8k": 12, "math": 8}  # decision 13 (owner, 2026-10-02): the most seeds an arm gets per task


def seed_count(correct: list[str], n_test: int, delta: float, alpha: float = 0.05, power: float = 0.8,
               ucb: float = 0.8, max_seeds: int = 50, cap: int | None = None) -> dict:
    """Decision 13: the seeds a TOST at `delta` needs on test, from one arm's runs on dev (A x k, the per-item
    correctness strings of the endpoint's snapshot). A paired difference's variance has a seed part and an item part;
    dev has fewer items than test, so its item part is too large:
      sigma_diff_dev^2  = 2 s^2, s the SD of the runs' accuracies (a difference of two runs)
      V_item(dev)       = mean over pairs of (discordance - diff^2) / n_dev: the item part of a pair's difference
      seed part         = max(sigma_diff_dev^2 - V_item(dev), 0), raised to its `ucb` upper bound on k - 1 df
      sigma_diff_test^2 = seed part + V_item(dev) x n_dev / n_test
    and the seeds are the fewest n with delta sqrt(n) / sigma_diff_test >= t(1 - alpha) + t(1 - (1 - power) / 2) on
    n - 1 df (a TOST's power when the true difference is 0)."""
    from scipy import stats
    k, n_dev = len(correct), len(correct[0]) if correct else 0
    if k < 2 or not n_dev or any(len(c) != n_dev or set(c) - {"0", "1"} for c in correct):
        raise AnalysisError("seed_count needs two or more runs scored on the same items")
    if n_test <= 0 or delta <= 0:
        raise AnalysisError("seed_count needs a positive test size and equivalence margin")
    acc = np.array([c.count("1") / n_dev for c in correct])
    pairs = list(itertools.combinations(range(k), 2))
    v_item = float(np.mean([(sum(a != b for a, b in zip(correct[i], correct[j])) / n_dev - (acc[i] - acc[j]) ** 2)
                            / n_dev for i, j in pairs]))
    sd2_dev = 2 * float(acc.var(ddof=1))
    seed_part = max(sd2_dev - v_item, 0.0)
    seed_ucb = seed_part * (k - 1) / float(stats.chi2.ppf(1 - ucb, k - 1))
    sigma = math.sqrt(max(0.0, seed_ucb + v_item * n_dev / n_test))
    n = next((n for n in range(2, max_seeds + 1) if sigma == 0 or delta * math.sqrt(n) / sigma >= float(
        stats.t.ppf(1 - alpha, n - 1) + stats.t.ppf(1 - (1 - power) / 2, n - 1))), None)
    # past the cap the arm runs the cap and its equivalence claims are reported as intervals only; delta is never
    # widened to fit (decision 13)
    over = cap is not None and (n is None or n > cap)
    return {"seeds": n, "run": cap if over else n, "intervals_only": over, "sigma_diff_test": sigma,
            "sigma_diff_dev": math.sqrt(sd2_dev), "v_item_dev": v_item, "seed_part": seed_part,
            "seed_part_ucb": seed_ucb, "degenerate": sigma == 0}


def tost_power(n: int, sigma: float, delta: float, alpha: float = 0.05) -> float:
    """The power of a paired TOST at `delta` with n seeds when the true difference is 0 and the paired difference's
    SD on test is `sigma` (the normal approximation to the rule seed_count inverts): 2 Phi(delta sqrt(n) / sigma -
    t(1 - alpha, n - 1)) - 1, floored at 0."""
    from scipy import stats
    if sigma <= 0:
        return 1.0
    z = delta * math.sqrt(n) / sigma - float(stats.t.ppf(1 - alpha, n - 1))
    return float(max(0.0, 2 * stats.norm.cdf(z) - 1))


def verdict(diff: float, ci90: list[float], p_adj: float, delta: float, alpha: float = 0.05) -> str:
    """One of five, as Lakens frames equivalence tests: a significant difference whose 90% interval also lies
    inside the margin is "different but within delta"; neither significant nor inside is "inconclusive", which is
    not "equivalent"."""
    significant, within = p_adj < alpha, -delta < ci90[0] and ci90[1] < delta
    if significant and within:
        return "different but within delta"
    if significant:
        return "superior" if diff > 0 else "inferior"
    return "equivalent" if within else "inconclusive"


# The equivalence claims (decision 12, owner 2026-10-02): two families, H4's D-E at natural depth alone and H2's
# per-sample claims as their own fixed sequence. Within a family each claim is tested at the full alpha until the
# first that fails; neither family blocks the other.
EQUIVALENCE = [["D-E"], ["B-A", "D-A", "E-A"]]
EQUIVALENCE_DEFAULT = ";".join(",".join(f) for f in EQUIVALENCE)  # the same, as the command line and the amendment
MIN_EXPOSURE = 0.05  # D's stale-KV share below which H4 at natural depth says nothing (decision 4)


def load_rows(root: Path, harness: bool = True, split: str = "test", tag: str = "",
              limits: dict | None = None) -> tuple[list[dict], list[dict]]:
    """(rows, excluded): every valid row under root with evaluations on `split`, and the rows left out with the
    reason. With harness, a row whose controller was the bottleneck is left out too (harness_problem, under the
    amendment's `limits` when given). The rows must form one experiment: one configuration and identity apart from
    REGIME_KEYS, one set of evaluation items."""
    from halfwritten.regimes import resolve
    out, excluded, seen, items, configs, identities, digests, scorers = [], [], {}, set(), {}, {}, set(), set()
    for r in rows(root):  # finished rows, those set aside skipped: a rerun (or nothing) is the row
        if r.config.get("tag", "") != tag:
            continue
        row, cfg = r.path, r.config
        if why := r.problem(harness=harness, limits=limits):
            excluded.append({"row": str(row), "reason": why})
            continue
        evals = r.evals(split)
        if not evals:
            excluded.append({"row": str(row), "reason": f"no evaluation on {split}"})
            continue
        if accuracy_at_T(evals) is None:
            raise AnalysisError(f"{row}: no evaluation of the snapshot at the budget (tend)")
        if why := mark_problem(r, evals):
            excluded.append({"row": str(row), "reason": why})
            continue
        items |= {e.get("n") for e in evals}
        digests |= {e["items_sha256"] for e in evals if e.get("items_sha256")}
        scorers |= {(e.get("scorer"), e.get("math_verify")) for e in evals if "scorer" in e}
        configs[str(row)] = cfg
        identities[str(row)] = json.dumps(r.identity or None, sort_keys=True)
        arm = resolve(cfg["regime"], bool(cfg.get("lora"))).label  # "D+lora" for an adapter arm; F stays F
        key = (arm, cfg["seed"])
        if key in seen:
            raise AnalysisError(f"two rows of arm {key[0]} seed {key[1]}: {seen[key]} and {row}")
        seen[key] = row
        out.append({"row": str(row), "arm": arm, "regime": cfg["regime"], "task": cfg.get("task"),
                    "seed": cfg["seed"], "budget_s": cfg.get("budget_s"), "evals": evals, "emulated": r.emulated,
                    "stale_kv_share": r.stale_kv_share()})
    rows_ = out
    if len(items) > 1:
        raise AnalysisError(f"evaluations differ in item count {sorted(items, key=str)}: rerun bench.evaluate")
    if len(digests) > 1:
        raise AnalysisError(f"evaluations were made on {len(digests)} different item sets: rerun bench.evaluate")
    if len(scorers) > 1:
        raise AnalysisError(f"evaluations were scored by different scorers {sorted(scorers, key=str)}: rerun "
                            "bench.evaluate with one")
    # one experiment: across arms only the regime keys may differ (external review F03), within an arm only the seed
    factor_keys = ("launch.lora_inplace_fix",) if tag == "lora1srv" else ()
    if tag == "lora1srv":
        for path, cfg in configs.items():
            expected_fix = cfg.get("lora", False) and cfg["regime"] in ("D", "E")
            if cfg.get("launch", {}).get("lora_inplace_fix", False) != expected_fix:
                raise AnalysisError(f"{path}: unexpected launch.lora_inplace_fix for the LoRA factor")
    if diff := config_mismatch(configs, (*NOT_THE_MEASUREMENT, *REGIME_KEYS, *factor_keys)):
        raise AnalysisError(f"the rows differ in {', '.join(diff)}: not one experiment (only "
                            f"{', '.join(REGIME_KEYS)} may differ)")
    for arm in {r["arm"] for r in rows_}:
        if diff := config_mismatch({r["row"]: configs[r["row"]] for r in rows_ if r["arm"] == arm},
                                   (*NOT_THE_MEASUREMENT, "seed")):
            raise AnalysisError(f"arm {arm}'s rows differ in {', '.join(diff)}: analyse one configuration at a time")
    if len(set(identities.values())) > 1:
        raise AnalysisError("the rows differ in what their config names by path (warm start, prompt filter, "
                            "development split): not one experiment")
    return rows_, excluded


# -- inference

def paired_t(d, level: float = 0.95) -> dict:
    from scipy import stats
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    if not n > 1:
        return {"n": n, "mean": mean, "se": se, "ci": [math.nan, math.nan], "p": math.nan}
    if se == 0:
        return {"n": n, "mean": mean, "se": 0.0, "ci": [mean, mean], "p": 0.0 if mean else 1.0}
    q = float(stats.t.ppf(0.5 + level / 2, n - 1))
    p = float(2 * stats.t.sf(abs(mean / se), n - 1))
    return {"n": n, "mean": mean, "se": se, "ci": [mean - q * se, mean + q * se], "p": p}


def tost(d, delta: float) -> dict:
    """Equivalence within delta: the 90% t interval inside (-delta, +delta) (two one-sided tests at 0.05)."""
    t90 = paired_t(d, 0.90)
    lo, hi = t90["ci"]
    return {"ci90": [lo, hi], "equivalent": bool(-delta < lo and hi < delta)}


def sign_flip_p(d) -> float:
    """Exact two-sided sign-flip p-value of the mean over all 2^n flips (n <= 20)."""
    d = np.asarray(d, dtype=float)
    obs = abs(d.mean())
    flips = np.array(list(itertools.product((1, -1), repeat=len(d))))
    return float((np.abs((flips * d).mean(axis=1)) >= obs - 1e-12).mean())


def _jackknife_accel(x: np.ndarray, stat) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    jk = np.array([stat(np.delete(x, i)) for i in range(n)])
    d = jk.mean() - jk
    den = 6 * (d ** 2).sum() ** 1.5
    return float((d ** 3).sum() / den) if den > 0 else 0.0


def bca(x, stat=np.mean, level: float = 0.95, n_boot: int = 20000, seed: int = 0) -> tuple[float, float, float]:
    """(estimate, low, high): the bias-corrected and accelerated bootstrap interval of stat(x). Sensitivity only:
    anti-conservative with a handful of seeds."""
    from statistics import NormalDist
    x = np.asarray(x, dtype=float)
    est = float(stat(x))
    if len(x) < 2 or np.all(x == x[0]):
        return est, est, est
    rng = np.random.default_rng(seed)
    boots = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    nd = NormalDist()
    frac = np.clip((boots < est).mean() + 0.5 * (boots == est).mean(), 1e-6, 1 - 1e-6)
    z0, a = nd.inv_cdf(frac), _jackknife_accel(x, stat)
    out = []
    for q in ((1 - level) / 2, (1 + level) / 2):
        z = nd.inv_cdf(q)
        adj = nd.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out.append(float(np.quantile(boots, adj)))
    return est, out[0], out[1]


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (a missing p counts as 1)."""
    order = sorted(pvalues, key=lambda k: (pvalues[k] if np.isfinite(pvalues[k]) else 1.0))
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        p = pvalues[k] if np.isfinite(pvalues[k]) else 1.0
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def iqm(x) -> float:
    """Interquartile mean (rliable's aggregate): the mean of the middle 50%."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def item_bootstrap(a_correct: dict[int, str], b_correct: dict[int, str], n_boot: int = 5000,
                   seed: int = 0) -> tuple[float, float, float]:
    """Accuracy difference between arms A and B, resampling seeds and evaluation items together (both arms answer
    the same items)."""
    seeds = sorted(set(a_correct) & set(b_correct))
    a = np.array([[c == "1" for c in a_correct[s]] for s in seeds], dtype=float)
    b = np.array([[c == "1" for c in b_correct[s]] for s in seeds], dtype=float)
    est = float((a - b).mean())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        si = rng.integers(0, len(seeds), len(seeds))
        ii = rng.integers(0, a.shape[1], a.shape[1])
        boots.append(float((a[si][:, ii] - b[si][:, ii]).mean()))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return est, float(lo), float(hi)


# -- the analysis

def a2_accuracy_at_T(a_row: dict) -> tuple[float | None, str | None]:
    """A2's accuracy at T from an A row: S1 (this row's accuracy by steps) at the step count A2 reaches by T."""
    em, T = a_row.get("emulated"), a_row["budget_s"]
    if not em or not em.get("steps"):
        return None, "no emulated.json (run bench.emulate on the A row)"
    if em["steps"][-1]["A2_cum"] < T:  # emulation replays A's own steps: A2 ran out of them before T
        return None, (f"A2 passes A's last step {len(em['steps'])} at t = {em['steps'][-1]['A2_cum']:.0f} s < T; "
                      "extend A rows")
    reached = sum(1 for s in em["steps"] if s["A2_cum"] <= T)
    acc = at_steps(a_row["evals"], reached)
    if acc is None:
        return None, f"A stopped at step {final_steps(a_row['evals']):.0f}; A2 reaches step {reached} by T"
    return acc, None


def _resolve(name: str, arms: set[str], best: str | None) -> str:
    if name == "best":
        if not best:
            raise AnalysisError("the family names 'best': pass --best (chosen from the systems map, before accuracy)")
        name = best
    if name == "A2" or name in arms:
        return name
    raise AnalysisError(f"no arm {name!r} among {sorted(arms)}")


PRIMARY = ("accuracy_T", "auc")


def analyse(rows: list[dict], *, family: list[str], delta: float, best: str | None = None,
            threshold_frac: float = 0.9, excluded: list[dict] | None = None, primary: str = "auc",
            equivalence: list[list[str]] | list[str] = EQUIVALENCE, min_exposure: float = MIN_EXPOSURE,
            h4: str = "auto", intervals_only: bool = False) -> dict:
    """The pre-registered analysis (HYPOTHESES.md). The contrasts test `primary`: the normalised area under accuracy
    against time over [0, T] (decision 1, owner 2026-10-02: less sensitive to where T falls on the climb), or the
    accuracy of the snapshot at T. Time to threshold uses gain_threshold(rows, threshold_frac); steps to the same
    threshold give decision 3's steps-domain secondary. `equivalence` is a list of families, each a fixed sequence
    (a flat list is one family). With `intervals_only` (decision 13: an arm needed more seeds than its cap) no
    verdict is given: every contrast reports its intervals and nothing is claimed."""
    if primary not in PRIMARY:
        raise AnalysisError(f"primary endpoint {primary!r} is not one of {PRIMARY}")
    by_arm: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_arm[r["arm"]][r["seed"]] = r
    arms = set(by_arm)
    threshold = gain_threshold(rows, threshold_frac)
    endpoint: dict[str, dict[int, float]] = defaultdict(dict)
    summary = {}
    for arm, seeds in by_arm.items():
        for s, r in seeds.items():
            endpoint[arm][s] = accuracy_at_T(r["evals"]) if primary == "accuracy_T" else auc(r["evals"], r["budget_s"])
        acc = [accuracy_at_T(r["evals"]) for r in seeds.values()]
        aucs = [auc(r["evals"], r["budget_s"]) for r in seeds.values() if r["budget_s"]]
        summary[arm] = {"seeds": sorted(seeds), "accuracy_T": float(np.mean(acc)), "iqm": iqm(acc),
                        "auc": float(np.mean([a for a in aucs if a is not None])) if any(
                            a is not None for a in aucs) else None,
                        "final_steps": float(np.mean([final_steps(r["evals"]) for r in seeds.values()]))}
        if threshold is not None:
            tt = [time_to(r["evals"], threshold, r["budget_s"]) for r in seeds.values()]
            summary[arm]["time_to_threshold"] = {"median": float(np.median(tt)),
                                                 "never": int(sum(math.isinf(x) for x in tt))}
    if "A" in by_arm:  # A2's emulated curve gives an accuracy at T, not an area: a contrast only on that endpoint
        a2 = {s: a2_accuracy_at_T(r) for s, r in by_arm["A"].items()}
        ok = all(v is not None for v, _ in a2.values())
        if ok and primary == "accuracy_T":
            endpoint["A2"] = {s: v for s, (v, _) in a2.items()}
        summary["A2"] = {"emulated": True, "missing": {s: why for s, (_, why) in a2.items() if why}} | (
            {"accuracy_T": float(np.mean([v for v, _ in a2.values()]))} if ok else {})
    tests, pvals = {}, {}
    for spec in family:
        parts = spec.split("-")
        if len(parts) != 2:
            raise AnalysisError(f"contrast {spec!r} is not X-Y")
        a, b = (_resolve(x, arms, best) for x in parts)
        name = f"{a}-{b}"
        common = sorted(set(endpoint.get(a, {})) & set(endpoint.get(b, {})))
        if len(common) < 2:
            why = summary.get("A2", {}).get("missing") if "A2" in (a, b) else None
            tests[name] = {"spec": spec, "missing": f"{len(common)} paired seeds" + (f"; {why}" if why else "")}
            pvals[name] = math.nan
            continue
        d = [endpoint[a][s] - endpoint[b][s] for s in common]
        t = paired_t(d)
        _, blo, bhi = bca(d)
        tests[name] = {"spec": spec, "seeds": common, "diff": t["mean"], "ci95": t["ci"], "p": t["p"],
                       "tost_descriptive": tost(d, delta),  # an equivalence claim only in the sequence below
                       "sensitivity": {"bca95": [blo, bhi], "sign_flip_p": sign_flip_p(d)}}
        if a in by_arm and b in by_arm and primary == "accuracy_T":
            ac = {s: (at_budget(by_arm[a][s]["evals"]) or {}).get("correct") for s in common}
            bc = {s: (at_budget(by_arm[b][s]["evals"]) or {}).get("correct") for s in common}
            if all(ac.values()) and all(bc.values()):
                ie, il, ih = item_bootstrap(ac, bc)
                tests[name]["sensitivity"]["item_level"] = {"diff": ie, "ci95": [il, ih]}
        pvals[name] = t["p"]
    for name, p in holm(pvals).items():
        tests[name]["p_holm"] = p
        if "diff" in tests[name]:  # the superiority family: superior, inferior or not significant, nothing else
            tests[name]["verdict"] = "interval only" if intervals_only else \
                ("superior" if tests[name]["diff"] > 0 else "inferior") if p < 0.05 else "not significant"
    h2 = {}
    for x in ("B", "D", "E"):
        if x not in by_arm or "A" not in by_arm:
            continue
        common = sorted(set(by_arm[x]) & set(by_arm["A"]))
        d, at = [], {}
        for s in common:
            n = min(final_steps(by_arm[x][s]["evals"]), final_steps(by_arm["A"][s]["evals"]))
            vx, va = at_steps(by_arm[x][s]["evals"], n), at_steps(by_arm["A"][s]["evals"], n)
            if vx is not None and va is not None:
                d.append(vx - va)
                at[s] = n
        if len(d) >= 2:
            t = paired_t(d)
            h2[f"{x}-A"] = {"steps": at, "diff": float(np.mean(d)), **tost(d, delta), "ci95": t["ci"], "p": t["p"]}
        else:
            h2[f"{x}-A"] = {"missing": f"{len(d)} paired seeds"}
    # decision 3's secondary: the ratio of optimiser steps each arm needs to reach the threshold, against A (H2's
    # arms) and D against E (H4), per seed block; descriptive, its interval a t interval on the log ratios
    steps_ratio = {}
    if threshold is not None:
        for x, y in (("B", "A"), ("D", "A"), ("E", "A"), ("D", "E")):
            if x not in by_arm or y not in by_arm:
                continue
            common = sorted(set(by_arm[x]) & set(by_arm[y]))
            pairs = {s: (steps_to(by_arm[x][s]["evals"], threshold), steps_to(by_arm[y][s]["evals"], threshold))
                     for s in common}
            logs = [math.log(a / b) for a, b in pairs.values() if 0 < a < math.inf and 0 < b < math.inf]
            never = {s: [arm for arm, v in zip((x, y), ab) if math.isinf(v)] for s, ab in pairs.items()}
            entry = {"seeds": common, "never": {s: arms_ for s, arms_ in never.items() if arms_}}
            if len(logs) >= 2:
                t = paired_t(logs)
                entry |= {"ratio": math.exp(t["mean"]), "ci95": [math.exp(t["ci"][0]), math.exp(t["ci"][1])]}
            steps_ratio[f"{x}/{y}"] = entry
    # the equivalence claims: only the pre-registered ones, in families; within a family a fixed sequence, each at
    # the full alpha until the first that is not equivalent (later ones in it are then not claimed); H4 enters only
    # if D was exposed to stale KV
    families = [equivalence] if equivalence and isinstance(equivalence[0], str) else equivalence
    claims = []
    for fam, names in enumerate(families, 1):
        claims += _sequence(names, fam, tests, h2, by_arm, h4, min_exposure, delta, intervals_only)
    return {"primary": primary, "delta": delta, "best": best, "threshold": threshold, "excluded": excluded or [],
            **({"intervals_only": True} if intervals_only else {}),  # absent otherwise: the golden pin stands
            "arms": summary, "family": tests, "h2": h2, "steps_to_threshold": steps_ratio, "equivalence": claims}


def _sequence(names: list[str], fam: int, tests: dict, h2: dict, by_arm: dict, h4: str, min_exposure: float,
              delta: float, intervals_only: bool = False) -> list[dict]:
    claims, open_ = [], True
    for name in names:
        res = tests.get(name) if name == "D-E" else h2.get(name)
        if res is not None and "tost_descriptive" in res:
            res = res | res["tost_descriptive"]
        entry = {"contrast": name, "family": fam}
        if name == "D-E":
            shares = [r["stale_kv_share"] for r in by_arm.get("D", {}).values() if r.get("stale_kv_share") is not None]
            share = float(np.mean(shares)) if shares else None
            entry["d_stale_kv_share"] = share
            # membership is fixed at the tag (h4 "in" or "out", decision 4); "auto" decides from these rows' exposure
            out = h4 == "out" or (h4 == "auto" and (share is None or share < min_exposure))
            if out:
                shown = "none" if share is None else f"{share:.3f}"
                claims.append(entry | {"result": f"uninformative: H4 at natural depth is out (D's stale-KV share "
                                                  f"{shown}, floor {min_exposure}, h4={h4})"})
                continue
        if not open_:
            claims.append(entry | {"result": "not tested: the sequence stopped earlier"})
        elif res is None or "missing" in res:
            claims.append(entry | {"result": "missing"})
            open_ = False
        elif intervals_only:  # no claim and no sequence: each contrast's interval, reported as such
            claims.append(entry | {"result": "interval only", "diff": res["diff"], "ci90": res["ci90"]})
        else:  # each at the full alpha: the five verdicts, and equivalence (the 90% interval inside delta) goes on
            claims.append(entry | {"result": "equivalent" if res["equivalent"] else "not shown equivalent",
                                   "verdict": verdict(res["diff"], res["ci90"], res["p"], delta),
                                   "ci90": res["ci90"]})
            open_ = res["equivalent"]
    return claims


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--delta", type=float, default=None, help="the pre-registered equivalence margin (or the "
                                                                "amendment's)")
    ap.add_argument("--best", default=None, help="the best async arm, from the systems map's steps/hour ranking")
    ap.add_argument("--family", default=None)
    ap.add_argument("--tag", default="", help="select exactly one experiment (default: untagged main rows; "
                                            "depth or lora1srv for the MATH factor studies)")
    ap.add_argument("--primary", choices=PRIMARY, default="auc", help="the pre-registered primary endpoint")
    ap.add_argument("--no-harness-check", action="store_true", help="keep rows whose controller was the bottleneck")
    ap.add_argument("--split", choices=("dev", "test"), default="test", help="the evaluations analysed")
    ap.add_argument("--equivalence", default=None,
                    help="the equivalence claims: families separated by ';', each a sequence in its order (default: "
                         f"the amendment's `equivalence`, else {EQUIVALENCE_DEFAULT!r})")
    ap.add_argument("--min-exposure", type=float, default=MIN_EXPOSURE, help="D's stale-KV share for H4's claim")
    ap.add_argument("--h4", choices=("auto", "in", "out"), default="auto",
                    help="H4's membership in the equivalence sequence as fixed at the tag (auto: by --min-exposure)")
    ap.add_argument("--intervals-only", action="store_true",
                    help="no verdicts, intervals only (decision 13; the amendment sets it past the seed cap)")
    ap.add_argument("--amendment", default=None,
                    help="the grid's amendment (bench.amend): its delta, best async arm, h4, equivalence families "
                         "(`equivalence`, in --equivalence's syntax) and `intervals_only` unless given here")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    limits = None
    if a.amendment:
        amendment = yaml.safe_load(Path(a.amendment).read_text())
        a.delta = a.delta if a.delta is not None else amendment["delta"]
        limits = amendment.get("harness_limits")  # decision 7's limits as fixed at the tag (bench.amend)
        if not a.tag:  # these decisions describe the natural-depth main experiment only
            a.best = a.best or amendment.get("best_async")
            a.h4 = amendment.get("h4", a.h4) if a.h4 == "auto" else a.h4
            a.equivalence = amendment.get("equivalence") if a.equivalence is None else a.equivalence
            a.intervals_only = a.intervals_only or bool(amendment.get("intervals_only"))
    if a.delta is None:
        ap.error("--delta or --amendment is required")
    family = a.family if a.family is not None else {"depth": "D-E", "lora1srv": "F-D,F-E,D+lora-D,E+lora-E"}.get(
        a.tag, ",".join(FAMILY))
    # tier 2 (a tagged experiment) is descriptive: TOST reported, no equivalence sequence, no claim (HYPOTHESES.md
    # "Amendments") unless --equivalence names one
    equivalence = a.equivalence if a.equivalence is not None else ("" if a.tag else EQUIVALENCE_DEFAULT)
    rows, excluded = load_rows(Path(a.root), harness=not a.no_harness_check, split=a.split, tag=a.tag,
                               limits=limits)
    result = analyse(rows, family=[c for c in family.split(",") if c], delta=a.delta, best=a.best,
                     excluded=excluded, primary=a.primary,
                     equivalence=[[c for c in f.split(",") if c]
                                  for f in equivalence.split(";") if f.strip()],
                     min_exposure=a.min_exposure, h4=a.h4, intervals_only=a.intervals_only)
    from halfwritten.results import HARNESS_LIMITS
    result.update(tag=a.tag, task=next((r["task"] for r in rows if r.get("task")), None),
                  harness_limits=None if a.no_harness_check else HARNESS_LIMITS | (limits or {}))
    text = json.dumps(result, indent=1, default=float)
    label = a.tag.replace("/", "_")
    Path(a.out or Path(a.root) / (f"analysis-{label}.json" if label else "analysis.json")).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
