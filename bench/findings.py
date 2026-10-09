"""What the data says, as findings: each one a function over the page's gathered data (bench.dashboard.current) and,
once a grid has run, the analysis (bench.stats), returning a record {id, state, kind, title, text, sources} or
None when its data is absent. The lab page, README's "What we know" block and the paper's result sentences all
render from this list, so a number appears in one place and nowhere drifts (review-2026-10-02 plan, section 4c).

    state: Established (a check that passed), Measured, Caveat, Found (a defect), Result (a pre-registered test)
    kind:  the page's colour: good, old (neutral), warn, bad

A row that ran outside the harness rule (decision 7: fence p50 or loop lag over its limit) is reported with that
diagnostic, not hidden: the rule's limit is applied in the analysis and reviewed on the data.
"""
import math

MIN_EXPOSURE = 0.05
HORIZON_FRACTION = 0.8  # bench.amend trusts a fitted t80 only within this share of the pilot's run


def fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if abs(v) >= 1000:
        return f"{round(v):,}"
    if 0 < abs(v) < 0.01:
        return f"{v:.1e}".replace("e-0", "e-")
    return f"{v:.{digits}g}"


def steps_fmt(v) -> str:
    """Optimiser steps are counts, whatever the file stored them as."""
    return "" if v is None else f"{int(round(v)):,}"


def pct(v, digits: int = 0) -> str:
    """A share as a percentage; `digits` decimals (accuracies keep one, shares of tokens none)."""
    return "" if v is None else f"{100 * v:.{digits}f}%"


def minutes(seconds) -> str:
    return "" if seconds is None else f"{seconds / 60:.0f}-minute"


def _order(data):
    """Sort key putting the newest archive session first (bench.dashboard.sessions)."""
    order = data.get("sessions", {})
    return lambda x: order.get(x.get("session"), x.get("session") or "")


def _newest(data, items):
    return sorted(items, key=_order(data), reverse=True)


def _row(data, regime, tag=None, task="gsm8k", **where):
    """The newest session's row of this regime (and tag, task, and whatever else `where` fixes), or None."""
    hits = [r for r in data.get("rows", []) if r["regime"] == regime and (tag is None or r.get("tag") == tag)
            and (task is None or r.get("task") == task) and all(r.get(k) == v for k, v in where.items())]
    return _newest(data, hits)[0] if hits else None


def _study(data, compute, gpu=None):
    for p in data.get("phase1", []):
        if p["compute"] == compute and (gpu is None or p["gpu"] == gpu):
            return p
    return None


def _at(study, lag):
    return next((t for t in study["table"] if t["lag"] == lag), None) if study else None


def _lags(study) -> list[int]:
    return sorted(t["lag"] for t in (study["table"] if study else []) if t["lag"] >= 1)


def _span(study, key, min_lag=1) -> str:
    v = [t[key] for t in (study["table"] if study else []) if t["lag"] >= min_lag and t.get(key) is not None]
    return f"{round(100 * min(v))}-{round(100 * max(v))}%" if v else ""


def _curve(data, regime, tag, task="gsm8k", split="test", file="evals.jsonl"):
    """The newest session's curve of this arm, tag, task and split (never by its label), or None."""
    hits = [c for c in data.get("curves", []) if c.get("regime") == regime and c.get("tag") == tag
            and c.get("task") == task and c.get("split") == split and c["file"] == file]
    return _newest(data, hits)[0] if hits else None


def _pilot_curve(data, arm):
    return _curve(data, arm, "pilot")


def _sources(*items) -> list[str]:
    return sorted({i["source"] for i in items if i and i.get("source")})


def _harness_note(*rows) -> str:
    """The diagnostic of every row that ran outside decision 7's harness rule, appended to what rests on it."""
    flagged = [(r["regime"], r.get("tag") or "", r["harness"]) for r in rows if r and r.get("harness")]
    if not flagged:
        return ""
    return (" " + "; ".join(f"{reg}'s {tag} row: {why}" for reg, tag, why in flagged)
            + ": outside the harness rule as pre-registered (decision 7), whose limit is reviewed on the data.")


def f_canary(data, analysis):
    """The newest loaded canary run that passed under the decisive gate, both regimes from the same report."""
    passing = [c for c in data.get("canaries", []) if c.get("loaded") and c.get("passed_now") and c.get("decisive")]
    runs = {}
    for c in sorted(passing, key=lambda c: (_order(data)(c), c.get("at") or 0)):
        runs.setdefault(c["source"], {})[c["regime"]] = c
    d, e = next(((r["D"], r["E"]) for r in reversed(list(runs.values())) if "D" in r and "E" in r), (None, None))
    if not (d and e):
        return None
    load = f" ({d['load']} answers alongside)" if d.get("load") else ""
    return {"id": "canary", "state": "Established", "kind": "good",
            "title": "vLLM's pause modes do what the regimes assume.",
            "text": f"Under the runs' own load{load}, the canary swapped a version-revealing update "
                    f"in mid-answer: D's later tokens read the stale-KV hypothesis at all {d['decisive']} positions "
                    f"where it and the recomputed one clearly differ, E's the recomputed one at all {e['decisive']}, "
                    "from the exact token the ledger records.", "sources": _sources(d, e)}


def f_exact_cut(data, analysis):
    # only rows whose syncs record the fence's count (`undelivered`): a row cut by the old window fence is not exact;
    # and one session's rows only (the newest with any), so the counts never mix two versions of the code
    certify = _newest(data, [r for r in data.get("rows", []) if (r.get("tag") or "").startswith("certify")
                             and r.get("cuts") and r["cuts"].get("recorded")])
    certify = [r for r in certify if r.get("session") == certify[0].get("session")] if certify else []
    if not certify:
        return None
    n = sum(r["cuts"]["n"] for r in certify)
    exact = sum(r["cuts"]["exact"] for r in certify)
    fences = [r["fence_p50"] for r in certify if r.get("fence_p50") is not None]
    text = (f"vLLM counts generated tokens after queueing them to their streams, so a sync cuts only once the tokens "
            f"delivered equal that count and hold still. The counts matched at {exact} of {n} cuts of the "
            f"{len(certify)} certification rows of {certify[0].get('session') or 'the archive'}")
    if fences:
        text += f", where the fence's median was {round(1000 * sum(fences) / len(fences))} ms"
    return {"id": "exact-cut", "state": "Established" if exact == n else "Caveat",
            "kind": "good" if exact == n else "warn",
            "title": "Every cut is exact." if exact == n else f"{n - exact} of {n} certification cuts were not exact.",
            "text": text + ".", "sources": _sources(*certify)}


def f_throughput(data, analysis):
    if analysis and analysis.get("task", "gsm8k") == "gsm8k":
        return None  # the grid's throughput figure (nine rows an arm) supersedes the pilot's one row each
    a, d, e = (_row(data, x, "pilot") for x in "ADE")
    if not (a and d and e):
        return None
    b = _row(data, "B")
    rate = lambda r: r["steps_per_hour"] / a["steps_per_hour"]  # noqa: E731
    text = (f"From the same warm start and prompts, A made {fmt(a['steps_per_hour'])} optimiser steps an hour, E "
            f"{fmt(e['steps_per_hour'])} and D {fmt(d['steps_per_hour'])}: the engine never waits for the trainer.")
    if b:
        text += (f" The drain (B) gives part of that back to its longest answers: {fmt(b['steps_per_hour'])} an hour "
                 f"in a {minutes(b.get('budget_s'))} row from {'W0' if b.get('from_w0') else 'the base model'}.")
    return {"id": "throughput", "state": "Measured", "kind": "old",
            "title": f"On GSM8K at K = {d['K']}, one {minutes(a.get('budget_s'))} row each from W0: D made "
                     f"{rate(d):.1f}x A's optimiser steps an hour, E {rate(e):.1f}x.",
            "text": text + _harness_note(a, d, e, b), "sources": _sources(a, d, e, b)}


def f_pilot(data, analysis):
    from bench.amend import AmendmentStop, fit_gain
    curves = {arm: _pilot_curve(data, arm) for arm in "ADE"}
    if not all(curves.values()):
        return None
    start = curves["A"]["points"][0]["acc"]
    t80, at_end, horizon = {}, {}, 0.0
    for arm, c in curves.items():
        pts = [(p["t"], p["acc"]) for p in c["points"] if p.get("t") is not None]
        try:
            t80[arm] = fit_gain(pts, start)["t80_s"] / 60
        except (AmendmentStop, RuntimeError):
            return None
        at_end[arm] = max(pts)[1]
        horizon = max(horizon, max(pts)[0] / 60)
    within = {arm: t <= HORIZON_FRACTION * horizon for arm, t in t80.items()}
    fitted = [f"{arm}'s {t80[arm]:.1f}" for arm in "DEA" if within[arm]]
    beyond = [f"{arm}'s {t80[arm]:.1f}" for arm in "DEA" if not within[arm]]
    title = "In the first pilot the fitted t80 was " + ", ".join(fitted) + " minutes"
    if beyond:
        title += "; " + " and ".join(beyond) + f" lie past what a {horizon:.0f}-minute run supports"
    rows = [_row(data, arm, "pilot") for arm in "ADE"]
    return {"id": "pilot", "state": "Measured", "kind": "old", "title": title + ".",
            "text": f"From W0 ({pct(start, 1)}), {horizon:.0f} minutes each, on test: A stood at "
                    f"{pct(at_end['A'], 1)}, E at {pct(at_end['E'], 1)} and D at {pct(at_end['D'], 1)}. t80 is the "
                    "time at which the fitted curve a0 + gain (1 - e^(-t/tau)) has made 80% of its gain; the "
                    f"amendment trusts a fit only when t80 is within {HORIZON_FRACTION:.0%} of the run, so the "
                    "others are extrapolations. One seed an arm; the dev rerun sets T and the seeds (decision 11)."
                    + _harness_note(*rows),
            "sources": _sources(*curves.values())}


def f_exposure(data, analysis, analyses=()):
    d = _row(data, "D", "pilot")
    if not d or d.get("stale_kv_share") is None:
        return None
    one = _row(data, "D", lora=False, api_servers=1)
    text = (f"Pauses of tens of milliseconds leave few GSM8K answers mid-flight at a sync: {pct(d['stale_kv_share'])} "
            "of D's consumed tokens were generated over stale KV (token-weighted, after the first minute)")
    if one and one.get("stale_kv_share") is not None:
        text += f", against {pct(one['stale_kv_share'])} on one API server, whose pauses are longer"
    # a tier's measured share, from its analysis (bench.stats records D's share with the D-E claim)
    shares = [(a.get("task", ""), a.get("tag", ""), c.get("d_stale_kv_share")) for a in analyses
              for c in a.get("equivalence", [])
              if c.get("contrast") == "D-E" and c.get("d_stale_kv_share") is not None and a.get("task") != "gsm8k"]
    if shares:
        text += ". " + "; ".join(f"On {task.upper()}{' (' + tag + ')' if tag else ''}, with its long answers, the "
                                 f"share was {pct(share)}" for task, tag, share in shares)
        text += ": there the question is live."
    else:
        text += (". At that exposure D and E cannot differ much in learning on GSM8K; the comparison that can "
                 "differ is MATH's longer answers.")
    low = d["stale_kv_share"] < 2 * MIN_EXPOSURE
    return {"id": "exposure", "state": "Measured", "kind": "warn" if low else "old",
            "title": f"On GSM8K, with an exact cut, D runs on stale KV for about {pct(d['stale_kv_share'])} of its "
                     "tokens" + (f"; on {shares[0][0].upper()} for {pct(shares[0][2])}." if shares else "."),
            "text": text + _harness_note(d, one), "sources": _sources(d, one)}


def f_s1(data, analysis):
    s1 = _curve(data, "A", "s1")
    if not s1:
        return None
    at = {p["step"]: p["acc"] for p in s1["points"]}
    plateau = [p["acc"] for p in s1["points"] if p["step"] >= 250]
    if not plateau or 200 not in at:
        return None
    return {"id": "s1", "state": "Measured", "kind": "old",
            "title": "The synchronous curve climbs for about 200 steps, then plateaus.",
            "text": f"Arm A from the base model (GSM8K test, all {s1['points'][0]['n']:,} items): {fmt(at.get(0))} at "
                    f"step 0, {fmt(at.get(50))} at 50, {fmt(at.get(100))} at 100, {fmt(at[200])} at 200, then "
                    f"{fmt(min(plateau))}-{fmt(max(plateau))} to step {steps_fmt(s1['points'][-1]['step'])}. Where "
                    "the regimes can differ is inside the climb.", "sources": _sources(s1)}


def f_phase1(data, analysis):
    fp32 = _study(data, "fp32")
    if not fp32 or not _lags(fp32):
        return None
    lags = _lags(fp32)
    return {"id": "phase1", "state": "Measured", "kind": "old",
            "title": "Right after a swap, a stale cache keeps most of the old-versus-new divergence.",
            "text": "In fp32 compute, over the first 8 tokens after a swap a stale cache keeps "
                    f"{_span(fp32, 'R_0_8')} of the old-versus-new KL, at every lag from {lags[0]} to {lags[-1]} "
                    f"steps; 32 or more tokens later it keeps {_span(fp32, 'R_32_999')}, as fresh KV accumulates. "
                    f"Recomputing the KV of the first 14 of 28 layers removes {_span(fp32, 'removed_l14')} of it, "
                    f"the first 21 {_span(fp32, 'removed_l21')}: layers 15-21 hold most of the staleness.",
            "sources": _sources(fp32)}


def f_bf16(data, analysis):
    """Two ratios, each with one statistic on both sides: medians for the per-position drift between checkpoints
    (bf16 against fp32 compute), means for the engine-trainer mismatch against the one-step drift."""
    fp32, bf16 = _study(data, "fp32"), _study(data, "bf16", "H100 NVL")
    floor = next((f for f in data.get("floors", []) if f["engine_head"] == "fp32" and f["condition"] == "steady"),
                 None)
    if not (fp32 and bf16 and _at(bf16, 1) and _at(fp32, 1) and floor):
        return None
    far = _lags(bf16)[-1]
    mismatch = floor["k3"] / _at(fp32, 1)["drift_mean"]
    return {"id": "bf16", "state": "Caveat", "kind": "warn",
            "title": "At lr 1e-6, one step moves the policy less than bf16 arithmetic does.",
            "text": f"Between checkpoints one step apart, bf16 compute moves a position's distribution by "
                    f"{fmt(_at(bf16, 1)['drift_median'], 2)} nats (median), fp32 compute by "
                    f"{fmt(_at(fp32, 1)['drift_median'], 2)}; in bf16 the median is still "
                    f"{fmt(_at(bf16, 4)['drift_median'], 2) if _at(bf16, 4) else ''} at lag 4 and "
                    f"{fmt(_at(bf16, far)['drift_median'], 2)} at lag {far}. The engine-trainer mismatch at identical "
                    f"weights (k3 with the fp32 head, mean over {floor['tokens']:,} tokens) is {fmt(floor['k3'], 2)}, "
                    f"about {mismatch:.0f} times the mean one-step drift in fp32 ({fmt(_at(fp32, 1)['drift_mean'], 2)})"
                    ". Staleness within K = 4 steps sits inside the engine's own noise, which is why the study "
                    "measures its cost in learning, not in log-probabilities.",
            "sources": _sources(fp32, bf16, floor)}


def f_lora(data, analysis):
    lora_d = _row(data, "D", lora=True, api_servers=1)
    one = _row(data, "D", lora=False, api_servers=1)
    f = _row(data, "F", api_servers=1)
    if not (lora_d and one):
        return None
    text = ("After an in-place reload every later request carries the reload's flag, so each engine step reloads an "
            "adapter from disk, and a batch with old and new requests applies the first one's: a request sent after "
            f"the swap generated the old adapter's answer. D with in-place LoRA made {fmt(lora_d['steps_per_hour'])} "
            f"steps an hour against full-weight D's {fmt(one['steps_per_hour'])} on the same single API server")
    if f:
        text += f"; F, a new adapter name per version, made {fmt(f['steps_per_hour'])}"
    text += ". halfwritten now clears the flag as SkyRL and prime-rl do; a GPU check is queued."
    return {"id": "lora", "state": "Found", "kind": "bad",
            "title": "vLLM 0.30 cannot reload a LoRA adapter in place correctly.",
            "text": text + _harness_note(lora_d, one, f), "sources": _sources(lora_d, one, f)}


TASK_NAMES = {"gsm8k": "GSM8K", "math": "MATH"}
TAG_NAMES = {"depth": "depth stress", "lora1srv": "the LoRA factor"}


def _experiment(analysis: dict) -> tuple[str, str]:
    """(the suffix of the block's id, the prefix of its title) naming an analysis's experiment: its task and tag,
    when the file records them (bench.stats writes both); an older file without them keeps the bare names."""
    task, tag = analysis.get("task") or "", analysis.get("tag") or ""
    suffix = "-".join(x for x in (task, tag) if x)
    name = " ".join(x for x in (TASK_NAMES.get(task, task), TAG_NAMES.get(tag, tag)) if x)
    return (f"-{suffix}" if suffix else ""), (f"{name}: " if name else "")


def _titled(prefix: str, title: str) -> str:
    return prefix + (title[0].lower() + title[1:] if prefix else title)


def _grid(analysis):
    """One experiment's pre-registered findings, once its grid has run: one bench.stats file (a task, or a task and
    a tag), its blocks named after it and reading its own delta. Several experiments are never merged: GSM8K and
    MATH share contrast names and differ in delta."""
    if not analysis:
        return []
    out = []
    suffix, prefix = _experiment(analysis)
    ci = lambda c: f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]" if c else ""  # noqa: E731
    endpoint = {"auc": "the area under held-out accuracy against wall-clock time to T",
                "accuracy_T": "held-out accuracy at T"}.get(analysis.get("primary", "auc"), analysis.get("primary"))
    # the claim families (bench.stats): 1 is H4's D-E on the primary endpoint, 2 is H2's at matched optimiser steps
    claim_on = lambda c: ("accuracy at matched optimiser steps, H2" if c.get("family") == 2  # noqa: E731
                          else "the primary endpoint over wall clock, H4")
    lines = []
    for name, t in analysis.get("family", {}).items():
        if "diff" in t:
            lines.append(f"{name}: {100 * t['diff']:+.1f} points, 95% CI {ci(t['ci95'])}, Holm p {t['p_holm']:.2g}: "
                         f"{t['verdict']}")
        else:
            lines.append(f"{name}: not computed ({t.get('missing', '')})")
    only = analysis.get("intervals_only")  # decision 13 (past the seed cap, nothing is claimed) or an exploratory tier
    why_only = ("an exploratory tier under a recorded amendment" if analysis.get("tag")
                else "an arm needed more seeds than its cap (decision 13)")
    if lines:
        kinds = {t.get("verdict") for t in analysis["family"].values()}
        out.append({"id": f"grid-family{suffix}", "state": "Result", "kind": "good" if "superior" in kinds else "old",
                    "title": _titled(prefix, f"The superiority contrasts on {endpoint}"
                                             + (f", as intervals only: {why_only}." if only else ".")),
                    "text": "; ".join(lines) + ".", "sources": []})
    claims = analysis.get("equivalence", [])
    if claims:
        text = "; ".join(f"{c['contrast']} (family {c.get('family', '')}, {claim_on(c)}): {c['result']}"
                         + (f", {c['verdict']}" if c.get("verdict") else "")
                         + (f", 90% CI {ci(c['ci90'])}" if c.get("result") == "interval only" else "")
                         for c in claims)
        out.append({"id": f"grid-equivalence{suffix}", "state": "Result", "kind": "old",
                    "title": _titled(prefix, f"The equivalence contrasts, as intervals only, against delta = "
                                             f"{100 * analysis['delta']:.0f} points ({why_only})." if only else
                                             f"The equivalence claims, within delta = {100 * analysis['delta']:.0f} "
                                             "points: H4 on the primary endpoint over wall clock, H2 at matched "
                                             "optimiser steps."),
                    "text": text + ".", "sources": []})
    ratios = analysis.get("steps_to_threshold", {})
    if any("ratio" in v for v in ratios.values()):
        text = "; ".join(f"{k}: {v['ratio']:.2f} (95% CI {v['ci95'][0]:.2f}-{v['ci95'][1]:.2f})"
                         for k, v in ratios.items() if "ratio" in v)
        out.append({"id": f"grid-steps{suffix}", "state": "Result", "kind": "old",
                    "title": _titled(prefix, "Optimiser steps to the threshold, arm against arm."),
                    "text": text + ".", "sources": []})
    return out


ORDER = (f_canary, f_exact_cut, f_throughput, f_pilot, f_exposure, f_s1, f_phase1, f_bf16, f_lora)


def findings(data: dict, analyses=None) -> list[dict]:
    """Every finding whose data is present, in the page's order; each grid's results first once there are any
    (`analyses`: bench.stats's files, one experiment each, in the order given; one dict is one experiment)."""
    analyses = [analyses] if isinstance(analyses, dict) else list(analyses or [])
    out = [rec for a in analyses for rec in _grid(a)]
    for f in ORDER:
        rec = f(data, analyses[0] if analyses else None, analyses) if f is f_exposure else \
            f(data, analyses[0] if analyses else None)
        if rec is not None:
            out.append(rec)
    return out


def markdown(items: list[dict]) -> str:
    """README's "What we know" block."""
    return "\n".join(f"- **{f['title']}** {f['text']}" for f in items)
