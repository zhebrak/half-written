# ruff: noqa: E501  (the page's template lines run long by nature)
"""Builds the interactive charts page, docs/charts.html, from the archived session (data/two_gpu-2026-10-03),
its two analysis files (docs/analysis/two_gpu-2026-10-03/) and the results page's regimes explainer (bench/dashboard.html):
every chart's data is computed here and inlined, so the page needs no server and no library.

    python scripts/build_post.py                              docs/charts.html, the self-contained charts page
    python scripts/build_post.py --site <site> --note <text>  a site's post from a text file (its first line the title,
                                                              its placeholder lines replaced by the panels): the
                                                              Markdown prose for the site's own renderer, the panels and
                                                              scripts as HTML blocks, the chart styles scoped under .hw
"""
import argparse
import collections
import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root, when run as a script
from bench.dashboard import current, gather  # noqa: E402
from halfwritten.regimes import resolve  # noqa: E402
from halfwritten.results import rows  # noqa: E402

ROOT = Path("data/two_gpu-2026-10-03")


def curves_for(task, tag):
    by, sph, sync, pause, steps = (collections.defaultdict(list) for _ in range(5))
    for r in rows(ROOT / task):
        c = r.config
        if (c.get("tag") or "") != tag:
            continue
        arm = resolve(c["regime"], bool(c.get("lora"))).label
        ev = sorted(r.evals("test"), key=lambda e: e["t"])
        by[arm].append((c.get("seed"), [(round(e["t"], 1), e["accuracy"]) for e in ev]))
        m = r.metrics
        sph[arm].append(m["steps_per_hour"]); steps[arm].append(m["steps"])
        sync[arm].append((m.get("sync_s") or {}).get("p50")); pause[arm].append((m.get("pause_s") or {}).get("p50"))
    arms = {}
    for arm, rows_ in by.items():
        cs = [series for _, series in rows_]
        pts = []
        for i, (t, _) in enumerate(cs[0]):
            v = [c[i][1] for c in cs if i < len(c)]
            pts.append({"t": t, "mean": round(st.mean(v), 4), "lo": round(min(v), 4), "hi": round(max(v), 4),
                        "sd": round(st.stdev(v), 4) if len(v) > 1 else 0})
        if arm == "C1":
            continue  # E by another route (abort and resubmit); the post does not show it
        arms[arm] = {"n": len(cs), "points": pts, "steps_per_hour": [round(x) for x in sph[arm]], "steps": steps[arm],
                     "seeds": [{"seed": seed, "points": [[t, round(a, 4)] for t, a in series]} for seed, series in rows_],
                     "sync_p50": round(st.mean(x for x in sync[arm] if x is not None), 3),
                     "pause_p50": round(st.mean(x for x in pause[arm] if x is not None), 3)}
    return arms


def analysis(path):
    a = json.loads(Path(path).read_text())
    fam = [{"contrast": k, "diff": t["diff"], "ci95": t["ci95"], "p": t.get("p"), "p_holm": t.get("p_holm"),
            "verdict": t["verdict"]} for k, t in a["family"].items() if "diff" in t]
    eq = [{"contrast": c["contrast"], "family": c.get("family"), "result": c["result"], "verdict": c.get("verdict"),
           "ci90": c.get("ci90"), "share": c.get("d_stale_kv_share")} for c in a.get("equivalence", [])]
    arms = {k: {"n": len(v["seeds"]), "accuracy_T": v["accuracy_T"], "auc": v["auc"], "final_steps": v["final_steps"],
                "ttt_median_s": (v.get("time_to_threshold") or {}).get("median")}
            for k, v in a["arms"].items() if "accuracy_T" in v and k != "C1"}
    return {"family": fam, "equivalence": eq, "arms": arms, "delta": a["delta"], "threshold": a.get("threshold"),
            "intervals_only": bool(a.get("intervals_only"))}


def exposure_of(task, tag):
    r = next(r for r in rows(ROOT / task) if (r.config.get("tag") or "") == tag and r.config["regime"] == "D"
             and r.config["seed"] == 0)
    led = [x for x in r.ledger if x["t_open"] - r.ledger[0]["t_open"] > 60]
    total = sum(x["tokens"] for x in led)
    stale = sum(s["end"] - s["start"] for x in led for s in x["segments"] if s["context"] == "stale_kv")
    crossed = [x for x in led if any(s["context"] == "stale_kv" for s in x["segments"])]
    pos = [next(s["start"] for s in x["segments"] if s["context"] == "stale_kv") / x["tokens"] for x in crossed if x["tokens"]]
    hist = [0] * 10
    for p in pos:
        hist[min(9, int(p * 10))] += 1
    dur = sorted(x["t_close"] - x["t_open"] for x in led)
    m = r.metrics
    return {"answers": len(led), "crossed": len(crossed), "crossed_share": round(len(crossed) / len(led), 4),
            "stale_share": round(stale / total, 4), "mean_duration_s": round(st.mean(dur), 2),
            "p90_duration_s": round(dur[int(0.9 * len(dur))], 2), "sync_every_s": round(m["wall_s"] / m["syncs"], 2),
            "swap_position_hist": hist, "mean_tokens": round(st.mean(x["tokens"] for x in led))}


def phase1_story():
    """Three views of the fp32 teacher-forcing study, from its per-position records: the share of the update a stale
    cache undoes by distance from the swap (pooled over lags), and the share of the stale divergence each band of
    layers holds. Intervals are 95% bootstraps over prompts."""
    import math
    import numpy as np
    import pyarrow.parquet as pq
    from bench.stale_kv import _cluster_ci
    d = Path("data/h100nvl-2026-10-01/phase1/gsm8k-qwen3-0.6b-h100-fp32compute")
    rows_ = [r for r in pq.read_table(d / "positions.parquet").to_pylist()
             if r["lag"] > 0 and math.isfinite(r["kl_fresh_stale"]) and math.isfinite(r["kl_fresh_old"])]
    ratio = lambda sel: sum(r["kl_fresh_stale"] for r in sel) / sum(r["kl_fresh_old"] for r in sel)  # noqa: E731
    fade = {"all": [], "bins": []}
    edges = [0, 4, 8, 16, 32, 64, 96]  # widening bins: a few positions carry most of the divergence, so narrow bins are noise
    for lo, hi in zip(edges, edges[1:]):
        sel = [r for r in rows_ if lo <= r["offset"] < hi]
        a, b = _cluster_ci(sel, "kl_fresh_stale", "kl_fresh_old")
        x = (lo + hi) / 2
        fade["bins"].append({"x": x, "label": f"{lo}–{hi}"})
        fade["all"].append({"x": x, "y": round(ratio(sel), 4), "lo": round(a, 4), "hi": round(b, 4)})
    layers = [1, 7, 14, 21, 27]
    by_prompt = collections.defaultdict(lambda: np.zeros(len(layers) + 1))
    for r in rows_:
        g = by_prompt[r["prompt"]]
        g[0] += r["kl_fresh_stale"]
        for i, l in enumerate(layers):
            g[i + 1] += r[f"kl_fresh_stale_l{l}"]
    a = np.array(list(by_prompt.values()))
    def band_shares(sums):  # cumulative share removed through layer l, then the bands between the cuts
        cum = 1 - sums[..., 1:] / np.maximum(sums[..., :1], 1e-30)
        return np.stack([cum[..., 1], cum[..., 2] - cum[..., 1], cum[..., 3] - cum[..., 2], cum[..., 4] - cum[..., 3], 1 - cum[..., 4]], -1)
    point = band_shares(a.sum(0))
    rng = np.random.default_rng(0)
    boot = band_shares(a[rng.integers(0, len(a), (2000, len(a)))].sum(1))
    ci = np.quantile(boot, [0.025, 0.975], axis=0)
    bands = [{"label": lab, "share": round(float(p), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}
             for lab, p, lo, hi in zip(["1 to 7", "8 to 14", "15 to 21", "22 to 27", "28"], point, ci[0], ci[1])]
    return {"fade": fade, "bands": bands}


def chart_data():
    out = {"experiments": {
        "gsm8k": {"label": "GSM8K", "task": "GSM8K", "T_s": 300, "start": "W0 (59.1%)", "start_acc": 0.591,
                  "curves": curves_for("gsm8k", ""), **analysis("docs/analysis/two_gpu-2026-10-03/gsm8k.json")},
        "math": {"label": "MATH", "task": "MATH", "T_s": 600, "start": "the base model (24.4%)", "start_acc": 0.244,
                 "curves": curves_for("math", "math-base"), **analysis("docs/analysis/two_gpu-2026-10-03/math-base.json")}}}
    sc = next(r for r in rows(ROOT / "math") if r.config.get("tag") == "sync-curve")
    out["math_sync_curve"] = [{"step": e["steps"], "acc": e["accuracy"], "tokens": round(e["mean_tokens"]), "n": e["n"]}
                              for e in sorted(sc.evals("dev"), key=lambda e: e["steps"])]
    out["exposure"] = {"gsm8k": exposure_of("gsm8k", ""), "math": exposure_of("math", "math-base")}
    data = current(gather([Path("data")]))
    fp32 = next((p for p in data["phase1"] if p["compute"] == "fp32"), None)
    keys = ("lag", "drift_mean", "drift_median", "R", "R_ci95", "R_0_8", "R_0_8_ci95", "R_8_32", "R_32_999", "R_32_999_ci95",
            "alpha_median", "removed_l14", "removed_l14_ci95", "removed_l21", "removed_l21_ci95")
    out["phase1"] = {"gpu": fp32["gpu"], "table": [{k: t.get(k) for k in keys} for t in fp32["table"]]} if fp32 else None
    out["phase1_story"] = phase1_story() if fp32 else None
    can = [c for c in data["canaries"] if c["session"] == ROOT.name]
    out["canary"] = {reg: {k: c[k] for k in ("boundary", "total", "old", "stale", "fresh", "decisive", "own_on_decisive", "load")}
                     for reg, c in ((reg, [c for c in can if c["regime"] == reg][-1]) for reg in ("D", "E"))}
    return out



tpl = Path("bench/dashboard.html").read_text().split("\n")
def between(start_pat, end_pat, inclusive_end=False):
    s = next(i for i, l in enumerate(tpl) if start_pat in l); e = next(i for i, l in enumerate(tpl) if end_pat in l and i > s)
    return "\n".join(tpl[s:e + (1 if inclusive_end else 0)])
explainer_css = between("/* The regimes explainer: one lane", "/* A value's bar inside its table cell. */")
explainer_markup = between('<div class="controls">', '<div class="lanes" id="rx-lanes"></div>', inclusive_end=True)
explainer_js = between("// -- the regimes explainer", "</script>")  # the whole block up to its closing tag
line_chart_js = between("function lineChart(el, opt) {", "function render() {")
DATA = chart_data()
CHARTS = Path("docs/charts.html")
PLACEHOLDERS = ["<Interactive chart of regimes.>", "<Interactive GSM chart>", "<Interactive MATH chart>",
                "<Staleness divergence charts>"]


def note_parts(note: Path | None) -> tuple[str, str, list[str]]:
    """A note's title (its first line, a '# ' heading), its opening line, and the rest of its lines; with no note, the
    charts alone: the four placeholder lines and nothing else."""
    if note is None:
        return "Half-written answers", "The regimes, and the charts behind the post.", [""] + [l for ph in PLACEHOLDERS for l in (ph, "")]
    lines = note.read_text().split("\n")
    assert lines[0].startswith("# "), "the note starts with its title: a '# ' line"
    title, body = lines[0][2:].strip(), lines[1:]
    standfirst = next((l for l in body if l.strip()), "").strip()
    return title, standfirst, body


def render_markdown(text: str) -> str:
    """Markdown to HTML with the markdown package, as the page has always been rendered (tables and smart quotes)."""
    tmp = Path("docs/.post-render.md"); tmp.write_text(text)
    try:
        return subprocess.run(["uvx", "--from", "markdown==3.11", "markdown_py", "-x", "tables", "-x", "smarty", str(tmp)],
                              capture_output=True, text=True, check=True).stdout
    finally:
        tmp.unlink(missing_ok=True)


title, standfirst, note_body = note_parts(None)
PANELS = {
 "The question": '''
<section id="answers" class="wide">
  <h3>Scrub one swap through one answer</h3>
  <p class="note">Time runs left to right on the same wall clock in every lane; nothing else is shared. Each lane follows one answer, which started on weights <b class="v">v</b>, and the two processes behind it: the engine that generates and the trainer that updates. The weight update, when <b class="n">v+1</b> is copied into the engine, is the black segment on each lane's bars; it comes when the regime allows it: at once in the in-flight and abort regimes, after the running answers finish in the drain, and only after the batch and the trainer's turn in the synchronous regime, where the two take turns and the trainer has nothing to train on until the batch ends. Each token cell is coloured by the weights that sampled it and the KV cell below by the weights that computed its cached keys and values, which every later token reads. Hover a cell for its word, a bar segment for its phase.</p>
''' + explainer_markup + '''
</section>''',
 "Three instruments": '''
<div class="panel"><h3>What the canary saw before the MATH tier ran</h3><div class="scroll"><table id="canary-table"></table></div>
<p class="cap">Under 127 answers of load, teacher-forced over the 400-token canary answer. A decisive position is one where STALE and FRESH predict different tokens by more than 2 bf16 ulps. The GSM8K day's canary, under 255 answers of load, read D as STALE at all 186 decisive positions and E as FRESH at all 47.</p></div>''',
 "Experiment 1: GSM8K under pre-registration": '''
<div class="panel chart" id="gsm8k-curves"></div>
<div class="grid2"><div class="panel chart" id="gsm8k-forest"></div><div class="panel chart" id="gsm8k-strip"></div></div>''',
 "Experiment 2: MATH during the climb": '''
<div class="panel chart" id="math-sync"></div>
<div class="panel chart" id="math-curves"></div>
<div class="grid2"><div class="panel chart" id="math-forest"></div><div class="panel chart" id="math-strip"></div></div>
<div class="panel chart" id="exposure"></div>''',
 "The mechanism, measured directly": '''
<div class="grid2"><div class="panel chart" id="p1-fade"></div><div class="panel chart" id="p1-bands"></div></div>''',
}
NOTE_PANELS = {
    "<Interactive chart of regimes.>": lambda: '<section id="answers" class="wide">' + explainer_markup + "</section>",  # the lanes alone
    "<Interactive GSM chart>": lambda: '<div class="panel chart" id="gsm8k-curves"></div>',
    "<Interactive MATH chart>": lambda: '<div class="panel chart" id="math-curves"></div>',
    "<Staleness divergence charts>": lambda: PANELS["The mechanism, measured directly"],
}


def note_markdown(body: list[str], wrap, skip_standfirst: bool) -> str:
    """The note's lines as Markdown with each placeholder line replaced by its panels' HTML (through `wrap`), the
    closing heading marked, and `<1%` escaped so no parser reads it as a tag."""
    out, seen_first = [], False
    for line in body:
        if line.strip() in NOTE_PANELS:
            out.append(wrap(NOTE_PANELS[line.strip()]()))
        elif line.strip() == "Related work and further reading":
            out.append("## " + line.strip() + "\n")
        else:
            if skip_standfirst and line.strip() and not seen_first:
                seen_first = True
                continue  # the header shows it
            out.append(line.replace("<1%", "\\<1%"))
    return "\n".join(out)


content = render_markdown(note_markdown(note_body, lambda html: "\n" + html + "\n", skip_standfirst=False))
content = content.replace("<table>", '<div class="scroll"><table>').replace("</table>", "</table></div>")
page = f'''<title>Half-written answers</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=JetBrains+Mono:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
/* Layout: a reading column with interactive panels set wider than the text; the project's own palette (blue = old
   weights, amber = new) and its three faces. Every chart is hand-drawn SVG from the inlined data; no library. */
:root {{
  --paper: #f2f4f6; --panel: #fbfcfd; --ink: #17212b; --muted: #5a6774; --rule: #d3dae1;
  --old: #2f5d8c; --old-soft: #d6e2ee; --new: #c77700; --new-soft: #f6e3c4; --idle: #e3e8ed;
  --good: #2e7d4f; --good-soft: #d9efe1; --warn: #a15c00; --warn-soft: #f6e6cc; --bad: #b0352b; --bad-soft: #f5dcd9;
  --display: "Bricolage Grotesque", "Avenir Next", system-ui, sans-serif;
  --body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper: #11171d; --panel: #161e26; --ink: #e6ebf0; --muted: #96a3b0; --rule: #2a3540;
  --old: #6fa3d8; --old-soft: #20364d; --new: #f0a83a; --new-soft: #4a3412; --idle: #222c36;
  --good: #5cc28a; --good-soft: #17352a; --warn: #e6a64a; --warn-soft: #3d2c12; --bad: #ef7d73; --bad-soft: #43201d;
  color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --paper: #11171d; --panel: #161e26; --ink: #e6ebf0; --muted: #96a3b0; --rule: #2a3540;
  --old: #6fa3d8; --old-soft: #20364d; --new: #f0a83a; --new-soft: #4a3412; --idle: #222c36;
  --good: #5cc28a; --good-soft: #17352a; --warn: #e6a64a; --warn-soft: #3d2c12; --bad: #ef7d73; --bad-soft: #43201d;
  color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background: var(--paper); color: var(--ink); font: 17px/1.6 var(--body); margin: 0; padding-inline: 16px; }}
main {{ max-width: 760px; margin: 0 auto; padding-block: 36px 80px; min-width: 0; }}
header {{ display: grid; gap: 14px; padding-bottom: 28px; margin-bottom: 36px; border-bottom: 1px solid var(--rule); }}
.eyebrow {{ font: 600 0.72rem/1 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
h1 {{ font: 700 clamp(2.1rem, 5vw, 3rem)/1.08 var(--display); letter-spacing: -0.015em; margin: 0; text-wrap: balance; }}
.standfirst {{ font-size: 1.15rem; line-height: 1.5; color: var(--muted); margin: 0; max-width: 60ch; }}
h2 {{ font: 700 1.5rem/1.2 var(--display); margin: 2.4em 0 0.7em; text-wrap: balance; }}
h3 {{ font: 600 1.05rem/1.3 var(--display); margin: 0 0 8px; }}
p {{ margin: 0 0 1.1em; max-width: 68ch; }}
ul {{ padding-left: 1.2em; margin: 0 0 1.2em; }} li {{ margin-bottom: 0.55em; max-width: 66ch; }} li::marker {{ color: var(--old); }}
strong {{ font-weight: 600; }}
code {{ font: 0.84em var(--mono); background: var(--panel); border: 1px solid var(--rule); border-radius: 4px; padding: 0.05em 0.35em; }}
.scroll {{ overflow-x: auto; margin: 0 0 1.4em; }}
.panel {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 16px; min-width: 0; margin: 0 0 16px; }}
.panel .scroll {{ margin: 0; }}
.wide {{ margin-inline: calc(50% - min(50vw - 16px, 540px)); }}
@media (max-width: 1120px) {{ .wide {{ margin-inline: 0; }} }}
table {{ border-collapse: collapse; font: 0.8rem/1.45 var(--mono); font-variant-numeric: tabular-nums; width: 100%; min-width: 0; }}
th {{ text-align: right; font-weight: 600; color: var(--muted); padding: 6px 10px; border-bottom: 1px solid var(--rule); white-space: normal; vertical-align: bottom; }}
td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; white-space: normal; min-width: 10ch; }}
td.l, th.l {{ text-align: left; }}
tr:last-child td {{ border-bottom: 0; }}
.grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap: 16px; margin: 0 0 16px; }}
.grid2 > .panel {{ margin: 0; }}
.grid3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 16px; margin: 0 0 16px; }}
.grid3 > .panel {{ margin: 0; }}
.chart {{ display: grid; gap: 8px; min-width: 0; align-content: start; }}
.chart svg {{ width: 100%; height: auto; display: block; }}
.cap {{ font-size: 0.86rem; line-height: 1.45; color: var(--muted); max-width: 68ch; margin: 0; }}
.note {{ font-size: 0.92rem; color: var(--muted); max-width: 72ch; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 14px; font: 0.72rem/1.2 var(--mono); color: var(--muted); }}
.legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.legend i {{ width: 14px; height: 3px; border-radius: 2px; display: inline-block; }}
svg text {{ font-family: var(--mono); font-size: 10.5px; fill: var(--muted); }}
svg .axis {{ stroke: var(--rule); stroke-width: 1; fill: none; }}
svg .grid {{ stroke: var(--rule); stroke-width: 1; stroke-dasharray: 2 3; fill: none; opacity: 0.8; }}
.ptable th {{ white-space: nowrap; }}
.ptable {{ font-size: 0.8rem; }} .ptable td, .ptable th {{ padding: 4px 10px; }}
.ptable small {{ color: var(--muted); font-size: 0.82em; white-space: nowrap; }}
.ptable td.l {{ white-space: nowrap; }}
.tslider {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; font: 0.8rem var(--mono); color: var(--muted); }}
.tslider input {{ flex: 1 1 200px; min-width: 0; accent-color: var(--old); }}
.tslider output {{ font: 600 0.9rem var(--mono); color: var(--ink); min-width: 8ch; font-variant-numeric: tabular-nums; }}
.tslider button, .seg button {{ font: 600 0.75rem/1 var(--mono); padding: 7px 11px; border-radius: 999px; border: 1px solid var(--rule); background: var(--panel); color: var(--muted); cursor: pointer; }}
.seg button[aria-pressed="true"] {{ background: var(--ink); color: var(--paper); border-color: var(--ink); }}
button:focus-visible, input[type=range]:focus-visible {{ outline: 2px solid var(--old); outline-offset: 2px; }}
.pill {{ display: inline-flex; align-items: center; font: 600 0.72rem/1 var(--mono); padding: 5px 9px; border-radius: 999px; border: 1px solid var(--rule); background: var(--panel); color: var(--muted); white-space: nowrap; }}
.pill.good {{ background: var(--good-soft); border-color: transparent; color: var(--good); }}
footer {{ margin-top: 48px; padding-top: 18px; border-top: 1px solid var(--rule); font: 0.85rem/1.5 var(--mono); color: var(--muted); }}
@media (max-width: 640px) {{ body {{ font-size: 16px; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
{explainer_css}
#answers .v {{ color: var(--old); font-weight: 600; }} #answers .n {{ color: var(--new); font-weight: 600; }}
#answers {{ margin-block: 0 24px; }}
</style>
<main>
  <header>
    <div class="eyebrow">Async RL on vLLM 0.30 · Qwen3-0.6B-Base · two H100 SXM, and one H100 NVL for the teacher-forcing study</div>
    <h1>{title}</h1>
    <p class="standfirst">{standfirst}</p>
  </header>
{content}
  <footer>Every number here comes from data/ and docs/analysis/: the GSM8K and MATH panels from the two_gpu-2026-10-03 grids (two H100 SXM) and their two analysis files, the teacher-forcing panels from the h100nvl-2026-10-01 study (one H100 NVL). Charts are drawn from the data inlined below.</footer>
</main>

<script>
const DATA = {json.dumps(DATA)};
const $ = (id) => document.getElementById(id);
const cssRoot = () => document.querySelector(".hw") || document.documentElement;  // the tokens' scope: the page's root, or the site's .hw
const css = (v) => getComputedStyle(cssRoot()).getPropertyValue(v).trim();
const fmt = (v, d = 3) => v === null || v === undefined || Number.isNaN(v) ? "" :
  (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString("en") : (Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(1) : (+v).toPrecision(d)));
const pct = (v, d = 0) => v === null || v === undefined ? "" : `${{(100 * v).toFixed(d)}}%`;
const NAMES = {{ A: "A sync", B: "B drain", D: "D stale KV", E: "E fresh KV", F: "F versioned LoRA" }};
const COLOR = () => ({{ A: css("--muted"), B: css("--good"), D: css("--new"), E: css("--old") }});
const ORDER = ["A", "B", "D", "E"];
const mean = (v) => v.reduce((s, x) => s + x, 0) / v.length;

{line_chart_js}

// -- accuracy by wall clock with a draggable T: the mean over seeds, the seeds' range as a band, the table at T
const TQ95_TABLE = {{ 1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086 }};
const TQ95 = (df) => TQ95_TABLE[df] || 1.96;
const pmSteps = (v) => {{ const n = v.length; if (n < 2) return ""; const mu = mean(v), h = TQ95(n - 1) * Math.sqrt(v.reduce((a, x) => a + (x - mu) ** 2, 0) / (n - 1)) / Math.sqrt(n); return ` <small>±${{Math.round(h).toLocaleString("en")}}</small>`; }};
function curvePanel(el, exp, key) {{
  const arms = ORDER.filter(a => exp.curves[a]); const Tmax = exp.T_s / 60;
  el.innerHTML = `<h3>Held-out accuracy on ${{exp.task}}</h3><div class="pchart"></div>
    <div class="tslider"><label for="${{key}}-T">T</label><input id="${{key}}-T" type="range" min="0" max="${{Tmax}}" step="0.25" value="${{Tmax}}">
      <output id="${{key}}-T-out" for="${{key}}-T">${{Tmax}} min</output></div>
    <div class="scroll"><table class="ptable"></table></div>`;
  const input = el.querySelector("input"), out = el.querySelector("output");
  const at = (pts, T) => {{ for (let i = 1; i < pts.length; i++) if (pts[i].t >= T) {{ const a = pts[i - 1], b = pts[i]; return a.mean + (b.mean - a.mean) * (T - a.t) / (b.t - a.t || 1); }} return pts[pts.length - 1].mean; }};
  function draw() {{
    const T = +input.value, C = COLOR(); out.textContent = `${{T.toFixed(2).replace(/\\.?0+$/, "")}} min`;
    const W = 640, H = 210, m = {{ l: 46, r: 12, t: 12, b: 36 }};
    const all = arms.flatMap(a => exp.curves[a].points.map(p => p.mean));
    const y0 = Math.floor((Math.min(...all) - 0.01) * 20) / 20, y1 = Math.ceil((Math.max(...all) + 0.01) * 20) / 20;
    const sx = (v) => m.l + (W - m.l - m.r) * v / Tmax, sy = (v) => H - m.b - (H - m.t - m.b) * (v - y0) / (y1 - y0);
    let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{exp.label}}: accuracy against minutes, T at ${{T}} minutes">`;
    svg += `<rect x="${{m.l}}" y="${{m.t}}" width="${{(sx(T) - m.l).toFixed(1)}}" height="${{H - m.t - m.b}}" fill="${{css("--idle")}}" opacity="0.45"/>`;
    for (let v = y0; v <= y1 + 1e-9; v += 0.05) svg += `<line class="grid" x1="${{m.l}}" x2="${{W - m.r}}" y1="${{sy(v)}}" y2="${{sy(v)}}"/><text x="${{m.l - 6}}" y="${{sy(v) + 3.5}}" text-anchor="end">${{Math.round(v * 100)}}%</text>`;
    for (let v = 0; v <= Tmax; v += Tmax > 6 ? 2 : 1) svg += `<text x="${{sx(v)}}" y="${{H - m.b + 15}}" text-anchor="middle">${{v}}</text>`;
    svg += `<line class="axis" x1="${{m.l}}" x2="${{W - m.r}}" y1="${{H - m.b}}" y2="${{H - m.b}}"/><text x="${{(m.l + W - m.r) / 2}}" y="${{H - 5}}" text-anchor="middle">minutes of training</text>`;
    for (const a of arms) {{
      const pts = exp.curves[a].points.map(p => ({{ ...p, t: p.t / 60 }}));
      svg += `<path d="${{pts.map((p, i) => `${{i ? "L" : "M"}}${{sx(p.t).toFixed(1)}},${{sy(p.mean).toFixed(1)}}`).join("")}}" fill="none" stroke="${{C[a]}}" stroke-width="2"/>`;
      svg += `<circle cx="${{sx(T).toFixed(1)}}" cy="${{sy(at(pts, T)).toFixed(1)}}" r="4" fill="${{C[a]}}"/>`;
    }}
    svg += `<line x1="${{sx(T)}}" x2="${{sx(T)}}" y1="${{m.t}}" y2="${{H - m.b}}" stroke="${{css("--ink")}}" stroke-width="1.5"/></svg>`;
    el.querySelector(".pchart").innerHTML = svg;
    // per seed, accuracy at T; a 95% t-interval over seeds
    const TQ = {{ 1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086 }};
    const tq = (df) => TQ[df] || 1.96;
    const seedPts = (sd) => sd.points.map(([t, acc]) => ({{ t: t / 60, mean: acc }}));
    const perSeed = (a, f) => Object.fromEntries(exp.curves[a].seeds.map(sd => [sd.seed, f(seedPts(sd), T)]));
    const interval = (xs) => {{ const n = xs.length; if (n < 2) return null; const mu = mean(xs), sd = Math.sqrt(xs.reduce((s, x) => s + (x - mu) ** 2, 0) / (n - 1)), h = tq(n - 1) * sd / Math.sqrt(n); return [mu - h, mu + h]; }};
    const pct = (v) => (100 * v).toFixed(1);
    const pm = (c) => c ? ` <small>±${{pct((c[1] - c[0]) / 2)}}</small>` : "";
    const cell = (xs) => `${{pct(mean(xs))}}%` + pm(interval(xs));
    el.querySelector(".ptable").innerHTML = `<tr><th class="l">Regime</th><th>Steps / h</th><th>Accuracy at T</th></tr>` +
      arms.map(a => {{ const c = exp.curves[a]; return `<tr><td class="l"><i style="display:inline-block;width:12px;height:3px;background:${{C[a]}};vertical-align:middle;margin-right:6px"></i>${{NAMES[a]}}</td><td>${{fmt(mean(c.steps_per_hour))}}${{pmSteps(c.steps_per_hour)}}</td><td>${{cell(Object.values(perSeed(a, at)))}}</td></tr>`; }}).join("");
  }}
  input.addEventListener("input", draw);
  draw(); return draw;
}}

// -- the contrasts as a forest plot: a dot at the difference, a line over the 95% interval, zero marked
function forest(el, exp) {{
  const rows = exp.family; const W = 380, rh = 30, m = {{ l: 54, r: 92, t: 12, b: 44 }}, H = m.t + rows.length * rh + m.b;
  const lo = Math.min(0, ...rows.map(r => 100 * r.ci95[0])) - 1, hi = Math.max(0, ...rows.map(r => 100 * r.ci95[1])) + 1;
  const sx = (v) => m.l + (W - m.l - m.r) * (v - lo) / (hi - lo);
  let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{exp.label}}: the contrasts with 95% intervals">`;
  svg += `<line x1="${{sx(0)}}" x2="${{sx(0)}}" y1="${{m.t - 8}}" y2="${{H - m.b + 4}}" stroke="${{css("--ink")}}" stroke-width="1" opacity="0.6"/>`;
  if (exp.delta) for (const d of [-100 * exp.delta, 100 * exp.delta]) svg += `<line class="grid" x1="${{sx(d)}}" x2="${{sx(d)}}" y1="${{m.t - 8}}" y2="${{H - m.b + 4}}"/>`;
  rows.forEach((r, i) => {{ const y = m.t + i * rh + rh / 2, good = r.verdict === "superior";
    const c = good ? css("--good") : (r.verdict === "interval only" ? css("--old") : css("--muted"));
    svg += `<text x="${{m.l - 8}}" y="${{y + 4}}" text-anchor="end" style="fill:${{css("--ink")}}">${{r.contrast}}</text>`;
    svg += `<line x1="${{sx(100 * r.ci95[0]).toFixed(1)}}" x2="${{sx(100 * r.ci95[1]).toFixed(1)}}" y1="${{y}}" y2="${{y}}" stroke="${{c}}" stroke-width="2.5"/>`;
    svg += `<circle cx="${{sx(100 * r.diff).toFixed(1)}}" cy="${{y}}" r="4.5" fill="${{c}}"/>`;
    svg += `<text x="${{W - m.r + 8}}" y="${{y + 4}}">${{(100 * r.diff) >= 0 ? "+" : ""}}${{(100 * r.diff).toFixed(1)}} ${{r.verdict === "interval only" ? "" : (good ? "superior" : "n.s.")}}</text>`; }});
  for (const v of [Math.ceil(lo), 0, Math.floor(hi)]) svg += `<text x="${{sx(v)}}" y="${{H - 24}}" text-anchor="middle">${{v > 0 ? "+" : ""}}${{v}}</text>`;
  svg += `<text x="${{(m.l + W - m.r) / 2}}" y="${{H - 6}}" text-anchor="middle">difference in area to T, points</text></svg>`;
  const eq = exp.equivalence.map(c => `${{c.contrast}} ${{c.family === 2 ? "at matched steps" : "over wall clock"}}: ${{c.result}}${{c.verdict ? ", " + c.verdict : ""}}${{c.ci90 ? ` [${{(100 * c.ci90[0]).toFixed(1)}}, ${{(100 * c.ci90[1]).toFixed(1)}}]` : ""}}`).join("; ");
  el.innerHTML = `<h3>${{exp.label}}: the contrasts</h3>${{svg}}<p class="cap">${{exp.delta ? `Dashed lines: ±${{100 * exp.delta}} points, the equivalence margin. ` : ""}}Paired over seed blocks, 95% intervals${{exp.intervals_only ? ", reported as intervals only (an exploratory tier)" : ", Holm-adjusted"}}. Equivalence: ${{eq}}.</p>`;
}}

// -- throughput: optimiser steps per hour, one dot per row, a bar at the mean, the sync's cost beside it
function strip(el, exp) {{
  const arms = ORDER.filter(a => exp.curves[a]), C = COLOR(); const W = 380, H = 260, m = {{ l: 48, r: 8, t: 18, b: 40 }}, step = (W - m.l - m.r) / arms.length;
  const all = arms.flatMap(a => exp.curves[a].steps_per_hour); const y1 = Math.ceil(Math.max(...all) * 1.1 / 200) * 200; const sy = (v) => H - m.b - (H - m.t - m.b) * v / y1;
  let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{exp.label}}: optimiser steps per hour by regime">`;
  for (let v = 0; v <= y1; v += y1 > 1200 ? 400 : 100) svg += `<line class="grid" x1="${{m.l}}" x2="${{W - m.r}}" y1="${{sy(v)}}" y2="${{sy(v)}}"/><text x="${{m.l - 6}}" y="${{sy(v) + 3.5}}" text-anchor="end">${{v.toLocaleString("en")}}</text>`;
  svg += `<line class="axis" x1="${{m.l}}" x2="${{W - m.r}}" y1="${{H - m.b}}" y2="${{H - m.b}}"/>`;
  arms.forEach((a, i) => {{ const v = exp.curves[a].steps_per_hour, x = m.l + step * (i + 0.5), mu = mean(v);
    svg += `<text x="${{x}}" y="${{H - m.b + 15}}" text-anchor="middle">${{a}}</text><line x1="${{x - step * 0.3}}" x2="${{x + step * 0.3}}" y1="${{sy(mu).toFixed(1)}}" y2="${{sy(mu).toFixed(1)}}" stroke="${{C[a]}}" stroke-width="3"/>`;
    svg += `<text x="${{x}}" y="${{(sy(mu) - 8).toFixed(1)}}" text-anchor="middle" style="fill:${{C[a]}}">${{Math.round(mu).toLocaleString("en")}}</text>`;
    v.forEach((s, j) => {{ const jx = x + (v.length > 1 ? (j / (v.length - 1) - 0.5) * step * 0.45 : 0); svg += `<circle cx="${{jx.toFixed(1)}}" cy="${{sy(s).toFixed(1)}}" r="3.2" fill="${{C[a]}}" opacity="0.75"/>`; }}); }});
  svg += `</svg>`;
  const tbl = `<div class="scroll"><table><tr><th class="l">Regime</th><th>Steps / h</th><th>Sync p50</th><th>Pause p50</th></tr>` + arms.map(a => {{ const c = exp.curves[a]; return `<tr><td class="l">${{NAMES[a]}}</td><td>${{fmt(mean(c.steps_per_hour))}}${{pmSteps(c.steps_per_hour)}}</td><td>${{Math.round(1000 * c.sync_p50).toLocaleString("en")}} ms</td><td>${{Math.round(1000 * c.pause_p50)}} ms</td></tr>`; }}).join("") + "</table></div>";
  el.innerHTML = `<h3>${{exp.label}}: optimiser steps per hour</h3>${{svg}}${{tbl}}<p class="cap">One dot per row. A sync is the engine's pause to its resume with the push inside it; the pause alone is the fence's wait for every counted token. The drain's pause is its wait for in-flight answers to finish.</p>`;
}}

// -- the MATH synchronous curve by optimiser step, with answer length
function syncCurve(el) {{
  const pts = DATA.math_sync_curve, band = (p) => 1.96 * Math.sqrt(p.acc * (1 - p.acc) / p.n);
  lineChart(el, {{ title: "MATH, arm A from the base model: accuracy by optimiser step (500 development problems)", xticks: [0, 50, 100, 150, 200], yticks: [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55], ymin: 0.24, ymax: 0.56, xmin: 0, xmax: 200,
    xlabel: "optimiser steps from the base model", ylabel: "greedy accuracy (95% interval)", yfmt: (v) => `${{Math.round(100 * v)}}%`,
    series: [{{ name: "accuracy", color: css("--new"), points: pts.map(p => ({{ x: p.step, y: p.acc, lo: p.acc - band(p), hi: p.acc + band(p) }})) }}],
    caption: `The whole gain comes by step 40. Mean answer length: ${{pts[0].tokens}} tokens at step 0, ${{pts.find(p => p.step === 40).tokens}} at step 40, ${{pts.find(p => p.step === 80).tokens}} at step 80, ${{pts[pts.length - 1].tokens}} at step 200: the next 160 steps add length and no accuracy. No settled snapshot leaves the warm-start rule its ten points of headroom, so the MATH tier starts at the base model.` }});
}}

// -- exposure: how many answers were in flight at a swap, where the swap landed, and the stale-token share
function exposure(el) {{
  const E = DATA.exposure, C = {{ gsm8k: css("--old"), math: css("--new") }}; const W = 520, H = 230, m = {{ l: 40, r: 12, t: 28, b: 36 }};
  const top = Math.max(...["gsm8k", "math"].map(t => {{ const h = E[t].swap_position_hist, n = h.reduce((s, x) => s + x, 0); return Math.max(...h) / n; }}));
  const ymax = Math.ceil(top * 20) / 20;
  const sx = (i) => m.l + (W - m.l - m.r) * i / 10, sy = (v) => H - m.b - (H - m.t - m.b) * v / ymax;
  let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="Where the swap landed within the answers that crossed one">`;
  for (let v = 0; v <= ymax + 1e-9; v += 0.1) svg += `<line class="grid" x1="${{m.l}}" x2="${{W - m.r}}" y1="${{sy(v)}}" y2="${{sy(v)}}"/><text x="${{m.l - 6}}" y="${{sy(v) + 3.5}}" text-anchor="end">${{Math.round(100 * v)}}%</text>`;
  for (const [task, off] of [["gsm8k", 0], ["math", 1]]) {{ const h = E[task].swap_position_hist, n = h.reduce((s, x) => s + x, 0);
    h.forEach((c, i) => {{ const w = (W - m.l - m.r) / 10 * 0.42; svg += `<rect x="${{(sx(i) + 2 + off * w).toFixed(1)}}" y="${{sy(c / n).toFixed(1)}}" width="${{w.toFixed(1)}}" height="${{(sy(0) - sy(c / n)).toFixed(1)}}" fill="${{C[task]}}" opacity="0.85"/>`; }}); }}
  for (let i = 0; i <= 10; i += 2) svg += `<text x="${{sx(i)}}" y="${{H - m.b + 15}}" text-anchor="middle">${{i * 10}}%</text>`;
  svg += `<line class="axis" x1="${{m.l}}" x2="${{W - m.r}}" y1="${{H - m.b}}" y2="${{H - m.b}}"/><text x="${{(m.l + W - m.r) / 2}}" y="${{H - 5}}" text-anchor="middle">how far through the answer the swap landed</text></svg>`;
  const row = (task, name) => {{ const e = E[task]; return `<tr><td class="l"><i style="display:inline-block;width:12px;height:3px;background:${{C[task]}};vertical-align:middle;margin-right:6px"></i>${{name}}</td><td>${{e.mean_duration_s}} s</td><td>${{e.p90_duration_s}} s</td><td>${{e.sync_every_s}} s</td><td>${{pct(e.crossed_share, 1)}}</td><td>${{pct(e.stale_share, 1)}}</td></tr>`; }};
  el.innerHTML = `<h3>Exposure: which answers were half-written at a swap (regime D, seed 0)</h3><div class="scroll"><table><tr><th class="l">Task</th><th>Mean answer</th><th>p90 answer</th><th>Sync every</th><th>Answers crossing a swap</th><th>Tokens over stale KV</th></tr>${{row("gsm8k", "GSM8K")}}${{row("math", "MATH")}}</table></div>${{svg}}
    <div class="legend"><span><i style="background:${{C.gsm8k}}"></i>GSM8K, ${{E.gsm8k.crossed}} of ${{E.gsm8k.answers.toLocaleString("en")}} answers</span><span><i style="background:${{C.math}}"></i>MATH, ${{E.math.crossed.toLocaleString("en")}} of ${{E.math.answers.toLocaleString("en")}} answers</span></div>
    <p class="cap">From the ledger: every answer's tokens are segmented by the weights that sampled them and the cache they read. On GSM8K an answer lasts under a second and the trainer step, the bottleneck, two seconds, so generation runs in a burst after each push and almost nothing is in flight at the next; the few answers that do cross a swap cross it near their end. On MATH answers last seconds against a nine-second step, so swaps land anywhere, and most of a long answer's tokens come after one. Exposure is a property of answer duration against the sync interval, not of the regime.</p>`;
}}

// -- Phase 1: how wrong a stale cache is per token, from teacher forcing checkpoints k steps apart
function phase1() {{
  const S = DATA.phase1_story; if (!S) return;
  const pct = (v) => `${{Math.round(100 * v)}}%`;
  const binLabel = Object.fromEntries(S.fade.bins.map(b => [b.x, b.label]));
  lineChart($("p1-fade"), {{ title: "Share of the update undone, by distance from the swap", xlog: true, xmin: Math.log2(1.5), xmax: Math.log2(120), xticks: S.fade.bins.map(b => b.x), xfmt: (v) => binLabel[v], yticks: [0, 0.2, 0.4, 0.6, 0.8, 1], ymin: 0, ymax: 1, yfmt: pct,
    xlabel: "tokens generated since the swap", ylabel: "share undone",
    series: [{{ name: "cache 1 to 32 optimiser steps older than the weights, pooled; band: 95% over prompts", color: css("--old"), points: S.fade.all }}],
    caption: "" }});
  barChart($("p1-bands"), {{ title: "Where the staleness lives, by layer", xlabel: "layers whose cache is recomputed (of 28)", ylabel: "share of the stale divergence removed", yfmt: pct,
    bars: S.bands.map(b => ({{ label: b.label, y: b.share, lo: b.lo, hi: b.hi }})), color: css("--old") }});
}}
function barChart(el, opt) {{
  const W = 520, H = 300, m = {{ l: 58, r: 14, t: 16, b: 42 }}, n = opt.bars.length, slot = (W - m.l - m.r) / n, bw = slot * 0.56;
  const y1 = Math.max(1, ...opt.bars.map(b => b.hi ?? b.y)); const sy = (v) => H - m.b - (H - m.t - m.b) * v / y1;
  let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{opt.title}}">`;
  for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1]) svg += `<line class="grid" x1="${{m.l}}" x2="${{W - m.r}}" y1="${{sy(v)}}" y2="${{sy(v)}}"/><text x="${{m.l - 6}}" y="${{sy(v) + 3.5}}" text-anchor="end">${{opt.yfmt(v)}}</text>`;
  opt.bars.forEach((b, i) => {{ const x = m.l + slot * (i + 0.5);
    svg += `<rect x="${{(x - bw / 2).toFixed(1)}}" y="${{sy(b.y).toFixed(1)}}" width="${{bw.toFixed(1)}}" height="${{(sy(0) - sy(b.y)).toFixed(1)}}" fill="${{opt.color}}" opacity="0.85"><title>layers ${{b.label}}: ${{opt.yfmt(b.y)}}</title></rect>`;
    if (b.lo !== undefined) svg += `<line x1="${{x}}" x2="${{x}}" y1="${{sy(b.lo)}}" y2="${{sy(b.hi)}}" stroke="${{css("--ink")}}" stroke-width="1.5"/><line x1="${{x - 5}}" x2="${{x + 5}}" y1="${{sy(b.lo)}}" y2="${{sy(b.lo)}}" stroke="${{css("--ink")}}"/><line x1="${{x - 5}}" x2="${{x + 5}}" y1="${{sy(b.hi)}}" y2="${{sy(b.hi)}}" stroke="${{css("--ink")}}"/>`;
    svg += `<text x="${{x}}" y="${{Math.min(sy(b.hi ?? b.y) - 6, sy(b.y) - 6)}}" text-anchor="middle" style="fill:${{css("--ink")}}">${{opt.yfmt(b.y)}}</text><text x="${{x}}" y="${{H - m.b + 15}}" text-anchor="middle">${{b.label}}</text>`; }});
  svg += `<line class="axis" x1="${{m.l}}" x2="${{W - m.r}}" y1="${{H - m.b}}" y2="${{H - m.b}}"/><text x="${{(m.l + W - m.r) / 2}}" y="${{H - 6}}" text-anchor="middle">${{opt.xlabel}}</text>`;
  svg += `<text transform="translate(12 ${{(m.t + H - m.b) / 2}}) rotate(-90)" text-anchor="middle">${{opt.ylabel}}</text></svg>`;
  el.innerHTML = `<h3>${{opt.title}}</h3>${{svg}}`;
}}

function canaryTable() {{
  const el = $("canary-table"); if (!el) return; const C = DATA.canary;
  el.innerHTML = `<tr><th class="l">After the swap</th><th>Swap at token</th><th>Matches OLD</th><th>Matches STALE</th><th>Matches FRESH</th><th>Decisive positions</th><th class="l">Its own hypothesis there</th></tr>` +
    [["D", "D: keep, cache kept", "STALE"], ["E", "E: keep, cache cleared", "FRESH"]].map(([k, name, own]) => {{ const c = C[k]; return `<tr><td class="l">${{name}}</td><td>${{c.boundary}} of ${{c.total}}</td><td>${{pct(c.old)}}</td><td>${{pct(c.stale)}}</td><td>${{pct(c.fresh)}}</td><td>${{c.decisive}}</td><td class="l"><span class="pill good">${{own}} ${{pct(c.own_on_decisive)}}</span></td></tr>`; }}).join("");
}}

const redraws = [];
function renderAll() {{
  redraws.length = 0;
  for (const [key, ids] of [["gsm8k", ["gsm8k-curves", "gsm8k-forest", "gsm8k-strip"]], ["math", ["math-curves", "math-forest", "math-strip"]]]) {{
    const exp = DATA.experiments[key];
    if ($(ids[0])) redraws.push(curvePanel($(ids[0]), exp, key));
    if ($(ids[1])) forest($(ids[1]), exp);
    if ($(ids[2])) strip($(ids[2]), exp);
  }}
  if ($("math-sync")) syncCurve($("math-sync"));
  if ($("exposure")) exposure($("exposure"));
  if ($("p1-fade")) phase1();
  if ($("canary-table")) canaryTable();
}}
renderAll();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
new MutationObserver(renderAll).observe(document.documentElement, {{ attributes: true, attributeFilter: ["data-theme"] }});
</script>
<script>
{explainer_js}
</script>
'''
CHARTS.write_text(page)
print("bytes", len(page.encode()), "| panels placed", sum(1 for k in NOTE_PANELS if any(l.strip() == k for l in note_body)), "of", len(NOTE_PANELS))


# ---------------------------------------------------------------- the site's post (Astro, a Markdown collection entry)

def site_post(site: Path, note: Path) -> Path:
    """The same post for the site: frontmatter, the Markdown prose (the site renders it), and the panels, styles and
    scripts as HTML blocks that CommonMark passes through untouched (no blank line inside a block; script and style
    blocks end at their closing tag). Chart styles are scoped under .hw, the explainer keeps its #answers id, and the
    colour and font tokens come from the site's own theme variables."""
    def block(html):  # one HTML block: no blank line inside, so the Markdown parser never re-enters it
        return "\n".join(line for line in html.split("\n") if line.strip()) + "\n"

    def panel_block(html):
        return block('<div class="hw">' + html.replace('<section id="answers" class="wide">', '<section id="answers" class="hw wide">') + "</div>")

    def front(title, description):
        return f"""---
title: "{title}"
pubDatetime: 2026-10-09T12:00:00Z
description: "{description.replace('"', "'")}"
tags:
  - blog
featured: false
draft: false
---
"""
    # the note's text, word for word: its first line is the title, each placeholder line becomes its panels
    title, opening, body = note_parts(note)
    out = [front(title, opening.split(". ")[0].strip()), note_markdown(body, panel_block, skip_standfirst=False)]
    style = page.split("<style>", 1)[1].split("</style>", 1)[0]
    # the page's three token blocks go: the site's theme variables replace them (each block ends as written below)
    style = re.sub(r":root \{.*?\n\}\n", "", style, count=1, flags=re.S)
    style = re.sub(r"@media \(prefers-color-scheme: dark\) \{ :root:not\(\[data-theme=\"light\"\]\) \{.*?color-scheme: dark; \} \}\n", "", style, count=1, flags=re.S)
    style = re.sub(r":root\[data-theme=\"dark\"\] \{.*?color-scheme: dark; \}\n", "", style, count=1, flags=re.S)
    tokens = """
.hw, #answers {
  --paper: var(--background); --panel: color-mix(in srgb, var(--background) 96%, var(--foreground)); --ink: var(--foreground);
  --muted-text: color-mix(in srgb, var(--foreground) 68%, var(--background)); --rule: var(--border); --idle: color-mix(in srgb, var(--foreground) 9%, var(--background));
  --old: #2f5d8c; --old-soft: #d6e2ee; --new: #c77700; --new-soft: #f6e3c4;
  --good: #2e7d4f; --good-soft: #d9efe1; --warn: #a15c00; --warn-soft: #f6e6cc; --bad: #b0352b; --bad-soft: #f5dcd9;
  --display: var(--font-app, system-ui, sans-serif); --body: var(--font-reading, Georgia, serif); --mono: var(--font-app, ui-monospace, Menlo, monospace);
}
.hw, #answers { --muted: var(--muted-text); }
html[data-theme="dark"] .hw, html[data-theme="dark"] #answers {
  --old: #6fa3d8; --old-soft: #20364d; --new: #f0a83a; --new-soft: #4a3412;
  --good: #5cc28a; --good-soft: #17352a; --warn: #e6a64a; --warn-soft: #3d2c12; --bad: #ef7d73; --bad-soft: #43201d;
}
"""
    own = ("body", "main", "header", "h1", "h2", "footer", ".eyebrow", ".standfirst", "* ")  # the site's layout and type own these
    scoped = []
    for line in style.split("\n"):
        t_ = line.strip()
        if not t_ or t_.startswith(("/*", "@", "}", "#answers", "from", "to", "color-scheme", "--")) or "{" not in line:
            scoped.append(line); continue
        sel, _, rest_ = line.partition("{")
        if sel.strip().startswith(own) and not sel.strip().startswith(".hw"):
            continue
        sel = ", ".join(".hw " + x.strip() for x in sel.split(","))
        scoped.append(sel + " {" + rest_)
    style = tokens + "\n".join(scoped) + """
.hw p, .hw li { max-width: none; }
.hw h3 { margin-top: 0; }
.hw table { margin: 0; }
.hw .chart .scroll { margin-top: 4px; }
.hw.wide, .hw .wide, .hw #answers { width: min(100vw - 32px, 1120px); margin-inline: calc(50% - min(50vw - 16px, 560px)); }
@media (max-width: 1180px) { .hw.wide, .hw .wide, .hw #answers { width: auto; margin-inline: 0; } }
"""
    scripts = page.split("</main>", 1)[1]
    scripts = scripts.replace("renderAll();\nmatchMedia(", 'const hwInit = () => { if (document.getElementById("gsm8k-curves")) renderAll(); };\nhwInit(); document.addEventListener("astro:page-load", hwInit);\nmatchMedia(')
    scripts = scripts.replace("<script>", "<script data-astro-rerun>")
    out.append(block("<style>" + style + "</style>"))
    out.append(block(scripts))
    dest = site / "src" / "data" / "blog" / "half-written-answers.md"
    dest.write_text("\n".join(out))
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--site", default=None, help="the site's repository: also write its Markdown post")
    ap.add_argument("--note", default=None, help="with --site: the post's text, its first line the title, its placeholder "
                    "lines replaced by the panels")
    a = ap.parse_args()
    if a.site:
        if not a.note:
            ap.error("--site needs --note, the post's text")
        print(site_post(Path(a.site).expanduser(), Path(a.note)))
