"""The lab page: what the current measurements show, in one page (docs/index.html), rebuilt from data/.

    python -m bench.dashboard [--roots data] [--out docs/index.html] [--analysis FILE ...] [--readme README.md]

Gathers result rows (metrics.json, steps.parquet, trace.parquet), canary reports, noise floors and Phase 1 studies
into one JSON object and writes it into bench/dashboard.html (the page's template) at /*DATA*/. Every row, curve
and canary carries its session (the archive directory it came from, data/<session>/), and where two sessions hold
the same measurement the newer one is current.
"""
import argparse
import json
import re
from pathlib import Path

from halfwritten.results import Row, gpu_label, is_set_aside, read_jsonl, read_parquet
from halfwritten.results import rows as result_rows

HERE = Path(__file__).resolve().parent


def session_of(path: Path, root: Path) -> str:
    """The archive session a file belongs to: the first directory under the root (data/<session>/...)."""
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else root.name


def rows(roots: list[Path]) -> list[dict]:
    out = []
    for root in roots:
        for r in result_rows(root):
            row, cfg, met = r.path, r.config, r.metrics
            launch = cfg.get("launch") or {}
            out.append({
                "label": row.name, "gpu": r.gpu, "source": str(row.relative_to(root.parent)),
                "session": session_of(row, root),
                "regime": cfg["regime"], "B": cfg["B"], "G": cfg["G"], "M": cfg["M"], "K": cfg["K"],
                "task": cfg.get("task"), "lora": bool(cfg.get("lora")), "tag": cfg.get("tag"),
                "api_servers": launch.get("api_servers"), "fence_exact": bool(cfg.get("fence_exact", True)),
                "from_w0": bool(cfg.get("init_from")), "filtered": bool(cfg.get("prompt_filter")),
                "stale_kv_share": r.stale_kv_share(),
                # the exact fence's record: how many cuts, how many found every counted token delivered, and
                # whether the syncs record the count at all (a row cut by the old window fence does not)
                "cuts": {"n": len(r.syncs), "exact": sum(1 for s in r.syncs if s.get("undelivered") == 0),
                         "recorded": any(s.get("undelivered") is not None for s in r.syncs)} if r.syncs else None,
                # decision 7's harness diagnostic when the row ran outside its rule (fence p50, loop lag): shown,
                # not hidden; the analysis applies the rule, and its limit is reviewed on the data
                "harness": p.removeprefix("harness: ") if (p := r.problem()) and p.startswith("harness: ") else None,
                "budget_s": cfg.get("budget_s"),
                "colocated": launch.get("engine_gpu") == launch.get("trainer_gpu"),
                "valid": met.get("valid"), "steps": met.get("steps"), "wall_s": met.get("wall_s"),
                "steps_per_hour": met.get("steps_per_hour"), "goodput": met.get("goodput_tokens_per_s"),
                "lag_mean": met.get("lag_mean"), "fence_p50": (met.get("fence_s") or {}).get("p50"),
                "host_cpu": (met.get("host") or {}).get("main_thread_cpu_frac"),
                # the throughput figure: a sync's cost (pause to resume, median), the syncs, the raw token rate
                "sync_p50": (met.get("sync_s") or {}).get("p50"), "pause_p50": (met.get("pause_s") or {}).get("p50"),
                "syncs": met.get("syncs"), "tokens_per_s": met.get("tokens_per_s"),
            })
    return out


def curves(roots: list[Path]) -> list[dict]:
    """Held-out accuracy of every evaluated row, by optimiser steps (bench/evaluate.py's evals.jsonl). S1 has no
    metrics.json (stopped before its row was written), so curves are found by their evaluation files; each carries
    its row's name and configuration, so a curve is selected by what it is, never by its label."""
    from bench.stats import snapshot_steps
    out = []
    for root in roots:
        for f in sorted(root.rglob("evals*.jsonl")):
            if is_set_aside(f.parent):
                continue
            ev = read_jsonl(f)
            r = Row(f.parent)
            cfg = r.config
            for split in sorted({e.get("split", "test") for e in ev}):  # one curve per split: dev and test differ
                pts = sorted(({"step": s, "acc": e["accuracy"], "n": e["n"], "t": e.get("t"),
                               "tokens": e.get("mean_tokens")} for e in ev
                              if e.get("split", "test") == split and (s := snapshot_steps(e)) is not None),
                             key=lambda p: p["step"])
                if pts:
                    out.append({"label": f.parent.name + ("" if split == "test" else f" ({split})"),
                                "row": f.parent.name, "file": f.name, "gpu": r.gpu, "points": pts, "split": split,
                                "task": cfg.get("task"), "tag": cfg.get("tag"), "regime": cfg.get("regime"),
                                "lora": bool(cfg.get("lora")), "session": session_of(f, root),
                                "source": str(f.relative_to(root.parent))})
    return out


def timeline(row: Path, window: tuple[float, float] = (60.0, 90.0)) -> dict:
    """Engine, trainer and sender states over a window of the run (seconds from its first event)."""
    states = Row(row).states()
    if not states:
        return {}
    t0 = min(e["t0"] for e in states)
    lanes = {("controller", "engine"): "engine", ("trainer", "trainer"): "trainer", ("trainer", "sender"): "sender"}
    out = {name: [] for name in lanes.values()}
    a, b = window
    for e in states:
        name = lanes.get((e["proc"], e["lane"]))
        s, f = e["t0"] - t0, e["t1"] - t0
        if name and f > a and s < b:
            out[name].append([round(max(s, a) - a, 3), round(min(f, b) - a, 3), e["name"]])
    return {"window": [a, b], "lanes": out}


def canary_time(name: str, session: str) -> int:
    """When a canary ran, from its directory's name: the queue names one by its epoch (canary-1790944919), a hand
    session by the clock (canary-exact-115416, within the session's date); 0 when the name says nothing."""
    if m := re.search(r"-(\d{10})$", name):
        return int(m.group(1))
    if (m := re.search(r"-(\d{2})(\d{2})(\d{2})$", name)) and (d := re.search(r"(\d{4}-\d{2}-\d{2})", session)):
        from datetime import datetime, timezone
        day = datetime.strptime(d.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(day.timestamp()) + 3600 * int(m.group(1)) + 60 * int(m.group(2)) + int(m.group(3))
    return 0


def canaries(roots: list[Path]) -> list[dict]:
    out = []
    for root in roots:
        for d in sorted(root.rglob("canary-load-*")):  # loaded canaries that died before writing a report
            if d.is_dir() and not (d / "report.json").exists():
                died = "Meta tensors" in (d / "server.log").read_text() if (d / "server.log").exists() else False
                out.append({"gpu": gpu_label(str(d)), "source": str(d.relative_to(root.parent)), "regime": "D, E",
                            "session": session_of(d, root), "at": canary_time(d.name, session_of(d, root)),
                            "loaded": True, "passed": False, "boundary": None,
                            "note": "the engine core died: weights pushed under live generation (probe bug, fixed)"
                            if died else "no report"})
        for rep in sorted(root.rglob("report.json")):
            if "canary" not in rep.parent.name and "probe" not in rep.parent.name:
                continue
            r = json.loads(rep.read_text())
            if "canary_D" not in r:
                continue
            gpu = r.get("gpu") or gpu_label(str(rep))
            session = session_of(rep, root)
            again = rep.parent / "reclassified.json"  # judged again under the current gate (bench.canary --reclassify)
            judged = json.loads(again.read_text()) if again.exists() else {}
            for reg, want in (("D", "stale"), ("E", "fresh")):
                c = r[f"canary_{reg}"]
                decisive = judged.get(f"canary_{reg}") or c  # reports from the decisive gate carry these themselves
                # the verdict under the decisive gate: a reclassification's, else the report's own when the gate
                # that wrote it was already the decisive one (it then records the decisive positions itself)
                now = (not judged.get("canary_failures")) if again.exists() else \
                    (not r.get("canary_failures")) if "decisive_stale_fresh" in c else None
                out.append({"gpu": gpu, "source": str(rep.relative_to(root.parent)), "regime": reg, "want": want,
                            "session": session, "at": canary_time(rep.parent.name, session),
                            "old": c.get("post_old"), "fresh": c.get("post_fresh"), "stale": c.get("post_stale"),
                            "pre_old": c.get("pre_old"), "boundary": c.get("boundary"), "total": c.get("total"),
                            "verified": c.get(f"boundary_verified_{want}"),
                            "rule": "2 bf16 ulps" if "misses" in c else "0.05 logits",
                            "loaded": bool(c.get("load_streams")), "load": c.get("load_streams"),
                            "passed": not r.get("canary_failures"),
                            "passed_now": now, "decisive": decisive.get("decisive_stale_fresh"),
                            "own_on_decisive": decisive.get(f"post_{want}_on_decisive"),
                            # older reports do not record these launch settings; their directory names do
                            "setup": "8 tokens a chunk" if "si8" in rep.parent.name else
                            "4 API servers" if "api4" in rep.parent.name else ""})
    return out


def floors(roots: list[Path]) -> list[dict]:
    out = []
    for root in roots:
        for f in sorted(root.rglob("noise_floor*.json")):
            n = json.loads(f.read_text())
            gpu = n.get("gpu") or gpu_label(str(f))
            head = "fp32" if n.get("head_fp32") else "bf16"
            trainer = "bf16 (bug)" if "a100" in str(f) else "fp32 (fixed)"  # the head fix landed after the A100
            for cond, part in (("steady", "before"), ("steady_forced", "before"), ("noop_keep", "after"),
                               ("noop_clear", "after")):
                x = n.get(cond, {}).get(part, {})
                if x.get("tokens"):
                    out.append({"gpu": gpu, "engine_head": head, "trainer_head": trainer, "condition": cond,
                                "tokens": x["tokens"], "k3": x["k3"], "p50": x["abs_p50"], "p99": x["abs_p99"],
                                "max": x["abs_max"], "source": str(f.relative_to(root.parent))})
    return out


def phase1(roots: list[Path]) -> list[dict]:
    from bench.stale_kv import phase1_table
    out = []
    for root in roots:
        for f in sorted(root.rglob("positions.parquet")):
            d = f.parent
            name = d.name
            summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
            gpu = summary.get("gpu") or gpu_label(name if "h100" in name else "a100")
            compute = "fp32" if "fp32compute" in name else "bf16"
            table = phase1_table(d)
            pos = read_parquet(f)
            decay = {}
            for k in sorted({p["lag"] for p in pos if p["lag"] > 0}):
                buckets = {}
                for p in pos:
                    if p["lag"] == k:
                        b = buckets.setdefault(int(p["offset"] // 8) * 8, [0.0, 0.0])
                        b[0] += p["kl_fresh_stale"]
                        b[1] += p["kl_fresh_old"]
                decay[k] = [[b, v[0] / v[1]] for b, v in sorted(buckets.items()) if v[1] > 0]
            out.append({"name": name, "gpu": gpu, "compute": compute, "source": str(d.relative_to(root.parent)),
                        "table": table, "decay": decay})
    return out


def sessions(roots: list[Path]) -> dict[str, str]:
    """Each archive session's order key: the date in its name (data/<layout>-<date>), else its manifest's date."""
    out = {}
    for root in roots:
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            m = d / "MANIFEST.json"
            archived = json.loads(m.read_text()).get("archived", "") if m.exists() else ""
            date = re.search(r"\d{4}-\d{2}-\d{2}", d.name)
            out[d.name] = (date.group(0) if date else archived or "") + "/" + d.name
    return out


def _unique(items: list[dict], key) -> list[dict]:
    """The archive (data/) holds copies of results/ rows: keep one of each."""
    seen, out = set(), []
    for x in items:
        k = key(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _newest_first(items: list[dict], order: dict[str, str]) -> list[dict]:
    return sorted(items, key=lambda x: order.get(x.get("session"), x.get("session") or ""), reverse=True)


def current(data: dict) -> dict:
    """What the page reports: the rows on the current harness (two GPUs, the exact fence, the controller's host
    record that decision 7's check reads; runs of three minutes or more on any task, every invariant green, the
    harness diagnostic shown where a row ran outside its rule), the newest session's where two hold the same row,
    and the curves, canaries and studies they rest on. Everything earlier stays in data/ and NOTEBOOK.md."""
    order = data.get("sessions", {})
    rows = _unique(_newest_first([r for r in data["rows"] if r["valid"] and r["host_cpu"] is not None
                                  and r["fence_exact"] and not r["colocated"] and (r["budget_s"] or 0) >= 180],
                                 order), lambda r: (r["task"], r["label"]))  # a label repeats across tasks
    keys = {(r["task"], r["label"]) for r in rows}
    tl = {}
    for r in rows:
        if r["tag"] == "pilot" and r["task"] == "gsm8k" and r["regime"] in ("A", "D") \
                and (Path(r["source"]) / "trace.parquet").exists():
            tl[r["regime"]] = timeline(Path(r["source"]), window=(300.0, 330.0)) | {"label": r["label"],
                                                                                      "gpu": r["gpu"]}
    # the current rows' curves on every split (a grid's, a pilot's, a dev rerun's) and the synchronous curve, which
    # has no row; the newest session's where two hold the same row and split
    curves = _unique(_newest_first([c for c in data["curves"] if (c["task"], c["row"]) in keys or c["tag"] == "s1"],
                                   order), lambda c: (c["task"], c["row"], c["file"], c["split"]))
    canaries = sorted([c for c in data["canaries"] if c.get("loaded") and c.get("passed_now") and c.get("decisive")],
                      key=lambda c: (order.get(c["session"], c["session"]), c["at"]))  # the last is the newest
    return data | {
        "rows": rows, "timelines": tl, "curves": curves, "canaries": canaries,
        "floors": [f for f in data["floors"] if "fixed" in f["trainer_head"]],  # the trainer's head as it now runs
        "phase1": [p for p in data["phase1"] if p["gpu"] != "A100 40GB"],
    }


def gather(roots: list[Path]) -> dict:
    roots = [r for r in roots if r.exists()]
    all_rows = _unique(rows(roots), lambda r: (r["task"], r["gpu"], r["label"], round(r["wall_s"] or 0, 3)))
    tl = {}
    for root in roots:
        for row in root.rglob("trace.parquet"):
            label = row.parent.name
            if row.parent.parent.name == "A100-SXM4-40GB" and label in ("D-B8G8M8-K2-S1-s0", "A-B16G8M16-K0-S1-s0"):
                tl[f"{label[0]}"] = timeline(row.parent) | {"label": label}
    return {"sessions": sessions(roots), "rows": all_rows, "timelines": tl,
            "curves": _unique(curves(roots), lambda c: (c["task"], c["label"], c["file"], c["points"][-1]["step"])),
            # one per canary run: its directory's name (reports are all report.json; data/ copies results/)
            "canaries": _unique(canaries(roots), lambda c: (c["gpu"], c["regime"], Path(c["source"]).parent.name)),
            "floors": _unique(floors(roots), lambda f: (f["gpu"], f["engine_head"], f["condition"], f["tokens"])),
            "phase1": _unique(phase1(roots), lambda p: (p["name"], p["table"][1]["drift_mean"] if len(p["table"]) > 1
                                                         else 0))}


def write_readme_block(readme: Path, items: list[dict]) -> None:
    """README's "What we know": the findings as Markdown between its <!-- findings --> markers."""
    from bench.findings import markdown
    text = readme.read_text()
    start, end = "<!-- findings -->\n", "<!-- /findings -->"
    a, b = text.index(start) + len(start), text.index(end)
    readme.write_text(text[:a] + markdown(items) + "\n" + text[b:])


def archived_on(roots: list[Path]) -> str | None:
    """The newest archive date among the roots' manifests: the page's 'as of'."""
    dates = []
    for root in roots:
        for m in root.rglob("MANIFEST.json"):
            try:
                dates.append(json.loads(m.read_text()).get("archived"))
            except (OSError, ValueError):
                pass
    return max(d for d in dates if d) if any(dates) else None


def main(argv=None):
    from bench.findings import findings
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--roots", nargs="*", default=["data"])
    ap.add_argument("--out", default="docs/index.html")
    ap.add_argument("--analysis", nargs="*", default=[], help="bench.stats analysis.json files: the grid's results")
    ap.add_argument("--readme", default=None, help="rewrite this README's findings block from the same findings")
    ap.add_argument("--updated", default=None, help="the 'as of' text shown on the page (default: the archive's date)")
    ap.add_argument("--links", default=None, metavar="URL",
                    help="point the page's repository links (../X) at URL + X, for a copy served outside docs/")
    ap.add_argument("--status", default=None,
                    help="a status pill in the page's header, e.g. 'provisional: 17 of 38 grid rows' (default: none)")
    a = ap.parse_args(argv)
    roots = [Path(r) for r in a.roots]
    data = current(gather(roots))
    data["updated"] = a.updated or (f"as of {archived_on(roots)}" if archived_on(roots) else "as of 2026-10-02")
    # one experiment per file, in the order given: each gets its own block on the page (findings._grid)
    analyses = [json.loads(Path(f).read_text()) for f in a.analysis]
    data["findings"] = findings(data, analyses)
    data["analyses"] = analyses  # the pre-registered run's section: one panel per experiment
    data["status"] = a.status
    if a.readme:
        write_readme_block(Path(a.readme), data["findings"])
    page = (HERE / "dashboard.html").read_text().replace("/*DATA*/", json.dumps(data, default=float))
    if a.links:
        page = page.replace('href="../', f'href="{a.links.rstrip("/")}/')
    Path(a.out).write_text(page)
    print(f"{a.out}: {len(data['rows'])} rows, {len(data['canaries'])} canary verdicts, {len(data['floors'])} floor "
          f"conditions, {len(data['phase1'])} Phase 1 studies")


if __name__ == "__main__":
    main()
