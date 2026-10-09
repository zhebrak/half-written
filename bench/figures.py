"""The paper's figures, as static SVG from the page's gathered data (bench.dashboard.current): the same numbers the
page draws, exported for the write-up.

    python -m bench.figures [--roots data] [--analysis FILE ...] --out docs/figures/

Writes what the data allows: pilot.svg (held-out accuracy against wall clock, the first pilot), phase1_R.svg (how
much of the drift a stale cache keeps, by lag), phase1_layers.svg (the share removed by recomputing the first
layers), and, once a grid has run, accuracy_vs_time-<task>.svg per task (mean over seeds with a t band, per arm, from
the main experiment's rows: no tag, the test split).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

COLORS = {"A": "#5a6774", "B": "#7a5195", "C1": "#2e7d4f", "D": "#c77700", "E": "#2f5d8c", "F": "#b0352b"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "halfwritten"})  # the same SVG from the same data
    return plt


def _newest_per(curves: list[dict], order: dict, key) -> list[dict]:
    """One curve per `key`, the newest session's (bench.dashboard.sessions gives the order)."""
    out = {}
    for c in sorted(curves, key=lambda c: order.get(c.get("session"), c.get("session") or "")):
        out[key(c)] = c
    return list(out.values())


def pilot(data: dict, out: Path) -> Path | None:
    """The first GSM8K pilot's test curves, one per arm, from the newest session that has them."""
    curves = _newest_per([c for c in data.get("curves", []) if c.get("tag") == "pilot" and c.get("task") == "gsm8k"
                          and c.get("split") == "test" and c["file"] == "evals.jsonl"],
                         data.get("sessions", {}), lambda c: c["regime"])
    if not curves:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for c in sorted(curves, key=lambda c: c["regime"]):
        arm = c["regime"]
        pts = sorted((p["t"] / 60, p["acc"]) for p in c["points"] if p.get("t") is not None)
        ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], color=COLORS.get(arm, "#333"), label=arm, lw=1.8)
    ax.set_xlabel("minutes of training from W0")
    ax.set_ylabel("held-out accuracy (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out / "pilot.svg"
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    return path


def phase1(data: dict, out: Path) -> list[Path]:
    studies = [p for p in data.get("phase1", []) if p.get("table")]
    if not studies:
        return []
    plt = _plt()
    written = []
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for s in studies:
        rows = [t for t in s["table"] if t["lag"] > 0 and t.get("R") is not None]
        lo = [float(t["R_ci95"].split("-")[0]) if t.get("R_ci95") else t["R"] for t in rows]
        hi = [float(t["R_ci95"].split("-")[1]) if t.get("R_ci95") else t["R"] for t in rows]
        x = [t["lag"] for t in rows]
        ax.fill_between(x, lo, hi, alpha=0.15, color="#c77700" if s["compute"] == "fp32" else "#2f5d8c")
        ax.plot(x, [t["R"] for t in rows], marker="o", ms=3, lw=1.6, label=f"{s['compute']} compute, {s['gpu']}",
                color="#c77700" if s["compute"] == "fp32" else "#2f5d8c")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("optimiser steps between the checkpoints (k)")
    ax.set_ylabel("R = KL(fresh‖stale) / KL(fresh‖old)")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out / "phase1_R.svg"
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    written.append(path)
    fp32 = next((s for s in studies if s["compute"] == "fp32"), None)
    if fp32:
        layers = sorted({int(m.group(1)) for t in fp32["table"] for k in t if (m := re.fullmatch(r"removed_l(\d+)", k))})
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        for k, color in ((2, "#5a6774"), (8, "#2f5d8c"), (32, "#c77700")):
            t = next((t for t in fp32["table"] if t["lag"] == k), None)
            if t:
                ax.plot([0, *layers], [0, *(100 * max(0.0, t.get(f"removed_l{l}") or 0) for l in layers)],
                        marker="o", ms=3, lw=1.6, color=color, label=f"k = {k}")
        ax.set_xlabel("layers whose KV is recomputed (of 28)")
        ax.set_ylabel("share of KL(fresh‖stale) removed (%)")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = out / "phase1_layers.svg"
        fig.savefig(path, metadata={"Date": None})
        plt.close(fig)
        written.append(path)
    return written


def accuracy_vs_time(data: dict, out: Path) -> list[Path]:
    """The grids' curves, one figure per experiment: the main experiment's rows (the test split, no tag) as
    accuracy_vs_time-<task>.svg, and every tagged experiment with at least two arms of two or more seeds (a tier
    such as MATH's math-base; never a one-seed pilot) as accuracy_vs_time-<task>-<tag>.svg, titled as exploratory:
    per arm, the mean over seeds at each mark with a 95% t band (paired-t machinery is the analysis's; this is the
    picture)."""
    from scipy import stats

    from halfwritten.regimes import resolve
    grid = [c for c in data.get("curves", []) if c["file"] == "evals.jsonl" and c.get("split") == "test"
            and c.get("task")]
    from_w0 = {r["label"]: r.get("from_w0") for r in data.get("rows", [])}
    written = []
    for task, tag in sorted({(c["task"], c.get("tag") or "") for c in grid}):
        by_arm: dict[str, list] = {}
        for c in grid:
            if c["task"] == task and (c.get("tag") or "") == tag:
                by_arm.setdefault(resolve(c["regime"], bool(c.get("lora"))).label, []).append(c)
        by_arm = {a: cs for a, cs in by_arm.items() if len(cs) >= 2}
        if not by_arm or (tag and len(by_arm) < 2):
            continue
        plt = _plt()
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        for arm, cs in sorted(by_arm.items()):
            n = min(len(c["points"]) for c in cs)
            t = np.array([[p["t"] or 0.0 for p in c["points"][:n]] for c in cs]).mean(0) / 60
            acc = np.array([[p["acc"] for p in c["points"][:n]] for c in cs])
            mean, se = acc.mean(0), acc.std(0, ddof=1) / np.sqrt(len(cs))
            band = stats.t.ppf(0.975, len(cs) - 1) * se
            ax.fill_between(t, 100 * (mean - band), 100 * (mean + band), alpha=0.15, color=COLORS.get(arm, "#333"))
            ax.plot(t, 100 * mean, lw=1.8, color=COLORS.get(arm, "#333"), label=f"{arm} (n = {len(cs)})")
        start = "W0" if any(from_w0.get(c["row"]) for cs in by_arm.values() for c in cs) else "the base model"
        ax.set_xlabel(f"minutes of training from {start}")
        ax.set_ylabel(f"held-out accuracy (%), {task}")
        if tag:
            ax.set_title(f"{task.upper()}, {tag}: an exploratory tier, intervals only", fontsize=9)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = out / (f"accuracy_vs_time-{task}-{tag}.svg" if tag else f"accuracy_vs_time-{task}.svg")
        fig.savefig(path, metadata={"Date": None})
        plt.close(fig)
        written.append(path)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--roots", nargs="*", default=["data"])
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args(argv)
    from bench.dashboard import current, gather
    data = current(gather([Path(r) for r in a.roots]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written = [p for p in (pilot(data, out), *phase1(data, out), *accuracy_vs_time(data, out)) if p]
    print(json.dumps([str(p) for p in written]))
    return 0


if __name__ == "__main__":
    main()
