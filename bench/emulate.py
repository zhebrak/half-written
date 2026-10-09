"""Timing emulation of the colocated synchronous baseline A2 from a real arm-A row (PLAN.md §4.6).

A and A2 run the same algorithm at lag 0, so they share one accuracy-against-step curve (S1); they differ only in how
long a step takes. This replays each step of an A row with the answers it actually consumed:
    gen     the step's answers (their recorded lengths, forced) through FakeEngine with the calibrated CostModel, in
            virtual time: one engine for A; for A2, half the groups on each of two engines, the step waiting for the
            slower
    train   the row's measured fwd_bwd time for that step; A2 halves it and adds an all-reduce of the gradients
    sync    the row's measured push time for A; for A2 a colocated (IPC) push, measured on the one-GPU layout
and returns per-step and cumulative times. Its credibility rests on the first column: emulated A must reproduce the
row's measured A step times (the paper reports the error; PLAN.md asks for +-5%). A2's own terms (the all-reduce, the
IPC push, KV capacity shared with a co-resident trainer) are assumptions the output records, not measurements; the
KV check fails when an engine's capacity cannot hold its half of the batch at full length.

A1 (one GPU, engine and trainer colocated) is not emulated: configs/one_gpu is exactly its layout, so it is measured.

    python -m bench.emulate A_ROW --calibration calibration.json [--allreduce-s 0.05] [--out emulated.json]
"""
import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from halfwritten.clock import run_sim
from halfwritten.engines.base import collect
from halfwritten.engines.fake import CostModel, FakeEngine
from halfwritten.lengths import LengthModel
from halfwritten.results import Row
from halfwritten.types import GenRequest


def gen_seconds(answers: list[tuple[int, int]], cost: CostModel, engines: int = 1, max_num_seqs: int = 256,
                max_num_batched_tokens: int = 8192) -> float:
    """Virtual seconds for `engines` engines to generate the answers [(prompt_len, length)], split by alternation."""
    async def one_engine(part):
        eng = FakeEngine(lengths=LengthModel(mean=1, sigma=0.0), cost=cost, max_num_seqs=max_num_seqs,
                         max_num_batched_tokens=max(max_num_batched_tokens, 4096))
        await eng.start()
        loop = asyncio.get_running_loop()
        t0 = loop.time()

        await asyncio.gather(*(collect(eng, GenRequest(f"e{i}#0", tuple(range(p)), n, ignore_eos=True))
                               for i, (p, n) in enumerate(part)))
        dt = loop.time() - t0
        await eng.close()
        return dt

    async def main():
        parts = [answers[i::engines] for i in range(engines)]
        return max(await asyncio.gather(*(one_engine(p) for p in parts if p)), default=0.0)
    return run_sim(main())


def steps_of(row: Path) -> list[dict]:
    """Per consumed step of a row: its answers [(prompt_len, tokens)], measured fwd_bwd and push seconds."""
    r_ = Row(row)
    by_step = defaultdict(list)
    for r in r_.ledger:
        if r["consumed_step"] is not None:
            by_step[r["consumed_step"]].append((r.get("prompt_len") or 0, r["tokens"]))
    fwd = r_.trainer_spans()
    syncs = r_.syncs
    steps = sorted(by_step)
    return [{"step": k, "answers": by_step[k], "train_s": fwd[i] if i < len(fwd) else None,
             "sync_s": syncs[i]["sync_s"] if i < len(syncs) else 0.0} for i, k in enumerate(steps)]


def emulate(row: Path, cost: CostModel, allreduce_s: float = 0.05, ipc_push_s: float | None = None,
            kv_tokens_per_engine: int | None = None) -> dict:
    cfg = Row(row).config
    kv_needed = cfg["M"] * cfg["G"] // 2 * cfg["max_model_len"]
    out = []
    for s in steps_of(row):
        if s["train_s"] is None:
            break
        g1, g2 = gen_seconds(s["answers"], cost, 1), gen_seconds(s["answers"], cost, 2)
        a2_push = s["sync_s"] if ipc_push_s is None else ipc_push_s
        out.append({"step": s["step"], "A": g1 + s["train_s"] + s["sync_s"],
                    "A2": g2 + s["train_s"] / 2 + allreduce_s + a2_push})
    for key in ("A", "A2"):
        t = 0.0
        for r in out:
            t += r[key]
            r[f"{key}_cum"] = t
    measured = Row(row).metrics
    return {"row": str(row), "steps": out, "measured_wall_s": measured["wall_s"], "measured_steps": measured["steps"],
            "emulated_A_wall_s": out[-1]["A_cum"] if out else 0.0,
            "a2_assumptions": {"allreduce_s": allreduce_s, "ipc_push_s": ipc_push_s,
                               "kv_tokens_per_engine": kv_tokens_per_engine, "kv_needed_per_engine": kv_needed,
                               "kv_ok": None if kv_tokens_per_engine is None else kv_tokens_per_engine >= kv_needed}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("row")
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--allreduce-s", type=float, default=0.05)
    ap.add_argument("--ipc-push-s", type=float, default=None, help="A2's colocated push (the probe measured 0.14 s)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cal = json.loads(Path(a.calibration).read_text())
    r = emulate(Path(a.row), CostModel(**cal["cost"]), a.allreduce_s, a.ipc_push_s, cal.get("kv_tokens"))
    Path(a.out or Path(a.row) / "emulated.json").write_text(json.dumps(r, indent=1))
    print(f"A measured {r['measured_wall_s']:.1f}s over {r['measured_steps']} steps; emulated A "
          f"{r['emulated_A_wall_s']:.1f}s; A2 {r['steps'][-1]['A2_cum']:.1f}s" if r["steps"] else "no steps")


if __name__ == "__main__":
    main()
