"""Offline evaluation: held-out accuracy of the snapshots a run saved at wall-clock marks, on a GPU that is not
timing anything. The area under these marks over [0, T] is the primary endpoint (HYPOTHESES.md decision 1); the
snapshot tagged "tend", taken at exactly the budget T in every arm, is the first secondary.
 One `vllm serve` of
the base model per session; each snapshot is pushed into it through the same weight sink the trainer uses (IPC when
the evaluating process shares the GPU), then the held-out prompts are decoded greedily through the same HTTP adapter
the controller uses and scored by the task's reward. No server restarts between snapshots.

    python -m bench.evaluate ROW_DIR [--split dev|test] [--limit N]    writes ROW_DIR/evals.jsonl

The split is the row's (config eval_split: dev unless a confirmatory grid set test). A record is reused only if its
fingerprint matches: the split, the items (ordered prompt token ids and references), the decoding, the scorer and the
snapshot file (external review F03: a same-size swap of items, or another max_tokens, kept a stale accuracy).
"""
import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from halfwritten.engines.base import collect
from halfwritten.results import Row, read_jsonl
from halfwritten.revisions import model_revision
from halfwritten.tasks import score
from halfwritten.types import BUDGET_MARK, GenRequest, parse_snapshot, snapshot_name


async def accuracy(engine, items, reward, *, max_tokens: int, stop=(), concurrency: int = 256,
                   tag: str = "eval", adapter: str | None = None, reward_pool=None) -> dict:
    """Greedy accuracy of `items` [(prompt ids, answer)] through `engine` (an Engine), under `adapter` if given."""
    sem = asyncio.Semaphore(concurrency)

    async def one(i, prompt, answer):
        async with sem:
            got = await collect(engine, GenRequest(f"{tag}-{i}#0", tuple(prompt), max_tokens, temperature=0.0,
                                                   adapter=adapter, stop=tuple(stop)))
            return await score(reward, reward_pool, answer, got.tokens, got.finish), len(got.tokens)

    t0 = time.monotonic()
    results = await asyncio.gather(*(one(i, p, a) for i, (p, a) in enumerate(items)))
    scores = [s for s, _ in results]
    # per-item scores, in item order: every arm answers the same items, so the analysis can pair them (bench/stats.py)
    return {"accuracy": sum(scores) / len(scores), "n": len(scores),
            "mean_tokens": sum(n for _, n in results) / len(results), "seconds": time.monotonic() - t0,
            "correct": "".join("1" if x >= 1.0 else "0" for x in scores)}


SCORER_VERSION = "2026-10-02"  # bump when a task's scoring changes (halfwritten/rewards.py)


def items_digest(items) -> str:
    """sha256 of the ordered evaluation items (prompt token ids, reference): the pairing key between arms."""
    h = hashlib.sha256()
    for prompt, answer in items:
        h.update(json.dumps([list(prompt), str(answer)]).encode())
    return h.hexdigest()


def scorer_versions() -> dict:
    """The scorer an evaluation used, as plain fields of its record: the analysis requires one value across rows."""
    from halfwritten.results import versions
    return {"scorer": SCORER_VERSION, "math_verify": versions()["math-verify"]}


def fingerprint(split: str, items: str, max_tokens: int, stop, snapshot: Path | None) -> str:
    """What an evaluation record measured; a record is reused only when it matches."""
    file = snapshot / "model.safetensors" if snapshot is not None else None
    stat = file.stat() if file is not None and file.exists() else None
    signature = [stat.st_size, stat.st_mtime_ns] if stat else None
    if signature is None and snapshot is not None and (snapshot / "pruned.json").exists():
        signature = json.loads((snapshot / "pruned.json").read_text()).get("stat")
    return hashlib.sha256(json.dumps({
        "split": split, "items": items, "max_tokens": max_tokens, "stop": list(stop), "temperature": 0.0,
        **scorer_versions(),
        "snapshot": signature}, sort_keys=True).encode()).hexdigest()


def snapshot_targets(row: Path) -> dict[str, Path]:
    """All expected snapshots, including missing files and legacy step snapshots recorded only as written."""
    snaps, work = Row(row).snapshots, row / "work"
    if "written" in snaps:
        names = [Path(p).name for _, p, *_ in snaps["written"] if p]
    else:
        names = sorted(p.name for p in work.iterdir() if (p / "model.safetensors").exists()
                       or (p / "pruned.json").exists()) if work.exists() else []
    targets = dict.fromkeys(names)
    for mark in snaps.get("marks", []):
        if "staged" in mark:
            targets[snapshot_name(mark["tag"], mark["staged"])] = None
        elif not any(parse_snapshot(n)[0] == mark["tag"] for n in targets):
            targets[mark["tag"]] = None  # legacy mark with no written snapshot: never silently omit it
    return {name: work / name for name in targets}


def default_init(cfg: dict) -> Path | None:
    """The weights evaluated as "init", the run's start: the warm start's directory, unless the row is a LoRA one,
    whose evaluation engine serves W0 itself (then "init" is the engine's base, pushed nothing). The one rule the
    evaluator, the queue's completeness check and the pruner share."""
    return Path(cfg["init_from"]) if cfg.get("init_from") and not cfg.get("lora") else None


def evaluation_problem(row: Path, split: str, *, records=None, fingerprints: dict | None = None,
                       current: bool = False) -> str | None:
    """Shared completeness check for resume, pruning and analysis; strict identity checks when inputs are known."""
    records = Row(row).evals(split) if records is None else [e for e in records if e.get("split", "test") == split]
    by_name = {e.get("snapshot"): e for e in records}
    cfg = Row(row).config
    targets = {"init": default_init(cfg), **snapshot_targets(row)}
    for name, path in targets.items():
        if name not in by_name:
            return f"snapshot {name} has no evaluation on {split}"
        if by_name[name].get("tag") != parse_snapshot(name)[0]:
            return f"snapshot {name} has an inconsistent evaluation tag"
        if fingerprints is not None and by_name[name].get("fingerprint") != fingerprints[name]:
            return f"snapshot {name} has a stale evaluation on {split}"
        if current and "max_tokens" in cfg and "items_sha256" in by_name[name]:
            from halfwritten.tasks import STOPS
            fp = fingerprint(split, by_name[name]["items_sha256"], cfg["max_tokens"],
                             STOPS.get(cfg.get("task"), ()), path)
            if by_name[name].get("fingerprint") != fp:
                return f"snapshot {name} has a stale evaluation on {split}"
    return None


def evaluation_plan(row: Path, split: str, items, max_tokens: int, stop, init: Path | None):
    """Pending targets and their fingerprints. Refuse missing weights before starting an evaluation server."""
    targets = {"init": init, **snapshot_targets(row)}
    digest = items_digest(items)
    fingerprints = {name: fingerprint(split, digest, max_tokens, stop, path) for name, path in targets.items()}
    done = {e.get("snapshot"): e for e in Row(row).evals(split)}
    pending = [(name, path) for name, path in targets.items()
               if done.get(name, {}).get("fingerprint") != fingerprints[name]]
    for name, path in pending:
        if name != "init" and not (path / "model.safetensors").exists():
            raise ValueError(f"{row}: snapshot {name} needs evaluation but its weights are missing or pruned")
    return pending, fingerprints


def snapshot_dirs(row: Path) -> list[Path]:
    """The row's snapshots in the order they were taken: those snapshots.json lists as written by this run (by name,
    under work/: a row copied off the pod keeps them), else every complete one under work/."""
    return [d for d in snapshot_targets(row).values() if (d / "model.safetensors").exists()]


async def evaluate_row(row: Path, engine, push, items, reward, *, max_tokens: int, stop=(), init: Path | None = None,
                       split: str = "test", log=print, reward_pool=None) -> list[dict]:
    """Evaluates the run's starting weights ("init": the warm start pushed first when the run had one, else the
    engine's own base weights) and then every snapshot, pushing each with `push(version, path)`. Each record carries
    its mark from snapshots.json (t, steps, staged and engine versions at the mark), so accuracy is plotted against
    the wall clock of the run, not of the evaluation. evals.jsonl is keyed by snapshot and rewritten whole: a rerun
    skips what is done and never duplicates a line."""
    marks = {m["tag"]: m for m in Row(row).snapshots.get("marks", [])}
    out_path = row / "evals.jsonl"
    done = {(r["snapshot"], r.get("split", "test")): r for r in read_jsonl(out_path)}  # records before splits: test
    digest = items_digest(items)
    targets, fingerprints = evaluation_plan(row, split, items, max_tokens, stop, init)
    version = 0
    for name, path in targets:
        fp = fingerprints[name]
        adapter = None
        if path is not None:
            version += 1
            if asyncio.iscoroutinefunction(push):  # an adapter loaded under a name (adapter_push)
                adapter = await push(version, path)
            else:
                await asyncio.to_thread(push, version, path)
        tag, _ = parse_snapshot(name)
        mark = marks.get(tag, {"t": 0.0, "steps": 0} if tag == "init" else {})
        r = await accuracy(engine, items, reward, max_tokens=max_tokens, stop=stop, tag=name, adapter=adapter,
                           reward_pool=reward_pool)
        done[(name, split)] = {"snapshot": name, "tag": tag, "split": split, "items_sha256": digest,
                               "fingerprint": fp, **scorer_versions(),
                               **{k: mark.get(k) for k in ("t", "steps", "staged", "engine")}, **r}
        records = sorted(done.values(), key=lambda x: (x["t"] is None, x["t"] or 0.0))
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(x) + "\n" for x in records))
        tmp.replace(out_path)
        log(f"{name}: accuracy {r['accuracy']:.3f} over {r['n']} ({r['seconds']:.0f} s)")
    if problem := evaluation_problem(row, split, fingerprints=fingerprints):
        raise ValueError(f"{row}: {problem}")
    return sorted(done.values(), key=lambda x: (x["t"] is None, x["t"] or 0.0))


def prune_snapshots(row: Path, split: str, keep: tuple[str, ...] = (BUDGET_MARK,)) -> list[dict]:
    """Deletes the weights of the row's snapshots once every scheduled mark has an evaluation record on `split`
    (a snapshot holds about 1.2 GB; a grid's rows would fill the box), keeping the marks in `keep` (the budget's: the
    primary endpoint's weights) and leaving each pruned directory with pruned.json (name, bytes, sha256) in place of
    its file. Nothing is pruned while a mark is unevaluated. Returns what was pruned."""
    from halfwritten.results import file_sha256
    if not snapshot_targets(row) or evaluation_problem(row, split, current=True):
        return []
    pruned = []
    for d in snapshot_dirs(row):
        tag, _ = parse_snapshot(d.name)
        if tag in keep:
            continue
        f = d / "model.safetensors"
        stat = f.stat()
        record = {"name": d.name, "bytes": stat.st_size, "sha256": file_sha256(f), "split": split,
                  "stat": [stat.st_size, stat.st_mtime_ns]}
        (d / "pruned.json").write_text(json.dumps(record, indent=1))
        f.unlink()
        pruned.append(record)
    return pruned


def vllm_push(url: str, gpu: str = "0"):
    """push(version, snapshot dir) over IPC from this process's GPU (the evaluation server shares it)."""
    import torch
    from safetensors.torch import load_file

    from halfwritten.gpu import local_device
    from halfwritten.trainer.sync import VllmSink
    device = local_device(gpu)
    sink = VllmSink(url, backend="ipc", packed=True)

    def push(version: int, path: Path):
        with torch.cuda.device(device):
            tensors = load_file(str(path / "model.safetensors"), device=device)
            sink.push(version, {k: v.to(torch.bfloat16) for k, v in tensors.items()})
    return push


def adapter_push(engine):
    """push(version, snapshot dir) for a LoRA row: loads the snapshot as an adapter under a new name (never in place:
    vLLM 0.30's in-place flag sticks, NOTEBOOK.md 2026-10-02) on an engine whose base is the row's own (W0), unloads
    the previous one, and returns the name to evaluate under. No bf16 merge, which would round small deltas away."""
    loaded = []

    async def push(version: int, path: Path) -> str:
        name = f"snapshot-{version}"
        await engine.load_adapter(name, str(path), inplace=False)
        if loaded:
            await engine.unload_adapter(loaded.pop())
        loaded.append(name)
        return name
    return push


def eval_config(cfg, sets: list[str] = ()):
    """The row's engine settings (model, context, LM head, scheduling) for an evaluation server: a plain synchronous
    launch, full weights pushed over IPC from this process on the engine's GPU (launch.tool_config), vLLM's memory
    share leaving room for the snapshot being pushed; `sets` override (launch.port=..., as every tool takes)."""
    from halfwritten.launch import tool_config
    return tool_config(cfg, "launch.gpu_memory_utilization=0.35", *sets, ipc=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("row")
    ap.add_argument("--split", choices=("dev", "test"), default=None, help="default: the row's eval_split")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], help="launch overrides, e.g. launch.port=8101")
    a = ap.parse_args(argv)

    from halfwritten.config import from_dict
    from halfwritten.engines.vllm_http import run_uvloop
    from halfwritten.launch import tool_engine
    from halfwritten.tasks import STOPS, TextReward, encode, eval_examples, scoring, too_long
    row = Path(a.row)
    cfg = eval_config(from_dict(Row(row).config), a.set)
    from halfwritten.results import row_identity
    identity = row_identity(cfg)
    recorded = Row(row).identity
    if "init_from_sha256" in recorded and recorded["init_from_sha256"] != identity.get("init_from_sha256"):
        raise SystemExit(f"{row}: the warm start changed since training; restore the recorded weights")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model, revision=model_revision(cfg.model))
    split = a.split or cfg.eval_split
    if split == "dev":  # dev is held out of training only from the rows that recorded it: earlier ones trained on it
        if recorded.get("dev_sha256") != identity.get("dev_sha256"):
            raise SystemExit(f"{row} did not hold this development split out of training: evaluate it on test")
    items = encode(tok, cfg.task, eval_examples(cfg.task, split)[:a.limit])
    if long := too_long(items, cfg.max_tokens, cfg.max_model_len):  # every GSM8K and MATH-500 item fits (2026-10-02)
        raise SystemExit(f"{len(long)} evaluation items leave no room for {cfg.max_tokens} tokens: {long[:5]}")
    adapters = bool(Row(row).config.get("lora"))
    init = default_init(Row(row).config)
    pending, _ = evaluation_plan(row, split, items, cfg.max_tokens, STOPS[cfg.task], init)
    if not pending:
        print(f"{row}: every snapshot already evaluated on these inputs")
        return
    if adapters:  # adapter snapshots, on an engine serving the row's base (W0) with LoRA on
        from halfwritten.launch import tool_config
        cfg = tool_config(cfg, "regime=F", "K=1", "lora=true", "launch.api_servers=1")
    with scoring(TextReward(cfg.model, cfg.task)) as (reward, pool):
        with tool_engine(cfg, row / "logs" / "eval") as (url, engine):
            push = adapter_push(engine) if adapters else vllm_push(url, cfg.launch.engine_gpu)
            run_uvloop(evaluate_row(row, engine, push, items, reward, max_tokens=cfg.max_tokens,
                                   stop=STOPS[cfg.task], init=init, split=split, reward_pool=pool))


if __name__ == "__main__":
    main()
