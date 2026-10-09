"""Where a run's files live, keyed by the hardware it ran on:

    results/<task>/<model>/<gpu>/<regime>[+lora]-B<b>G<g>M<m>-K<k>-S<s>[-<tag>][-lockstep]-s<seed>/
        config.yaml      the resolved RunConfig and the environment: versions, commit, GPUs, topology, host
        metrics.json     RunSummary.metrics(): validity and invariants, rates, idle fractions, pause and lag summaries
        steps.parquet    one row per optimiser step (lag and depth histograms, context fractions, trainer stats)
        syncs.parquet    one row per weight sync (pause and sync seconds, late tokens, re-prefill, resubmissions)
        ledger.parquet   one row per answer attempt with its version segments
        discards.parquet aborted (C2) and cancelled attempts
        trace.json       Chrome trace of every lane (open in Perfetto); trace.parquet the same events as a table

<gpu> is `sim` for simulated rows and `cpu` without a GPU; every command takes the tree's root as --out (default
results/). A row is written once: writing over a row that a different config produced raises, so a
matrix whose cells the label fails to tell apart cannot silently overwrite its own results.
"""
import json
import os
import platform
import re
import subprocess
import time
from functools import cached_property
from importlib import metadata
from pathlib import Path

import yaml

from .config import RunConfig, differences, to_dict
from .regimes import resolve
from .trace import to_chrome, to_table


def gpu_slug(name: str | None) -> str:
    """The <gpu> path part for an nvidia-smi name: 'NVIDIA H100 PCIe' -> 'H100-PCIe', None -> 'cpu'."""
    return "-".join(name.removeprefix("NVIDIA ").split()) if name else "cpu"


def model_name(model: str) -> str:
    return str(model).rstrip("/").rsplit("/", 1)[-1]


def row_label(cfg: RunConfig) -> str:
    label = f"{resolve(cfg.regime, cfg.lora).label}-B{cfg.B}G{cfg.G}M{cfg.M}-K{cfg.K}-S{cfg.S}"
    label += f"-{cfg.tag}" if cfg.tag else ""
    return label + ("-lockstep" if cfg.publication == "lockstep" else "") + f"-s{cfg.seed}"


def row_dir(cfg: RunConfig, gpu: str | None, root: Path) -> Path:
    return Path(root) / cfg.task / model_name(cfg.model) / gpu_slug(gpu) / row_label(cfg)


def shell(*cmd) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def versions() -> dict:
    """Package versions from installed metadata, without importing anything."""
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pyarrow", "torch", "transformers", "vllm", "peft", "math-verify"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def git_sha() -> str | None:
    """git describe of the code that ran, -dirty when a tracked file outside results/ differs from HEAD. A copy synced
    to a GPU box has no .git: there the sync writes REVISION (scripts/sync.sh) and that is the answer."""
    sha = shell("git", "describe", "--always")
    if sha and shell("git", "status", "--porcelain", "--untracked-files=no", "--", ":/", ":(top,exclude)results"):
        sha += "-dirty"
    if not sha:
        rev = Path(__file__).resolve().parents[1] / "REVISION"
        sha = rev.read_text().strip() if rev.exists() else None
    return sha


def host_gpu() -> str | None:
    """The first GPU's name as nvidia-smi reports it (the row directory's slug comes from it); None without one."""
    gpus = environment()["gpus"]
    return gpus[0] if gpus else None


def host_gpu_label() -> str | None:
    """The first GPU's readable label (gpu_label) for a tool's report; None without one."""
    gpu = host_gpu()
    return gpu_label(gpu) if gpu else None


def environment() -> dict:
    gpus = shell("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    return {"versions": versions(), "commit": git_sha(), "host": platform.node(), "cpu": platform.processor(),
            "gpus": gpus.splitlines() if gpus else [], "topology": shell("nvidia-smi", "topo", "-m"),
            "gpu_details": shell("nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,"
                                 "power.limit,clocks.max.sm,clocks.max.memory", "--format=csv"),
            "vllm_env": {k: v for k, v in os.environ.items() if k.startswith("VLLM_")}}


def _table(rows: list[dict]):
    """pyarrow infers a table's columns from its first row; steps gain trainer stats only once trained."""
    import pyarrow as pa
    keys = list(dict.fromkeys(k for r in rows for k in r))
    return pa.Table.from_pylist([{k: r.get(k) for k in keys} for r in rows])


class RowExists(FileExistsError):
    pass


def file_sha256(path: Path) -> str:
    """sha256 of a file, cached beside it in .<name>.sha256 against its size and mtime: W0's 1.2 GB is hashed once,
    not at every row or queue check."""
    import hashlib
    path = Path(path)
    st = path.stat()
    cache = path.with_name(f".{path.name}.sha256")
    stamp = f"{st.st_size} {st.st_mtime_ns}"
    if cache.exists():
        recorded = cache.read_text().split()
        if " ".join(recorded[:2]) == stamp:
            return recorded[2]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    try:
        cache.write_text(f"{stamp} {h.hexdigest()}\n")
    except OSError:  # a read-only artifact: hash it each time
        pass
    return h.hexdigest()


def row_identity(cfg: RunConfig) -> dict:
    """What a row's config names by path, by content: the warm start's weights and the prompt filter (external review
    F01: a changed W0 or filter behind the same path must not pass for the same experiment)."""
    ident = {}
    if cfg.task in ("gsm8k", "math"):  # the development split held out of its training pool, and the exact files
        from halfwritten.revisions import model_revision, task_revisions
        from halfwritten.tasks import dev_digest
        ident["dev_sha256"] = dev_digest(cfg.task)
        ident["revisions"] = {"model": model_revision(cfg.model), **task_revisions(cfg.task)}
    if cfg.task == "math":  # the training reward's verifier: rows trained under another version are another experiment
        ident["math_verify"] = versions()["math-verify"]
    if cfg.init_from:
        ident["init_from_sha256"] = file_sha256(Path(cfg.init_from) / "model.safetensors")
    if cfg.prompt_filter:
        ident["prompt_filter_sha256"] = file_sha256(Path(cfg.prompt_filter))
    return ident


SET_ASIDE = re.compile(r"\.(invalid|harness|legacy|signalled)-\d+$")


def set_aside(path: Path, why: str) -> Path:
    """Moves a row out of its label's way (an invalid, harness-excluded, legacy or signalled one), kept beside it for
    the record; the suffix is unique even for two set-asides of one row within a second."""
    new = path.with_name(f"{path.name}.{why}-{time.time_ns()}")
    path.rename(new)
    return new


def rerun_pending(path: Path) -> bool:
    """Whether the row at `path` is already the rerun of one set aside as invalid: HYPOTHESES.md decision 7 gives
    such a row one more run, so a second failure stands (the queue and finished() share this rule). A row set
    aside for its harness by the code before 2026-10-02 counts too: its rerun stands whatever it reads."""
    return any(path.parent.glob(f"{path.name}.invalid-*")) or any(path.parent.glob(f"{path.name}.harness-*"))


def is_set_aside(path: Path) -> bool:
    """A row moved aside (set_aside): readers of results skip it, as the queue and the analysis must."""
    return bool(SET_ASIDE.search(Path(path).name))


def _atomic(target: Path, write) -> None:
    """write(tmp) then rename onto target: a reader (or a crash) sees the old file or the whole new one."""
    tmp = target.with_name(f".{target.name}.tmp")
    write(tmp)
    tmp.replace(target)


def write_row(path: Path, cfg: RunConfig, summary, ledger, events, env: dict | None = None,
              overwrite: bool = False, snapshots: dict | None = None, identity: dict | None = None) -> Path:
    """Writes a row's files, each atomically, and metrics.json last: metrics.json is the row's completion marker
    (run_row and the queue treat a row with it as finished), so it must not exist before everything else does."""
    import pyarrow.parquet as pq
    if (path / "config.yaml").exists() and not overwrite:
        if diff := differences(Row(path).config, to_dict(cfg)):
            raise RowExists(f"{path} holds a row with other {', '.join(diff)}; give the runs distinct labels (tag)")
    path.mkdir(parents=True, exist_ok=True)
    text = lambda s: lambda tmp: tmp.write_text(s)  # noqa: E731
    _atomic(path / "config.yaml", text(yaml.safe_dump({"config": to_dict(cfg), "environment": env or environment(),
                                                       "identity": identity if identity is not None
                                                       else row_identity(cfg)}, sort_keys=False)))
    for name, rows in (("steps", summary.steps_log), ("syncs", summary.syncs)):
        if rows:
            _atomic(path / f"{name}.parquet", lambda tmp, rows=rows: pq.write_table(_table(rows), tmp))
    _atomic(path / "ledger.parquet", lambda tmp: pq.write_table(ledger.table(), tmp))
    if ledger.discards():
        _atomic(path / "discards.parquet", lambda tmp: pq.write_table(ledger.discards_table(), tmp))
    _atomic(path / "trace.json", text(json.dumps(to_chrome(events))))
    _atomic(path / "trace.parquet", lambda tmp: pq.write_table(to_table(events), tmp))
    if snapshots is not None:
        _atomic(path / "snapshots.json", text(json.dumps(snapshots, indent=1)))
    _atomic(path / "metrics.json", text(json.dumps(summary.metrics(), indent=1) + "\n"))
    return path


def read_jsonl(path: Path) -> list[dict]:
    """A JSON-lines file's records (blank lines skipped); [] if there is no such file."""
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def read_parquet(path: Path) -> list[dict]:
    """A row's parquet file as records; [] if there is no such file."""
    if not path.exists():
        return []
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


# The controller's main thread: x2f's saturated loop (0.89-1.0 of a core) stretched its fence to 469 ms. Half a core
# was PLAN.md's Day-1 gate, but rows from W0 sit at 0.48-0.55 with a loop lag p99 of 36-51 ms, no harm, and a cap
# there would exclude D and E rows by noise and not A (review of 2026-10-02): the cap is where harm begins, and the
# loop lag says whether it did.
MAX_HOST_CPU = 0.75
MAX_LOOP_LAG_P99 = 0.1  # seconds late from a 50 ms sleep, the 99th percentile (2026-10-02's rows: 36-51 ms)
MAX_FENCE_P50 = 0.1  # seconds paused after the ack per sync, at the median (x2d: 41 ms; x2f's saturated loop: 469 ms)
# decision 7's limits as the analysis applies them: these defaults, or the amendment's `harness_limits` (bench.amend
# writes them beside each pilot row's record for the owner to fix at the tag; a revision is a recorded amendment,
# never a code edit)
HARNESS_LIMITS = {"max_host_cpu": MAX_HOST_CPU, "max_loop_lag_p99_s": MAX_LOOP_LAG_P99,
                  "max_fence_p50_s": MAX_FENCE_P50}


def harness_record(metrics: dict) -> dict:
    """What decision 7 reads from a row, as the amendment records it beside the limits."""
    host = metrics.get("host") or {}
    return {"main_thread_cpu_frac": host.get("main_thread_cpu_frac"),
            "loop_lag_p99_s": (host.get("loop_lag_s") or {}).get("p99"),
            "fence_p50_s": (metrics.get("fence_s") or {}).get("p50"),
            "reward_timeouts": (metrics.get("events") or {}).get("reward_timeouts", 0)}


def harness_problem(metrics: dict, limits: dict | None = None) -> str | None:
    """Why a valid row's timing measures our harness rather than its regime (review 8), or None, under `limits`
    (HARNESS_LIMITS, or the amendment's). Rows from before the controller recorded itself (no `host`) cannot show
    they were not; a row whose scorer timed out labelled answers 0 that it never scored."""
    lim = HARNESS_LIMITS | (limits or {})
    host = metrics.get("host")
    if not host:
        return "no harness record (host): from before the fast client"
    if host.get("main_thread_cpu_frac", 0.0) > lim["max_host_cpu"]:
        return f"controller at {host['main_thread_cpu_frac']:.0%} of a core (limit {lim['max_host_cpu']:.0%})"
    if (lag := (host.get("loop_lag_s") or {}).get("p99")) is not None and lag > lim["max_loop_lag_p99_s"]:
        return f"event loop late by {lag * 1000:.0f} ms at p99 (limit {lim['max_loop_lag_p99_s'] * 1000:.0f} ms)"
    fence = (metrics.get("fence_s") or {}).get("p50")
    if fence is not None and fence > lim["max_fence_p50_s"]:
        return f"fence p50 {fence * 1000:.0f} ms (limit {lim['max_fence_p50_s'] * 1000:.0f} ms)"
    if n := (metrics.get("events") or {}).get("reward_timeouts"):
        return f"{n} reward timeout(s): answers scored 0 unscored"
    return None


def row_problem(metrics: dict) -> str | None:
    """Why a finished row does not stand for the queue, or None: not valid (an invariant failed or the run erred).
    The queue and launch.finished rerun such a row once (HYPOTHESES.md decision 7). A valid row whose harness
    record trips the limits (harness_problem) stands here and is flagged in the queue's report: the rule is
    applied in the analysis (Row.problem), not at run time, so the box runs on and the owner decides on the limit
    with every row in hand (the owner's policy, 2026-10-02)."""
    if not metrics.get("valid"):
        return "invalid: " + (metrics.get("error") or "; ".join(metrics.get("violations") or []) or "not valid")
    return None


class Row:
    """A result row's files, each read once and only when asked: the one reader the analysis, the queue, the
    launcher and the page share, where each used to open config.yaml, metrics.json and the parquet files itself."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"Row({self.path})"

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def finished(self) -> bool:
        """metrics.json is written last (write_row): with it the row is complete."""
        return (self.path / "metrics.json").exists()

    @cached_property
    def recorded(self) -> dict:
        """config.yaml as written: the resolved config, the environment and the identity."""
        f = self.path / "config.yaml"
        return (yaml.safe_load(f.read_text()) or {}) if f.exists() else {}

    @property
    def config(self) -> dict:
        """The resolved RunConfig the row ran with, as recorded (a raw dict: archived rows may hold keys the
        current RunConfig has dropped)."""
        return self.recorded.get("config") or {}

    @property
    def identity(self) -> dict:
        return self.recorded.get("identity") or {}

    @property
    def environment(self) -> dict:
        return self.recorded.get("environment") or {}

    @cached_property
    def metrics(self) -> dict:
        f = self.path / "metrics.json"
        return json.loads(f.read_text()) if f.exists() else {}

    @property
    def valid(self) -> bool:
        return bool(self.metrics.get("valid"))

    def problem(self, harness: bool = True, limits: dict | None = None) -> str | None:
        """Why the analysis leaves the row out, or None: not valid, or, with `harness`, excluded for its harness
        (harness_problem under `limits`, a missing record included). The queue's rerun rule is row_problem:
        validity alone."""
        if not self.valid:
            return row_problem(self.metrics)
        if harness and (why := harness_problem(self.metrics, limits)):
            return "harness: " + why
        return None

    @cached_property
    def steps(self) -> list[dict]:
        return read_parquet(self.path / "steps.parquet")

    @cached_property
    def syncs(self) -> list[dict]:
        return read_parquet(self.path / "syncs.parquet")

    @cached_property
    def ledger(self) -> list[dict]:
        return read_parquet(self.path / "ledger.parquet")

    @cached_property
    def trace(self) -> list[dict]:
        return read_parquet(self.path / "trace.parquet")

    @cached_property
    def snapshots(self) -> dict:
        f = self.path / "snapshots.json"
        return json.loads(f.read_text()) if f.exists() else {}

    @cached_property
    def emulated(self) -> dict | None:
        f = self.path / "emulated.json"
        return json.loads(f.read_text()) if f.exists() else None

    def evals(self, split: str | None = None) -> list[dict]:
        """The evaluation records (bench.evaluate), on one split if given (a record without a split is test's)."""
        records = read_jsonl(self.path / "evals.jsonl")
        return records if split is None else [e for e in records if e.get("split", "test") == split]

    @property
    def gpu(self) -> str:
        """The GPU's readable label, from the environment the row recorded; a directory name is the fallback (an
        archive may file rows by layout, as data/two_gpu-2026-10-02/gsm8k_one_gpu does)."""
        gpus = self.environment.get("gpus")
        return gpu_label(gpus[0]) if gpus else gpu_label(self.path.parent.name)

    def states(self) -> list[dict]:
        """The trace's state spans (proc, lane, name, t0, t1), the timeline's material."""
        return [e for e in self.trace if e["kind"] == "state"]

    def trainer_spans(self) -> list[float]:
        """The trainer's steps (fwd_bwd spans, seconds) in order; the last may be cut short by the end of the run."""
        return [e["t1"] - e["t0"] for e in sorted(self.states(), key=lambda e: e["t0"])
                if e["lane"] == "trainer" and e["name"] == "fwd_bwd"]

    def stale_kv_share(self, after_s: float = 60.0) -> float | None:
        """D's exposure (HYPOTHESES.md decision 4): the share of the tokens consumed after the first minute that
        were generated over stale KV, token-weighted (all such tokens over all consumed tokens; a per-step median
        reads far lower, since most steps have none). From the step records, before any accuracy."""
        xs = [(s["frac_stale_kv"], s["tokens"]) for s in self.steps
              if s.get("t") is not None and s["t"] > after_s and s.get("tokens") and s.get("frac_stale_kv") is not None]
        return float(sum(f * n for f, n in xs) / sum(n for _, n in xs)) if xs else None


def rows(root: Path) -> list[Row]:
    """Every finished row under `root`, in path order, skipping session directories (`_*`), task prep and rows set
    aside."""
    out = []
    for m in sorted(Path(root).rglob("metrics.json")):
        row = m.parent
        if any(part.startswith("_") for part in row.relative_to(root).parts) or "prep" in row.parts:
            continue
        if is_set_aside(row):
            continue
        out.append(Row(row))
    return out


GPU_LABELS = [("H100-80GB-HBM3", "H100 SXM"), ("H100 80GB HBM3", "H100 SXM"), ("h100x2", "H100 SXM"),
              ("H100-NVL", "H100 NVL"), ("H100 NVL", "H100 NVL"), ("h100", "H100 NVL"),
              ("A100-SXM4-40GB", "A100 40GB"), ("a100", "A100 40GB")]  # slugs, nvidia-smi names, session names


def gpu_label(text: str) -> str:
    """A readable GPU name from a GPU slug or a session directory name (results/_sessions/<name>)."""
    return next((label for key, label in GPU_LABELS if key in text), text)
