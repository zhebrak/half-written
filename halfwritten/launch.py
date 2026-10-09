"""One row, end to end: `python -m halfwritten run CONFIG [--set key=value ...] [--out results]`.

    engine    `vllm serve` (serve_command_for: the study's flags, M*G concurrency, LoRA slots) on launch.engine_gpu,
              or the fake server (engines/fake_server.py) for CPU tests; waited on until /health answers
    trainer   the worker subprocess (trainer/worker.py) on launch.trainer_gpu. Its sink: NCCL to an engine on
              another GPU, IPC to one on the same GPU, an adapter directory for LoRA, a version label for the fake
    task      prompts, reward and stop strings (tasks.py)
    run       Controller.run, then the row: results.write_row with both processes' traces merged

A finished, valid row of the same measurement is skipped (finished), so a queue can be rerun after a crash.
Subprocess output goes to
<row>/logs/{engine,trainer}.log; snapshots to <row>/work/snapshots.
"""
import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .advantages import loss_scale
from .config import RunConfig, differences, to_dict, with_overrides
from .controller import Controller
from .engines.vllm_http import VllmHttpEngine, healthy, run_uvloop, serve_command_for
from .regimes import resolve
from .results import Row, environment, host_gpu, rerun_pending, row_dir, row_identity, row_problem, set_aside, write_row
from .trace import lane_fractions, merge
from .trainer.client import ProcTrainerClient

REPO_ROOT = Path(__file__).resolve().parents[1]


def lora_base_dir(cfg: RunConfig, where: Path) -> Path:
    """The engine's base for a LoRA run from a warm start: an HF model directory at `where` holding W0's weights and the
    base model's config and tokenizer (links into the pinned snapshot). A LoRA run pushes only adapters, so the engine
    must start from W0, as the trainer does, or trainer and engine are two policies (external review F05)."""
    from huggingface_hub import snapshot_download

    from .revisions import model_revision
    base = Path(snapshot_download(cfg.model, revision=model_revision(cfg.model), local_files_only=True))
    where.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            (where / f.name).unlink(missing_ok=True)
            (where / f.name).symlink_to(f.resolve())
    (where / "model.safetensors").unlink(missing_ok=True)
    (where / "model.safetensors").symlink_to((Path(cfg.init_from) / "model.safetensors").resolve())
    return where


def engine_command(cfg: RunConfig, base_dir: Path | None = None) -> tuple[list[str], dict, str]:
    """(argv, environment additions, served model name) for the row's engine server. A LoRA run from a warm start
    serves W0 (lora_base_dir, under base_dir) under the model's name."""
    launch = cfg.launch
    env = {"PYTHONPATH": str(REPO_ROOT)}
    if launch.engine == "fake":
        argv = [sys.executable, "-m", "halfwritten.engines.fake_server", "--port", str(launch.port),
                "--lengths", launch.fake_lengths, "--vocab", str(launch.fake_vocab),
                "--max-num-seqs", str(cfg.M * cfg.G), "--max-model-len", str(cfg.max_model_len)]
        return argv, env, "fake"
    if resolve(cfg.regime, cfg.lora).weights == "lora_inplace" and not launch.lora_inplace_fix:
        raise ValueError("in-place LoRA reloads are wrong on stock vLLM 0.30: every later request carries the "
                         "reload's load_inplace flag, so each engine step reloads an adapter from disk, the path of "
                         "the batch's first request, an older version while a request from before the swap is in "
                         "flight (NOTEBOOK.md 2026-10-02, bench/lora_probe.py); "
                         "launch.lora_inplace_fix clears the flag as SkyRL and prime-rl do")
    model, extra = None, []
    if cfg.lora and cfg.init_from:
        if base_dir is None:
            raise ValueError("a LoRA run from a warm start needs a directory for the engine's base (base_dir)")
        model, extra = str(lora_base_dir(cfg, base_dir)), ["--served-model-name", cfg.model]
    argv, extra_env = serve_command_for(cfg, model=model, port=launch.port, async_scheduling=launch.async_scheduling,
                                        gpu_memory_utilization=launch.gpu_memory_utilization,
                                        weight_backend=launch.weight_backend, head_fp32=launch.head_fp32,
                                        stream_interval=launch.stream_interval, api_servers=launch.api_servers,
                                        extra=extra)
    exe = [sys.executable, "-m", "halfwritten.engines.vllm_serve"] if launch.lora_inplace_fix \
        else [str(Path(sys.executable).parent / "vllm")]
    env = env | extra_env | {"CUDA_VISIBLE_DEVICES": launch.engine_gpu} | nccl_env(cfg)
    return [*exe, *argv[1:]], env, cfg.model


def nccl_env(cfg: RunConfig) -> dict:
    """The environment both ends of an NCCL weight push need: NVLS off (its allocation failed in the
    containers the study ran in, CUDA error 401 in nvlsAllocateMem; NOTEBOOK.md, two H100 SXM). Set here, on the one
    path every two-GPU row takes, and not in a launcher script some runs never go through."""
    return {"NCCL_NVLS_ENABLE": "0"} if cfg.launch.weight_backend == "nccl" else {}


def _exit_on_sigterm(signum=signal.SIGTERM, frame=None):
    raise SystemExit(128 + signum)


ADAPTER_ROOT = "/dev/shm/halfwritten"  # where a LoRA run writes its adapters for the engine (a tmpfs)
MASTER_PORT = 29555  # the NCCL rendezvous between the trainer and the engine
GRACE_S = 15.0  # a signalled run ends within this, or exits at once; a supervisor kills 30 s after its TERM, and the row
SIGNALLED_ENGINE_WAIT_S = 8.0  # must also be written and the engine stopped (normally it gets 60 s) in that time
_signalled = False


@contextlib.contextmanager
def sigterm_stops(controller: Controller, grace_s: float = GRACE_S):
    """While a run is live (in the event loop's main thread), the first SIGTERM ends it as a budget would, so its row
    is still written (marked invalid); a second one, or grace_s later, exits at once. Before, SIGTERM raised
    SystemExit into the loop, whose shutdown waited for the trainer's goodbye: the S1 run outlived it by minutes."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    loop = asyncio.get_running_loop()
    previous = signal.getsignal(signal.SIGTERM)
    deadline = []

    def on_sigterm():
        global _signalled
        _signalled = True
        if not controller.request_stop():
            _exit_on_sigterm()
        deadline.append(loop.call_later(grace_s, _exit_on_sigterm))

    loop.add_signal_handler(signal.SIGTERM, on_sigterm)
    try:
        yield
    finally:
        for h in deadline:
            h.cancel()
        loop.remove_signal_handler(signal.SIGTERM)
        signal.signal(signal.SIGTERM, previous)


@contextlib.asynccontextmanager
async def host_watch(period: float = 0.05, start: Callable[[], float | None] | None = None,
                     live: Callable[[], bool] = lambda: True):
    """The controller process's own load while a run is live, filled into the dict it yields: how late the event loop
    wakes from a `period` sleep (a loop busy reading streams delays every timer, the fence's included), when the worst
    stall came, and the main thread's share of a core (one process reads every streamed token: on two H100s at ~15k
    chunks/s it was the bottleneck). Live: from `start()` (the run's clock origin, on the loop's clock: None until the
    first dispatch, so engine startup and the first push are not counted) while `live()` (the end-of-run summary,
    computed on the loop, is not). The worst stall's time is on the row's own clock (steps.parquet's t)."""
    from .metrics import quantiles
    loop = asyncio.get_running_loop()
    lags: list[float] = []
    worst = {"lag_s": 0.0, "at_s": None}
    t_enter = loop.time()
    start = start or (lambda: t_enter)
    t0 = c0 = None
    last: dict = {}

    async def watch():
        nonlocal t0, c0
        while True:
            t = loop.time()
            await asyncio.sleep(period)
            if t0 is None:
                if start() is None:
                    continue
                t0, c0 = start(), time.thread_time()
                last.update(t=loop.time(), c=c0)
                continue
            if not live():
                return
            lag = loop.time() - t - period
            lags.append(lag)
            if lag > worst["lag_s"]:
                worst.update(lag_s=lag, at_s=t - t0)  # when the stalled wait began
            last.update(t=loop.time(), c=time.thread_time())

    host: dict = {}
    task = loop.create_task(watch(), name="host-watch")
    try:
        yield host
    finally:
        task.cancel()
        wall = last["t"] - t0 if t0 is not None else 0.0
        host.update(main_thread_cpu_frac=(last["c"] - c0) / wall if wall > 0 else 0.0, loop_lag_s=quantiles(lags),
                    worst_stall_at_s=worst["at_s"])


@contextlib.contextmanager
def engine_process(cfg: RunConfig, logs: Path, health_timeout: float = 900.0):
    """Starts the engine server and yields (url, served model name); stops its whole process group on exit. While it
    runs, SIGTERM (a queue's or a script's timeout) exits through this block, so no engine is left holding a shared
    GPU."""
    launch = cfg.launch
    url = f"http://127.0.0.1:{launch.port}"
    argv, extra, served = engine_command(cfg, base_dir=logs.parent / "engine-base")
    env = {**os.environ, **extra}
    logs.mkdir(parents=True, exist_ok=True)
    log = open(logs / "engine.log", "w")
    main = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, _exit_on_sigterm) if main else None
    proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + health_timeout
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"engine exited with {proc.returncode}; see {logs / 'engine.log'}")
            if healthy(url):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("engine did not become healthy")
            time.sleep(1)
        yield url, served
    finally:
        _killpg(proc, signal.SIGTERM)
        try:
            proc.wait(SIGNALLED_ENGINE_WAIT_S if _signalled else 60)
        except subprocess.TimeoutExpired:
            _killpg(proc, signal.SIGKILL)
            proc.wait(10)
        log.close()
        if main:
            signal.signal(signal.SIGTERM, previous)


def tool_config(cfg: RunConfig, *sets: str, ipc: bool = False) -> RunConfig:
    """A run's config as a measurement session's engine (probe, calibration, noise floor, task prep, evaluation,
    pause benchmark): the row's model, context, LM head and scheduling, but synchronous, without LoRA and with room
    for 256 concurrent answers; with ipc, weights pushed from the tool's own process on the engine's GPU."""
    over = ["regime=A", "K=0", "S=1", "lora=false", "M=32", "G=8", *sets]
    if ipc:
        over += ["launch.weight_backend=ipc", f"launch.trainer_gpu='{cfg.launch.engine_gpu}'"]
    return with_overrides(cfg, over)


@contextlib.contextmanager
def tool_engine(cfg: RunConfig, logs: Path):
    """Starts tool_config's engine and yields (url, an HTTP client for it)."""
    with engine_process(cfg, logs) as (url, served):
        yield url, VllmHttpEngine(url, served, max_num_seqs=cfg.M * cfg.G, max_model_len=cfg.max_model_len)


def _killpg(proc: subprocess.Popen, sig):
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, sig)


def trainer_client(cfg: RunConfig, url: str, work: Path, adapter_root: str | None = None) -> ProcTrainerClient:
    launch, train = cfg.launch, cfg.train
    weights = resolve(cfg.regime, cfg.lora).weights
    if weights != "full":
        if adapter_root is None:  # run_row's temporary directory under ADAPTER_ROOT: one per attempt, removed after
            raise ValueError("a LoRA run needs its adapter namespace (adapter_root)")
        sink = {"kind": "adapter", "root": adapter_root}
    elif launch.engine == "fake":
        sink = {"kind": "version", "url": url}
    else:
        sink = {"kind": "vllm", "url": url, "backend": launch.weight_backend, "master_port": MASTER_PORT,
                "world_size": 2}
    settings = {"model": train.model or cfg.model, "device": train.device, "lr": train.lr,
                "warmup_steps": train.warmup_steps,
                "eps_max": cfg.eps_max, "temperature": cfg.temperature,
                "loss_scale": loss_scale(cfg.B, cfg.G, cfg.max_tokens), "micro_tokens": train.micro_tokens,
                "lora_rank": cfg.lora_rank if weights != "full" else None,
                "gradient_checkpointing": train.gradient_checkpointing, "staging_dtype": train.staging_dtype,
                "init_from": cfg.init_from, "profile": train.profile, "seed": cfg.seed}
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "CUDA_VISIBLE_DEVICES": launch.trainer_gpu, **nccl_env(cfg)}
    if launch.weight_backend == "ipc":
        env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    return ProcTrainerClient(settings, sink, workdir=str(work), publication=cfg.publication, env=env,
                             logdir=str(work.parent / "logs"))


async def _run_controller(cfg: RunConfig, url: str, served: str, work: Path, adapter_root: str | None = None):
    from . import tasks
    prompts, reward, stop = tasks.load(cfg)
    engine = VllmHttpEngine(url, served, max_num_seqs=cfg.M * cfg.G, max_model_len=cfg.max_model_len)
    trainer = trainer_client(cfg, url, work, adapter_root)
    # MATH's verifier is slow and signal-bound: score it in worker processes, off the event loop
    with tasks.scoring(reward) as (reward, pool):
        controller = Controller(cfg, engine, trainer, prompts, reward, stop=stop, reward_pool=pool,
                                journal=work / "steps.jsonl")
        try:
            with sigterm_stops(controller):
                async with host_watch(start=lambda: getattr(controller, "t_start", None),
                                      live=lambda: not controller.stopping) as host:
                    summary = await controller.run()
            summary.host = host
        finally:
            trainer.abort()  # a no-op once close() reaped the worker
    summary.lanes.update({lane: lane_fractions(trainer.events, lane) for lane in ("trainer", "sender")})
    return summary, controller, merge(controller.tracer.events, trainer.events), trainer



def row_dir_on_host(cfg: RunConfig, root: Path, gpu: str | None = None) -> Path:
    """The row's directory on this host (its GPU's name is part of the path)."""
    return row_dir(cfg, host_gpu() if gpu is None else gpu, root)


def finished(path: Path, cfg: RunConfig, dry_run: bool = False) -> bool:
    """Whether `path` holds a finished row of this measurement that stands (run_row and the queue then skip it): one
    that is valid, or one whose rerun already happened (results.row_problem; a second failure stands and is
    reported). A finished row of another measurement under the same label is an error; an invalid one is moved
    aside and run again once (HYPOTHESES.md decision 7); one from before rows recorded their identity is moved
    aside and run again. A valid row flagged for its harness stands: the analysis applies that rule. External
    review F01. A dry run moves nothing: it only says whether the row would run."""
    row = Row(path)
    if not row.finished:
        return False
    if "identity" not in row.recorded:
        if not dry_run:
            set_aside(path, "legacy")
        return False
    diff = differences(row.config, to_dict(cfg))
    diff += [k for k, v in row_identity(cfg).items() if row.identity.get(k) != v]
    if diff:
        raise ValueError(f"{path} is a finished row with other {', '.join(diff)}: the label does not tell them "
                         "apart, so give this one a tag")
    problem = row_problem(row.metrics)
    if problem is None:
        return True
    if rerun_pending(path):
        return True  # its rerun had the same problem: reported as it is
    if not dry_run:
        set_aside(path, "invalid")
    return False


def run_row(cfg: RunConfig, root: Path, gpu: str | None = None) -> Path:
    env = environment()
    path = row_dir_on_host(cfg, root, gpu)
    if finished(path, cfg):
        return path
    identity = row_identity(cfg)
    work = path / "work"
    if work.exists() and any(work.iterdir()):  # a failed attempt's: its snapshots must not mix with this one's
        work.rename(path / f"work.stale-{int(time.time())}")
    work.mkdir(parents=True, exist_ok=True)
    if cfg.lora:
        Path(ADAPTER_ROOT).mkdir(parents=True, exist_ok=True)
    # Keep adapter files until the engine exits; each retry gets a new namespace even after a killed attempt.
    adapters = tempfile.TemporaryDirectory(prefix=f"{path.name}-", dir=ADAPTER_ROOT) if cfg.lora \
        else contextlib.nullcontext(None)
    with adapters as adapter_root, engine_process(cfg, path / "logs") as (url, served):
        summary, controller, events, trainer = run_uvloop(_run_controller(cfg, url, served, work, adapter_root))
        # written before the engine stops: a signalled run has seconds left before its supervisor's kill
        write_row(path, cfg, summary, controller.ledger, events, env=env, identity=identity,
                  snapshots={"marks": controller.snapshots, "written": trainer.snapshots})
    if controller.signalled:  # kept, but out of the way: the row's name must stay free for its rerun, and the
        # caller must see the run was cut short
        set_aside(path, "signalled")
        _exit_on_sigterm()
    return path
