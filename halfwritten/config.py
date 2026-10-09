"""Run configuration: one YAML file per run, resolved into RunConfig and written back beside its results.

The knobs that define matched staleness (PLAN.md §4.2) are identical across arms of a comparison: B groups per batch,
G answers per group, at most M groups in flight, buffer bound K (the trainer starts step s+1 only if every group not
yet consumed was dispatched at version >= s+1-K), and a push every S optimiser steps.
"""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

import yaml

from .regimes import resolve


@dataclass
class SimSettings:
    """FakeEngine and SimTrainer parameters for `python -m halfwritten sim` and the tests."""
    prompt_len: int = 64
    length_mean: float = 200.0
    length_sigma: float = 0.8
    empirical_lengths: list[int] | None = None  # answer lengths to draw from (a ledger's), instead of the lognormal
    max_num_seqs: int = 256
    kv_tokens: int = 1 << 20
    max_num_batched_tokens: int = 8192  # per-step token budget; a prefill must fit whole (chunked prefill off)
    delivery: Literal["immediate", "one_step_late", "held_until_resume", "bursty"] = "immediate"
    submit_latency: float = 0.0  # seconds before a request reaches the engine (HTTP); can land after a pause
    stats_lag: bool = False  # counters published with each step's outputs, as vLLM's scheduler stats are
    async_scheduling: bool = False  # a pause lands after the next step's admissions were planned (vLLM's default)
    silent_waiting_aborts: bool = False  # an abort ends a never-scheduled request with no final chunk (vLLM 0.30)
    cost: dict = field(default_factory=dict)  # CostModel overrides
    train_s_fixed: float = 0.2
    train_s_per_token: float = 3e-5  # 6 * 0.6e9 FLOPs per token at ~120 TFLOP/s
    push_s: float = 0.05  # a push: full weights over NCCL, or an adapter written and loaded
    send_slowdown: float = 0.0  # fraction a step slows while a staged push overlaps it (GPU1 contention)
    p_correct: float = 0.5  # chance an answer is correct


@dataclass
class TrainSettings:
    """The trainer worker's optimisation knobs (trainer/model.py PolicyTrainerSettings gets these plus the run's)."""
    lr: float = 1e-6
    warmup_steps: int = 5
    micro_tokens: int = 16384  # padded tokens per micro-batch
    gradient_checkpointing: bool = True
    staging_dtype: str = "bfloat16"
    profile: bool = False  # per-phase step timings in steps.parquet (t_forward, t_backward, t_optimizer; slower)
    device: str = "cuda"
    model: str | None = None  # the trainer's model when it differs from the engine's ("tiny-qwen3" in CPU tests)


@dataclass
class LaunchSettings:
    """How `python -m halfwritten run` lays a row out on the host (PLAN.md §5.2)."""
    engine: Literal["vllm", "fake"] = "vllm"
    port: int = 8100
    engine_gpu: str = "0"
    trainer_gpu: str = "1"  # equal to engine_gpu: colocated, which needs weight_backend "ipc"
    weight_backend: Literal["nccl", "ipc"] = "nccl"
    gpu_memory_utilization: float = 0.9
    async_scheduling: bool = True  # per the session's canary
    head_fp32: bool = False  # vLLM's LM head writes fp32 logits (--hf-overrides head_dtype); see bench/noise_floor.py
    stream_interval: int = 1  # tokens per streamed chunk; >1 holds tokens past a pause (vLLM does not flush them)
    api_servers: int = 1  # vLLM API server processes (--api-server-count); one saturates near 8k chunks/s
    fake_lengths: str = "lognormal:200,0.8"  # the fake server's natural answer lengths
    fake_vocab: int = 151_936
    # in-place LoRA arms: clear the adapter record's load_inplace flag after each in-place load, as SkyRL and prime-rl
    # do (halfwritten/engines/vllm_serve.py); without it vLLM 0.30 runs them wrong, so config refuses them
    lora_inplace_fix: bool = False


@dataclass
class RunConfig:
    regime: str = "D"
    lora: bool = False
    task: str = "sim"
    model: str = "fake"
    tag: str = ""  # distinguishes rows that differ in something the label does not name (a length distribution)
    seed: int = 0
    B: int = 32  # groups (prompts) per optimiser step
    G: int = 8  # answers per prompt
    M: int = 32  # groups in flight
    K: int = 4  # buffer staleness bound, in versions
    S: int = 1  # optimiser steps per push
    max_tokens: int = 1024
    max_model_len: int = 4096
    forced_lengths: str | None = None  # answer lengths forced with ignore_eos (systems map): a lengths.parse spec
    temperature: float = 1.0
    budget_s: float | None = 2700.0  # wall-clock budget from the first dispatch
    max_steps: int | None = None
    timeout_s: float = 900.0  # every engine and trainer await (R6)
    publication: Literal["staged", "lockstep"] = "staged"
    eps_max: float = 5.0  # CISPO truncation of pi/mu
    lora_rank: int = 1
    reward_timeout_s: float = 30.0  # a reward that takes longer scores 0 and is counted
    snapshot_every_s: float | None = None  # wall-clock marks at which the trainer saves weights for offline eval
    snapshot_every_steps: int | None = None  # every n-th version (Phase 1's dense checkpoint series, from arm A)
    init_from: str | None = None  # the warm start W0's snapshot directory: the trainer loads it, v0 pushes it
    prompt_filter: str | None = None  # bench.prompt_filter output: train only on the prompts it keeps
    # what bench.evaluate scores the snapshots on: "dev" (held out of training: pilots, curves, decisions) or "test"
    # (the confirmatory grids set it); dev by default, so test is read only when asked for (external review)
    eval_split: str = "dev"
    sim: SimSettings = field(default_factory=SimSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    launch: LaunchSettings = field(default_factory=LaunchSettings)

    def validate(self) -> "RunConfig":
        r = resolve(self.regime, self.lora)
        if r.synchronous and (self.K != 0 or self.S != 1):
            raise ValueError(f"{self.regime} is synchronous: K must be 0 and S 1")
        if not r.synchronous and not (self.K >= 1 and 1 <= self.S <= self.K + 1):
            raise ValueError("async regimes need K >= 1 and 1 <= S <= K+1, else the trainer waits on a push that "
                             "only its own next step would trigger")
        if min(self.B, self.G, self.M, self.max_tokens) < 1:
            raise ValueError("B, G, M and max_tokens must be positive")
        if self.budget_s is None and self.max_steps is None:
            raise ValueError("set budget_s or max_steps")
        if r.weights != "full" and self.launch.api_servers > 1:  # adapters are registered per API server process
            raise ValueError("LoRA arms need launch.api_servers = 1: an adapter loaded through one API server process "
                             "is unknown to the others, which receive most of the requests")
        if self.eval_split not in ("dev", "test"):
            raise ValueError(f"eval_split {self.eval_split!r}: dev or test")
        if self.publication not in ("staged", "lockstep"):
            raise ValueError(f"unknown publication {self.publication!r}")
        if self.max_tokens >= self.max_model_len:
            raise ValueError("max_tokens must leave room for the prompt within max_model_len")
        if self.forced_lengths is not None and not self.forced_lengths.startswith("parquet:"):  # may not exist yet
            from .lengths import parse
            parse(self.forced_lengths)
        if self.launch.engine_gpu == self.launch.trainer_gpu and self.launch.engine == "vllm" \
                and self.launch.weight_backend != "ipc" and not self.lora:
            raise ValueError("a trainer on the engine's GPU needs weight_backend ipc (NCCL needs two devices)")
        return self


def load(path: str | Path) -> RunConfig:
    return from_dict(yaml.safe_load(Path(path).read_text()) or {})


_SECTIONS = {"sim": SimSettings, "train": TrainSettings, "launch": LaunchSettings}


# keys renamed since rows were first written: old row configs (data/, results/) still load
RENAMED = {"quiet_s": "fence_quiet_s", "fence_min_s": "fence_floor_s", "lengths": "forced_lengths"}
RENAMED_SIM = {"reward_p": "p_correct"}
# keys dropped since (review-2026-10-02 plan, section 2.2), with the value the code now assumes, or ANY where the
# key had no effect on the path that remains: an archived row holding the assumed value loads and compares equal
# to today's config (differences); another value is refused by name, and counts as a difference
ANY = object()
DROPPED = {
    "fence_exact": True,  # the exact fence is the only fence (the window fence went with it)
    "fence_quiet_s": ANY, "fence_floor_s": ANY,  # the window fence's parameters: no effect under the exact fence
    "reset_counts_as_preemption": True,  # vLLM 0.30 counts E's cache resets as preemptions (the probe fixture)
    "stop": [],  # stop strings come from the task (tasks.STOPS)
    "sim.seed": ANY,  # the run's seed drives the simulator's lengths (it was 0 in every row)
    "sim.adapter_push_s": ANY,  # a LoRA push is push_s like any other
    "launch.master_port": ANY, "launch.adapter_root": ANY,  # constants (launch.py): never the measurement
}


def _drop(d: dict) -> dict:
    """A recorded config without its dropped keys; a dropped key at another value than assumed raises."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items() if not _dropped(f"{k}.{kk}", vv)}
        elif _dropped(k, v):
            continue
        out[k] = v
    return out


def _dropped(key: str, value) -> bool:
    if key not in DROPPED:
        return False
    assumed = DROPPED[key]
    if assumed is ANY or value == assumed or (value is None and assumed is not ANY):
        return True
    raise ValueError(f"config key {key} was dropped; the code now assumes {assumed!r}, this config says {value!r}")


def from_dict(d: dict) -> RunConfig:
    d = {RENAMED.get(k, k): v for k, v in d.items()}
    if isinstance(d.get("sim"), dict):
        d["sim"] = {RENAMED_SIM.get(k, k): v for k, v in d["sim"].items()}
    d = _drop(d)
    known = {f.name for f in fields(RunConfig)}
    if unknown := set(d) - known:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    sections = {}
    for name, cls in _SECTIONS.items():
        sub = d.pop(name, None) or {}
        if unknown := set(sub) - {f.name for f in fields(cls)}:
            raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
        if name == "launch":  # GPU ids are CUDA_VISIBLE_DEVICES strings; YAML and --set parse a bare 1 as a number
            sub = {k: str(v) if k in ("engine_gpu", "trainer_gpu") else v for k, v in sub.items()}
        sections[name] = cls(**sub)
    return RunConfig(**d, **sections).validate()


# Settings that are not the measurement: where the processes listen. GPU indices count only as a layout (colocated
# or split), not as which index (the one-GPU pilot alternates them between seeds).
NOT_THE_MEASUREMENT = ("launch.port",)
# what an experiment's arms and seed blocks may differ in (bench.stats: everything else must match across rows)
REGIME_KEYS = ("regime", "K", "lora", "tag", "seed")


def _leaves(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out |= _leaves(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v}
    return out


def _normal(d: dict) -> dict:
    """A recorded config as today's code reads it, flattened: renamed keys renamed, a key older code did not
    record at its default (the row ran at the default then), a dropped key gone when it holds the assumed value and
    kept (so it differs) when not, GPU indices reduced to the layout."""
    d = {RENAMED.get(k, k): v for k, v in d.items()}
    if isinstance(d.get("sim"), dict):
        d["sim"] = {RENAMED_SIM.get(k, k): v for k, v in d["sim"].items()}
    flat = _leaves(d)
    for k in [k for k in flat if k in DROPPED]:
        assumed = DROPPED[k]
        if assumed is ANY or flat[k] == assumed or flat[k] is None:
            del flat[k]
    out = _leaves(to_dict(RunConfig())) | flat
    if "launch.engine_gpu" in out or "launch.trainer_gpu" in out:
        out["launch.colocated"] = str(out.pop("launch.engine_gpu", "")) == str(out.pop("launch.trainer_gpu", ""))
    return out


def differences(recorded: dict, now: dict, ignore: tuple[str, ...] = NOT_THE_MEASUREMENT) -> list[str]:
    """The settings, leaf by leaf, in which two configs (to_dict form, as a row's config.yaml records them) differ:
    the one definition of "the same measurement" for reruns (launch.finished), the row writer and the analysis
    (bench.stats). Numbers compare by value (300 is 300.0); the simulator's settings count only for task sim."""
    a, b = _normal(recorded), _normal(now)
    sim = "sim" in (a.get("task"), b.get("task"))
    return [k for k in sorted(set(a) | set(b)) if k not in ignore and (sim or not k.startswith("sim."))
            and a.get(k) != b.get(k)]


def with_overrides(cfg: RunConfig, sets: list[str]) -> RunConfig:
    """`key=value` and `section.key=value` overrides (YAML-parsed values), as the CLI's --set takes them."""
    d = to_dict(cfg)
    for item in sets:
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        head, _, tail = key.partition(".")
        if tail:
            d.setdefault(head, {})[tail] = value
        else:
            d[head] = value
    return from_dict(d)


def to_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)
