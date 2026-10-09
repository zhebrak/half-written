"""The policy trainer: one optimiser step per batch, and the staged copies of the weights the engine will receive.

Model: a Hugging Face causal LM with an fp32 master copy (bf16 autocast on CUDA), or a LoRA adapter on a frozen bf16
base (PEFT, all linear layers). Log-probabilities come from the decoder's final hidden states through the FP32 head in
chunks (logprobs.py), never from full-vocabulary logits of the whole batch. The loss is CISPO against the engine's
mu (loss.py) with the constant normaliser, so micro-batching never changes the gradient (pack.py packs answers).

Staging (PLAN.md §4.1, R4): after each step the weights are copied into one of two buffers in the engine's dtype and
the newer one is marked latest; the sender pushes the latest staged version while the trainer keeps training and
writes the other buffer. A buffer is never written while it is being sent, and the latest is never handed out while
it is being written.
"""
import contextlib
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

from .. import metrics
from ..types import TrainBatch
from .logprobs import token_logprobs
from .loss import cispo_loss
from .pack import pack

TINY = dict(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, tie_word_embeddings=True)


WEIGHT_DECAY = 0.0  # AdamW, identical in every arm (PLAN.md §4.4)
GRAD_CLIP = 1.0


@dataclass
class PolicyTrainerSettings:
    model: str  # a Hugging Face id or path; "tiny-qwen3" builds a small random Qwen3 for tests
    device: str = "cuda"
    lr: float = 1e-6
    warmup_steps: int = 5
    eps_max: float = 5.0
    temperature: float = 1.0
    loss_scale: float = 1.0  # advantages.loss_scale(B, G, max_tokens)
    micro_tokens: int = 16384  # padded tokens per micro-batch
    lora_rank: int | None = None
    gradient_checkpointing: bool = True
    staging_dtype: str = "bfloat16"  # what the engine serves
    init_from: str | None = None  # a snapshot directory (model.safetensors) to start from: the warm start W0
    profile: bool = False  # time each phase of a step (synchronising the GPU at each boundary: slower; diagnosis only)
    seed: int = 0


def load_model(name: str, device: str, seed: int = 0):
    from transformers import AutoModelForCausalLM, Qwen3Config
    torch.manual_seed(seed)
    if name.startswith("tiny-qwen3"):
        return AutoModelForCausalLM.from_config(Qwen3Config(**TINY), dtype=torch.float32).to(device)
    from ..revisions import model_revision
    return AutoModelForCausalLM.from_pretrained(name, revision=model_revision(name), dtype=torch.float32).to(device)


class Staging:
    """Two buffers of {name: tensor}; see the module docstring for the rules. Readers (the sender, a snapshot) hold
    a buffer between acquire (or pin) and release; the trainer writes only the buffer that is not the latest, once
    nobody holds it, and a buffer becomes the latest only once its copy has finished on the GPU: vLLM's packed
    producer reads on its own CUDA streams, which do not wait for the trainer's. Never overwriting the latest is what
    lets a snapshot pin the version staged at its mark (external review F02). The wait it can cost the trainer lasts
    until the push in progress ends, or until the snapshot thread has copied the pinned version to the host (after
    any earlier snapshot it is still writing); the worker traces it as the trainer's stage_wait state."""

    def __init__(self):
        self.buffers: list[dict | None] = [None, None]
        self.versions = [-1, -1]
        self.holds = [0, 0]
        self.latest: int | None = None
        self.writing: int | None = None
        self._cond = threading.Condition()

    def stage(self, version: int, tensors: dict[str, torch.Tensor], dtype: torch.dtype, waiting=None) -> None:
        """waiting(True) before blocking on a held buffer and waiting(False) once it is free (the worker's trace)."""
        free = lambda: any(self.holds[i] == 0 and i != self.latest for i in (0, 1))  # noqa: E731
        with self._cond:
            if waiting is not None and not free():
                waiting(True)
                self._cond.wait_for(free)
                waiting(False)
            self._cond.wait_for(free)
            idx = next(i for i in (0, 1) if self.holds[i] == 0 and i != self.latest)
            self.writing = idx
        buf = self.buffers[idx]
        if buf is None:
            buf = {k: torch.empty_like(v, dtype=dtype) for k, v in tensors.items()}
        with torch.no_grad():
            for k, v in tensors.items():
                buf[k].copy_(v)
        if any(t.is_cuda for t in buf.values()):
            torch.cuda.current_stream().synchronize()
        with self._cond:
            self.buffers[idx], self.versions[idx], self.latest, self.writing = buf, version, idx, None
            self._cond.notify_all()

    def acquire(self, at_least: int, timeout: float | None = None) -> tuple[int, dict, int]:
        """(version, tensors, handle) of the latest staged buffer once it holds a version >= at_least; the buffer
        stays as it is until release(handle)."""
        with self._cond:
            ok = self._cond.wait_for(lambda: self.latest is not None and self.writing != self.latest
                                     and self.versions[self.latest] >= at_least, timeout)
            if not ok:
                raise TimeoutError(f"no staged version >= {at_least}")
            idx = self.latest
            self.holds[idx] += 1
            return self.versions[idx], self.buffers[idx], idx

    def pin(self, version: int) -> tuple[dict, int]:
        """(tensors, handle) of the buffer staged as `version`, held until release(handle); never waits. A snapshot
        pins the version staged at its mark as soon as the request arrives, before the trainer can stage two more
        versions (the controller sends the request before the batch that could follow), so a miss is a bug."""
        with self._cond:
            idx = next((i for i in (0, 1) if self.versions[i] == version and self.writing != i), None)
            if idx is None:
                raise RuntimeError(f"version {version} is not staged (buffers hold {self.versions}, writing "
                                   f"{self.writing})")
            self.holds[idx] += 1
            return self.buffers[idx], idx

    def release(self, handle: int) -> None:
        with self._cond:
            self.holds[handle] -= 1
            self._cond.notify_all()


class PolicyTrainer:
    def __init__(self, s: PolicyTrainerSettings):
        self.s = s
        self.device = torch.device(s.device)
        if self.device.type == "cuda":  # the only fp32 matmul is the head, on bf16-valued inputs: exact in TF32
            torch.backends.cuda.matmul.allow_tf32 = True
        model = load_model(s.model, s.device, s.seed)
        if s.init_from:
            from safetensors.torch import load_file
            state = load_file(f"{s.init_from}/model.safetensors", device=s.device)
            state = {k: v.float() for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected or set(missing) - {"lm_head.weight"}:
                raise ValueError(f"{s.init_from}: missing {missing[:3]}, unexpected {unexpected[:3]}")
        if s.lora_rank:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(r=s.lora_rank, lora_alpha=2 * s.lora_rank,
                                                     target_modules="all-linear", lora_dropout=0.0))
        if s.gradient_checkpointing and self.device.type == "cuda":
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=s.lr, weight_decay=WEIGHT_DECAY)
        self.steps = 0
        self.staging = Staging()
        self.dtype = getattr(torch, s.staging_dtype)

    # -- weights as the engine sees them

    def engine_tensors(self) -> dict[str, torch.Tensor]:
        """Full weights as named_parameters() (what vLLM's ModuleSource sends: a tied LM head appears once, as the
        embedding), or for LoRA the adapter under PEFT's names."""
        if self.s.lora_rank:
            from peft.utils import get_peft_model_state_dict
            return get_peft_model_state_dict(self.model)
        return dict(self.model.named_parameters())

    def peft_config(self) -> dict | None:
        return self.model.peft_config["default"].to_dict() if self.s.lora_rank else None

    def stage(self, version: int, waiting=None) -> None:
        self.staging.stage(version, self.engine_tensors(), self.dtype, waiting)

    # -- a step

    def _decoder(self):
        base = self.model.get_base_model() if self.s.lora_rank else self.model
        return base.get_decoder()

    def _head(self) -> torch.Tensor:
        base = self.model.get_base_model() if self.s.lora_rank else self.model
        return base.get_output_embeddings().weight

    def logprobs(self, samples, micro: list[int]) -> list[torch.Tensor]:
        """Log pi of each sample's answer tokens, for the samples of one micro-batch (with grad)."""
        rows = [samples[i] for i in micro]
        width = max(len(s.prompt_ids) + len(s.token_ids) for s in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for r, s in enumerate(rows):
            seq = list(s.prompt_ids) + list(s.token_ids)
            ids[r, :len(seq)] = torch.tensor(seq)
            mask[r, :len(seq)] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        cuda = self.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16) if cuda else contextlib.nullcontext():
            hidden = self._decoder()(input_ids=ids, attention_mask=mask).last_hidden_state
        # one head pass over every answer token of the micro-batch (not one per sample: each pass re-reads the
        # 152k x 1,024 embedding and launches its own kernels)
        states = [hidden[r, len(s.prompt_ids) - 1:len(s.prompt_ids) - 1 + len(s.token_ids)] for r, s in enumerate(rows)]
        tokens = torch.tensor([t for s in rows for t in s.token_ids], dtype=torch.long, device=self.device)
        flat = token_logprobs(torch.cat(states), self._head(), tokens, temperature=self.s.temperature)
        return list(torch.split(flat, [len(s.token_ids) for s in rows]))

    def step(self, batch: TrainBatch) -> dict:
        s, samples = self.s, batch.samples
        lr = s.lr * min(1.0, (self.steps + 1) / max(1, s.warmup_steps))
        for group in self.opt.param_groups:
            group["lr"] = lr
        self.model.train()
        clock = _PhaseClock(s.profile and self.device.type == "cuda")
        self.opt.zero_grad(set_to_none=True)
        logp_all, loss_total = [], 0.0
        order = []
        micros = pack([len(x.prompt_ids) + len(x.token_ids) for x in samples], s.micro_tokens)
        for micro in micros:
            clock.mark("other")
            logps = self.logprobs(samples, micro)
            loss = 0.0
            for i, logp in zip(micro, logps):
                x = samples[i]
                mu = torch.tensor(x.logprobs, device=self.device, dtype=torch.float32)
                adv = torch.full_like(mu, x.advantage)
                part, _ = cispo_loss(logp, mu, adv, s.eps_max, s.loss_scale, stats=False)
                loss = loss + part
            clock.mark("forward")
            loss.backward()
            clock.mark("backward")
            loss_total += float(loss.detach())
            flat = torch.cat(logps).detach().float().cpu().numpy()
            logp_all += list(np.split(flat, np.cumsum([lp.numel() for lp in logps])[:-1]))
            order += micro
        clock.mark("other")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.params, GRAD_CLIP))
        self.opt.step()
        clock.mark("optimizer")
        self.steps += 1
        by_sample = dict(zip(order, logp_all))
        logp = np.concatenate([by_sample[i] for i in range(len(samples))]) if samples else np.zeros(0)
        mu = np.concatenate([np.asarray(x.logprobs) for x in samples]) if samples else np.zeros(0)
        codes = (np.concatenate([np.asarray(x.context_codes, dtype=np.int64) for x in samples]) if samples
                 else np.zeros(0))
        stats = metrics.token_stats(logp, mu, codes, s.eps_max)
        stats.update(loss=loss_total, grad_norm=grad_norm, lr=lr, micro_batches=len(micros), **clock.seconds())
        return stats


class _PhaseClock:
    """Seconds per phase of a step, with the GPU synchronised at each boundary (only when enabled)."""

    def __init__(self, enabled: bool):
        self.enabled, self.t, self.acc = enabled, None, {}
        if enabled:
            torch.cuda.synchronize()
            self.t = time.perf_counter()

    def mark(self, phase: str):
        if self.enabled:
            torch.cuda.synchronize()
            now = time.perf_counter()
            self.acc[phase] = self.acc.get(phase, 0.0) + now - self.t
            self.t = now

    def seconds(self) -> dict:
        return {f"t_{k}": v for k, v in self.acc.items()}
