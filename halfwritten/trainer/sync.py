"""Weight sinks: how the trainer worker delivers a staged version to the engine between Grant(v) and Pushed(v).
Each takes the staged tensors (model.Staging), never the live parameters, so training can continue meanwhile (R4).

    VllmSink     vLLM's weight-transfer API (v0.26+; the pre-0.28 trainer API is gone): trainer_init once, then per
                 push send_weights() from a WeightSource over the staged buffer, which runs start/update/finish.
                 backend "nccl" for an engine on another GPU, "ipc" for one sharing the trainer's GPU (colocated; the
                 server needs VLLM_ALLOW_INSECURE_SERIALIZATION=1 for the pickled CUDA handles). The client stamps
                 finish_weight_update with our version, so GET /weight_info names exactly what was pushed.
                 A full sync evicts every LoRA adapter, so LoRA runs never use it.
    AdapterSink  a LoRA adapter written atomically under /dev/shm (vLLM loads adapters only from a path); the
                 controller then calls load_lora_adapter in place (B, D, E) or under a new name (F).
    VersionSink  no tensors: POST /update_weight_version. The fake server applies it as the new weights; on vLLM it
                 is the no-op swap of the noise-floor measurement (same weights, new version label).

canary_update() turns a version into the canary's next one (PLAN.md §5.3): the tied embedding / LM-head rows
permuted by a fixed derangement (every position's KV changes, so D and E separate) and the signs of a fixed half of
the final norm's weights flipped (the logits change but no KV does). The permutation alone is a relabelling of the
vocabulary, and copying from the context survives relabelling: on a repetitive answer old and new weights then
predict the same token and the boundary cannot be seen (the first probe, 2026-10-01, E's post-swap tokens matched
the old weights 96% of the time).
"""
import json
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from ..types import Pushed


def _nbytes(tensors: dict) -> int:
    return sum(t.numel() * t.element_size() for t in tensors.values())


class VersionSink:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def push(self, version: int, tensors: dict) -> Pushed:
        import urllib.request
        t0 = time.monotonic()
        req = urllib.request.Request(f"{self.url}/update_weight_version", method="POST",
                                     data=json.dumps({"new_version": str(version)}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=60).close()  # an error status raises HTTPError
        return Pushed(version, 0, time.monotonic() - t0)


class AdapterSink:
    """Writes each version's adapter to <root>/v<version> atomically: the files go to a temporary directory that is
    renamed into place, so vLLM never reads a half-written adapter. `write(path, tensors)` saves one."""

    def __init__(self, root: str | Path, write: Callable, keep: int = 8):
        self.root, self.write, self.keep = Path(root), write, keep
        self.root.mkdir(parents=True, exist_ok=True)

    def push(self, version: int, tensors: dict | None = None) -> Pushed:
        t0 = time.monotonic()
        final, tmp = self.root / f"v{version}", self.root / f".v{version}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        self.write(tmp, tensors) if tensors is not None else self.write(tmp)
        size = sum(f.stat().st_size for f in tmp.rglob("*") if f.is_file())
        os.replace(tmp, final)
        for old in sorted(self.root.glob("v*"), key=lambda p: int(p.name[1:]))[:-self.keep]:
            shutil.rmtree(old, ignore_errors=True)
        return Pushed(version, size, time.monotonic() - t0, str(final))


def peft_adapter_writer(peft_config: dict):
    """A writer for AdapterSink: the staged PEFT state dict as adapter_model.safetensors + adapter_config.json."""
    def write(path: Path, tensors: dict):
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(path / "adapter_model.safetensors"))
        (path / "adapter_config.json").write_text(json.dumps(peft_config, default=list))
    return write


class _VersionedClient:
    """vLLM's HTTP weight-sync client, with our version on finish_weight_update (send_weights passes none)."""

    def __init__(self, url: str):
        from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
        self._inner = HTTPVLLMWeightSyncClient(url)
        self.version: str | None = None

    def init_weight_transfer_engine(self, init_info):
        return self._inner.init_weight_transfer_engine(init_info)

    def start_weight_update(self):
        return self._inner.start_weight_update()

    def update_weights(self, update_info):
        return self._inner.update_weights(update_info)

    def finish_weight_update(self, weight_version=None):
        return self._inner.finish_weight_update(weight_version=self.version)

    def __getattr__(self, name):  # any other client call a backend makes goes to vLLM's client unchanged
        return getattr(self._inner, name)


def staged_source(tensors: dict):
    """A vLLM WeightSource over a staged {name: tensor} buffer, in a fixed order."""
    from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource

    class StagedSource(WeightSource):
        def metadata(self):
            return [ParamMeta(n, t.dtype, tuple(t.shape)) for n, t in tensors.items()]

        def __iter__(self):
            yield from tensors.items()

    return StagedSource()


class VllmSink:
    def __init__(self, url: str, backend: str = "nccl", master_address: str = "127.0.0.1",
                 master_port: int = 29555, world_size: int = 2, packed: bool = True):
        self.url, self.backend = url, backend
        self.master_address, self.master_port, self.world_size, self.packed = master_address, master_port, \
            world_size, packed
        self.engine = None

    def _init(self, tensors: dict):
        from vllm.distributed.weight_transfer.factory import (
            WeightTransferTrainerFactory,
        )
        self.client = _VersionedClient(self.url)
        if self.backend == "ipc":
            from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo
            info = IPCTrainerInitInfo(rank=0, packed=self.packed)
        else:
            from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo
            info = NCCLTrainerInitInfo(self.master_address, self.master_port, self.world_size, rank=0,
                                       packed=self.packed)
        self.engine = WeightTransferTrainerFactory.trainer_init(info, client=self.client,
                                                                source=staged_source(tensors))

    def push(self, version: int, tensors: dict) -> Pushed:
        import torch
        # IPC's UUID is taken from the thread's current device, even when the tensors live elsewhere.
        device = next(iter(tensors.values())).device
        if self.backend == "ipc":
            if device.type != "cuda" or any(t.device != device for t in tensors.values()):
                raise ValueError("IPC weights must all be on the evaluation server's CUDA device")
            with torch.cuda.device(device):
                return self._push(version, tensors)
        return self._push(version, tensors)

    def _push(self, version: int, tensors: dict) -> Pushed:
        if self.engine is None:
            self._init(tensors)
        self.engine.source = staged_source(tensors)
        self.client.version = str(version)
        t0 = time.monotonic()
        self.engine.send_weights()
        return Pushed(version, _nbytes(tensors), time.monotonic() - t0)


def make_sink(spec: dict, *, peft_config: dict | None = None):
    kind = spec["kind"]
    if kind == "version":
        return VersionSink(spec["url"])
    if kind == "adapter":
        return AdapterSink(spec["root"], peft_adapter_writer(peft_config or {}), keep=spec.get("keep", 8))
    if kind == "vllm":
        return VllmSink(spec["url"], spec.get("backend", "nccl"), spec.get("master_address", "127.0.0.1"),
                        spec.get("master_port", 29555), spec.get("world_size", 2), spec.get("packed", True))
    raise ValueError(f"unknown sink {kind!r}")


def permute_vocab(tensors: dict, seed: int = 0, key: str = "model.embed_tokens.weight") -> dict:
    """The canary's next version: rows of the tied embedding permuted by a fixed derangement (no row stays put), so
    the new model's greedy choice differs from the old one's almost everywhere."""
    import torch
    out = dict(tensors)
    w = tensors[key]
    g = torch.Generator().manual_seed(seed)
    n = w.shape[0]
    perm = torch.randperm(n, generator=g)
    fixed = perm == torch.arange(n)
    if fixed.any():  # rotate the fixed points among themselves, or into a neighbour when there is only one
        idx = torch.nonzero(fixed).flatten()
        perm[idx] = perm[idx.roll(1)] if len(idx) > 1 else perm[(idx + 1) % n]
        if len(idx) == 1:
            perm[(idx + 1) % n] = idx
    out[key] = w[perm.to(w.device)].contiguous()
    return out


def canary_update(tensors: dict, seed: int = 0, emb: str = "model.embed_tokens.weight",
                  norm: str = "model.norm.weight") -> dict:
    """permute_vocab plus a sign flip of a fixed random half of the final norm's weights (see the module docstring)."""
    import torch
    out = permute_vocab(tensors, seed, emb)
    w = tensors[norm]
    flip = torch.randperm(w.numel(), generator=torch.Generator().manual_seed(seed + 1))[: w.numel() // 2]
    sign = torch.ones(w.numel(), dtype=w.dtype)
    sign[flip] = -1
    out[norm] = (w * sign.view_as(w).to(w.device)).contiguous()
    return out
