"""The real trainer path on CPU: staging rules, a CISPO step on a tiny random Qwen3 (full and LoRA), the canary's
vocabulary derangement, and the whole loop end to end: controller -> HTTP -> fake `vllm serve`, and a trainer worker
subprocess that trains the tiny model and pushes versions back to the server."""
import asyncio
import os
import threading
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import pytest

from bench.canary import compare  # noqa: E402
from halfwritten.config import RunConfig  # noqa: E402
from halfwritten.controller import Controller  # noqa: E402
from halfwritten.engines.fake import CostModel, FakeEngine, LengthModel  # noqa: E402
from halfwritten.engines.fake_server import serve  # noqa: E402
from halfwritten.engines.vllm_http import VllmHttpEngine  # noqa: E402
from halfwritten.rewards import sim_reward  # noqa: E402
from halfwritten.tasks import synthetic  # noqa: E402
from halfwritten.trainer.client import ProcTrainerClient  # noqa: E402
from halfwritten.trainer.model import (  # noqa: E402
    PolicyTrainer,
    PolicyTrainerSettings,
    Staging,
)
from halfwritten.trainer.sync import (
    AdapterSink,
    permute_vocab,  # noqa: E402
)
from halfwritten.types import TrainBatch, TrainSample  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _batch(step=0, n=4, vocab=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    samples = []
    for i in range(n):
        p, a = int(torch.randint(3, 9, (1,), generator=g)), int(torch.randint(2, 12, (1,), generator=g))
        tokens = tuple(torch.randint(0, vocab, (a,), generator=g).tolist())
        samples.append(TrainSample(f"s{i}", f"g{i // 2}", tuple(torch.randint(0, vocab, (p,), generator=g).tolist()),
                                   tokens, tuple([-6.2] * a), float(i % 2), 1.0 if i % 2 else -1.0,
                                   (0,) * a, (0,) * a, (0,) * a))
    return TrainBatch(step, tuple(samples), n // 2, 0)


def test_staging_never_hands_out_a_buffer_being_written():
    st = Staging()
    st.stage(0, {"w": torch.zeros(3)}, torch.float32)
    v, buf, h = st.acquire(0)
    assert v == 0
    st.stage(1, {"w": torch.ones(3)}, torch.float32)  # goes to the other buffer while 0 is being sent
    assert buf["w"].sum() == 0
    st.release(h)
    v, buf, h = st.acquire(1)
    assert v == 1 and buf["w"].sum() == 3
    st.release(h)
    with pytest.raises(TimeoutError):
        st.acquire(5, timeout=0.05)
    got = []
    t = threading.Thread(target=lambda: got.append(st.acquire(2, timeout=5)))
    t.start()
    time.sleep(0.05)
    st.stage(2, {"w": torch.full((3,), 2.0)}, torch.float32)  # the waiting sender gets it as soon as it is written
    t.join()
    assert got[0][0] == 2 and got[0][1]["w"].sum() == 6
    st.release(got[0][2])


def test_a_snapshot_pins_the_version_staged_at_its_mark():
    """External review F02: a snapshot copied "the latest" whenever its thread got to it, so a backlog (a 1.2 GB write
    in progress) or a step staged after the budget put newer weights under the mark. The worker now pins the mark's
    version on arrival; the trainer never overwrites the latest buffer, so that version is still there, and a pinned
    buffer waits for its copy."""
    st = Staging()
    st.stage(1, {"w": torch.full((2,), 1.0)}, torch.float32)
    tensors, handle = st.pin(1)  # the mark, at v1
    st.stage(2, {"w": torch.full((2,), 2.0)}, torch.float32)  # the trainer goes on: the other buffer
    done = threading.Event()
    threading.Thread(target=lambda: (st.stage(3, {"w": torch.full((2,), 3.0)}, torch.float32), done.set())).start()
    assert not done.wait(0.3)  # v3 would overwrite v1's buffer: it waits for the copy
    assert tensors["w"].tolist() == [1.0, 1.0]
    st.release(handle)
    assert done.wait(5) and st.acquire(3, timeout=5)[0] == 3
    with pytest.raises(RuntimeError, match="not staged"):
        st.pin(7)


def test_staging_never_overwrites_the_latest_version():
    """The sender holds v1's buffer; the trainer stages v2 into the other, then v3 must wait for the sender rather
    than overwrite v2 (the latest, the one a snapshot taken now would pin)."""
    st = Staging()
    st.stage(1, {"w": torch.full((2,), 1.0)}, torch.float32)
    _, _, hs = st.acquire(1)
    st.stage(2, {"w": torch.full((2,), 2.0)}, torch.float32)
    tensors, hp = st.pin(2)
    done = threading.Event()
    threading.Thread(target=lambda: (st.stage(3, {"w": torch.full((2,), 3.0)}, torch.float32), done.set())).start()
    assert not done.wait(0.3) and tensors["w"].tolist() == [2.0, 2.0]
    st.release(hs)  # the push ends: v3 goes where v1 was
    assert done.wait(5)
    assert tensors["w"].tolist() == [2.0, 2.0]
    st.release(hp)
    waits = []  # the worker's trace: a stage that waits for a held buffer says so, and when it is free
    _, _, hs = st.acquire(3)
    st.stage(4, {"w": torch.full((2,), 4.0)}, torch.float32, waiting=waits.append)  # v2's buffer is free: no wait
    assert waits == []
    t = threading.Thread(target=lambda: st.stage(5, {"w": torch.full((2,), 5.0)}, torch.float32,
                                                 waiting=waits.append))
    t.start()
    time.sleep(0.2)
    assert waits == [True]  # v3's buffer is the sender's
    st.release(hs)
    t.join(5)
    assert waits == [True, False]


@pytest.mark.parametrize("lora", [None, 2])
def test_a_step_moves_only_the_trainable_weights(lora):
    tr = PolicyTrainer(PolicyTrainerSettings(model="tiny-qwen3", device="cpu", lr=1e-2, warmup_steps=1, micro_tokens=32,
                                       lora_rank=lora, staging_dtype="float32"))
    before = {k: v.detach().clone() for k, v in tr.model.named_parameters()}
    stats = tr.step(_batch())
    assert stats["tokens"] > 0 and torch.isfinite(torch.tensor(stats["loss"])) and stats["grad_norm"] > 0
    moved = {k for k, v in tr.model.named_parameters() if not torch.equal(v, before[k])}
    if lora:
        assert moved and all("lora_" in k for k in moved)
        assert all("lora" in k for k in tr.engine_tensors())
    else:
        assert "model.embed_tokens.weight" in moved and "lm_head.weight" not in tr.engine_tensors()  # tied, sent once


def test_permute_vocab_moves_every_row():
    w = torch.arange(50.0)[:, None].repeat(1, 4)
    out = permute_vocab({"model.embed_tokens.weight": w, "x": torch.ones(2)}, seed=3)
    moved = out["model.embed_tokens.weight"][:, 0]
    assert sorted(moved.tolist()) == list(range(50)) and all(moved[i] != i for i in range(50))
    assert out["x"] is not None and torch.equal(permute_vocab({"model.embed_tokens.weight": w}, 3)[
        "model.embed_tokens.weight"], out["model.embed_tokens.weight"])


@pytest.mark.parametrize("regime", ["D", "C1"])
def test_end_to_end_controller_worker_and_server(regime, tmp_path):
    async def main():
        fake = FakeEngine(lengths=LengthModel(mean=24, sigma=0.6), cost=CostModel(5e-3, 1e-5, 0, 1e-6), vocab=512)
        await fake.start()
        runner, url = await serve(fake)
        trainer = ProcTrainerClient(
            {"model": "tiny-qwen3", "device": "cpu", "lr": 1e-3, "micro_tokens": 2048, "staging_dtype": "float32"},
            {"kind": "version", "url": url}, workdir=str(tmp_path),
            env={**os.environ, "PYTHONPATH": str(ROOT)})
        cfg = RunConfig(regime=regime, B=2, G=4, M=2, K=2, max_tokens=64, budget_s=12.0, timeout_s=120.0,
                        snapshot_every_s=5.0)
        engine = VllmHttpEngine(url, "fake", max_num_seqs=64, max_model_len=4096)
        controller = Controller(cfg, engine, trainer, synthetic(64, 8, vocab=512), sim_reward())
        try:
            summary = await controller.run()
        finally:
            await runner.cleanup()
        return summary, compare(controller.ledger, fake.truth), trainer, controller.snapshots

    summary, report, trainer, summary_marks = asyncio.run(main())
    assert summary.valid, (summary.error, summary.violations[:3], (tmp_path / "trainer.log").read_text()[-2000:])
    assert summary.steps >= 2 and len(summary.syncs) >= 2
    assert summary.steps_log[0].get("grad_norm", 0) > 0  # real gradients from the tiny model
    assert report["ok"], report
    assert {e.lane for e in trainer.events} >= {"trainer", "sender"}
    # snapshots at 5 s, 10 s and the budget, each complete on disk, no temporary directory left behind
    assert [t for t, _, _ in trainer.snapshots] == ["t01", "t02", "tend"]
    assert all((Path(p) / "model.safetensors").stat().st_size > 0 for _, p, _ in trainer.snapshots)
    assert not list(tmp_path.glob(".*.tmp"))
    # each holds the version staged at its mark (external review F02), the budget's included
    marks = {m["tag"]: m["staged"] for m in summary_marks}
    assert all(Path(p).name == f"{t}-v{marks[t]}" for t, p, _ in trainer.snapshots)


def test_canary_update_breaks_the_relabelling_symmetry():
    """With the vocabulary permutation alone, the new model is the old one relabelled: on a context permuted the same
    way it predicts the permuted token. The norm sign flip breaks that, so old and new disagree even when copying."""
    from halfwritten.trainer.model import load_model
    from halfwritten.trainer.sync import canary_update, permute_vocab
    old = load_model("tiny-qwen3", "cpu", seed=0).eval()
    params = {k: v.detach().clone() for k, v in old.named_parameters()}
    ids = torch.randint(0, 512, (1, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        relabelled = load_model("tiny-qwen3", "cpu", seed=0).eval()
        relabelled.load_state_dict(permute_vocab(params, seed=7), strict=False)
        updated = load_model("tiny-qwen3", "cpu", seed=0).eval()
        updated.load_state_dict(canary_update(params, seed=7), strict=False)
        perm = permute_vocab({"model.embed_tokens.weight": torch.arange(512.0)[:, None]}, seed=7)[
            "model.embed_tokens.weight"][:, 0].long()  # new row t = old row perm[t]
        inv = torch.argsort(perm)
        # relabelled model on context x == old model on context perm[x], relabelled back
        a = relabelled(ids).logits.argmax(-1)
        b = inv[old(perm[ids]).logits.argmax(-1)]
        assert torch.equal(a, b)
        c = updated(ids).logits.argmax(-1)
        assert not torch.equal(c, b)
    assert torch.equal(canary_update(params, 7)["model.norm.weight"].abs(), params["model.norm.weight"].abs())


@pytest.mark.parametrize("regime", ["D+lora", "F"])
def test_end_to_end_lora_adapters_through_the_stack(regime, tmp_path):
    """The worker trains a LoRA adapter on the tiny Qwen3 and writes each staged version as a real PEFT adapter
    (safetensors + config) under a fresh directory; the controller loads it in place (D+lora) or as a new name (F)."""
    async def main():
        fake = FakeEngine(lengths=LengthModel(mean=20, sigma=0.5), cost=CostModel(5e-3, 1e-5, 0, 1e-6), vocab=512)
        await fake.start()
        runner, url = await serve(fake)
        trainer = ProcTrainerClient(
            {"model": "tiny-qwen3", "device": "cpu", "lr": 1e-3, "micro_tokens": 2048, "staging_dtype": "float32",
             "lora_rank": 2},
            {"kind": "adapter", "root": str(tmp_path / "adapters")}, workdir=str(tmp_path / "work"),
            env={**os.environ, "PYTHONPATH": str(ROOT)})
        cfg = RunConfig(regime=regime.removesuffix("+lora"), lora=True, B=2, G=4, M=2, K=2, max_tokens=64,
                        budget_s=10.0, timeout_s=120.0)
        engine = VllmHttpEngine(url, "fake", max_num_seqs=64, max_model_len=4096)
        controller = Controller(cfg, engine, trainer, synthetic(64, 8, vocab=512), sim_reward())
        try:
            summary = await controller.run()
        finally:
            await runner.cleanup()
        return summary, compare(controller.ledger, fake.truth), fake

    summary, report, fake = asyncio.run(main())
    log = (tmp_path / "work" / "trainer.log").read_text()[-2000:]
    assert summary.valid, (summary.error, summary.violations[:3], log)
    assert summary.steps >= 2 and report["ok"], report
    written = sorted((tmp_path / "adapters").glob("v*"))
    assert written and all((d / "adapter_model.safetensors").exists() and (d / "adapter_config.json").exists()
                           for d in written)
    import json as _json
    assert _json.loads((written[-1] / "adapter_config.json").read_text())["r"] == 2
    if regime == "F":
        assert len(fake.adapters) <= 4 and all(name.startswith("policy-v") for name in fake.adapters)


def test_one_head_pass_per_micro_batch_equals_per_sample_logprobs():
    """The micro-batch's answer tokens go through the head together; each sample's log-probs must be what a pass of
    its own gives."""
    from halfwritten.trainer.logprobs import token_logprobs
    tr = PolicyTrainer(PolicyTrainerSettings(model="tiny-qwen3", device="cpu", micro_tokens=4096,
                                             staging_dtype="float32"))
    batch = _batch(n=6)
    with torch.no_grad():
        together = tr.logprobs(batch.samples, list(range(6)))
        for r, s in enumerate(batch.samples):
            ids = torch.tensor([list(s.prompt_ids) + list(s.token_ids)])
            hidden = tr._decoder()(input_ids=ids).last_hidden_state[0]
            p, n = len(s.prompt_ids), len(s.token_ids)
            alone = token_logprobs(hidden[p - 1:p - 1 + n], tr._head(), torch.tensor(s.token_ids))
            assert together[r].shape == (n,) and torch.allclose(together[r], alone, atol=1e-5)


def test_adapter_sink_writes_atomically_and_keeps_the_newest(tmp_path):
    def write(path):
        (path / "adapter_model.safetensors").write_bytes(b"x" * 10)

    sink = AdapterSink(tmp_path / "adapters", write, keep=2)
    for v in range(4):
        pushed = sink.push(v)
    assert pushed.adapter_path.endswith("/v3") and pushed.bytes == 10
    assert sorted(p.name for p in (tmp_path / "adapters").iterdir()) == ["v2", "v3"]


def test_an_adapter_snapshot_carries_its_config_and_base(tmp_path):
    """External review F05: an adapter snapshot held only the LoRA tensors, so nothing could load it as an adapter or
    tell which base it belongs to. It now carries adapter_config.json, the weights under PEFT's file name, and its
    base (the model and the warm start)."""
    import json
    from types import SimpleNamespace

    from halfwritten.trainer.worker import Worker
    w = object.__new__(Worker)
    w.trainer = PolicyTrainer(PolicyTrainerSettings(model="tiny-qwen3", device="cpu", lora_rank=2,
                                                    staging_dtype="float32"))
    sent = []
    w.chan = SimpleNamespace(send=lambda *m: sent.append(m))
    w.trainer.stage(3)
    tensors, handle = w.trainer.staging.pin(3)
    w._snapshot("t01", str(tmp_path), 3, tensors, handle)
    snap = tmp_path / "t01-v3"
    assert json.loads((snap / "adapter_config.json").read_text())["r"] == 2
    assert (snap / "adapter_model.safetensors").resolve() == (snap / "model.safetensors").resolve()
    assert json.loads((snap / "base.json").read_text()) == {"model": "tiny-qwen3", "init_from": None}
    assert sent[-1][0] == "snapshot_done"
