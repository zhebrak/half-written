import asyncio
import json

import pytest

from bench.evaluate import accuracy, evaluate_row
from halfwritten.engines.fake import CostModel, FakeEngine, LengthModel
from halfwritten.engines.fake_server import serve
from halfwritten.engines.vllm_http import VllmHttpEngine
from halfwritten.rewards import sim_reward
from halfwritten.tasks import synthetic


def test_evaluate_row_labels_snapshots_by_their_marks_and_reruns_add_nothing(tmp_path):
    row = tmp_path / "row"
    for tag in ("tend-v9", "t01-v4"):
        (row / "work" / tag).mkdir(parents=True)
        (row / "work" / tag / "model.safetensors").write_bytes(b"")
    (row / "work" / "w0").mkdir()
    (row / "snapshots.json").write_text(json.dumps({"marks": [{"tag": "t01", "t": 5.0, "steps": 4},
                                                              {"tag": "tend", "t": 40.0, "steps": 9}]}))
    stream = synthetic(16, 8, vocab=512)
    items = [(p, a) for _, p, a in (next(stream) for _ in range(12))]
    pushed = []

    async def main(init, items=items, max_tokens=32, split="test"):
        fake = FakeEngine(lengths=LengthModel(mean=10, sigma=0.5), cost=CostModel(1e-3, 0, 0, 0), vocab=512)
        await fake.start()
        runner, url = await serve(fake)
        engine = VllmHttpEngine(url, "fake", max_num_seqs=64, max_model_len=4096)

        def push(version, path):
            fake.apply_weights(version)  # idle between evaluations, as vLLM is
            pushed.append((version, path.name))

        try:
            return await evaluate_row(row, engine, push, items, sim_reward(), max_tokens=max_tokens, init=init,
                                      split=split, log=lambda *_: None)
        finally:
            await engine.close()
            await runner.cleanup()

    out = asyncio.run(main(row / "work" / "w0"))
    assert [(r["snapshot"], r["t"]) for r in out] == [("init", 0.0), ("t01-v4", 5.0), ("tend-v9", 40.0)]
    assert pushed == [(1, "w0"), (2, "t01-v4"), (3, "tend-v9")]  # the warm start is what "init" evaluates
    assert all(0 <= r["accuracy"] <= 1 and r["n"] == 12 for r in out)
    again = asyncio.run(main(row / "work" / "w0"))
    assert len(again) == 3 and len(pushed) == 3  # nothing re-evaluated
    assert len((row / "evals.jsonl").read_text().splitlines()) == 3
    # external review F03: the same number of other items, or another max_tokens, is another measurement
    swapped = [(p, f"other-{a}") for p, a in items]
    asyncio.run(main(row / "work" / "w0", items=swapped))
    assert len(pushed) == 6
    asyncio.run(main(row / "work" / "w0", items=swapped, max_tokens=16))
    assert len(pushed) == 9
    # another split is kept beside the first, not over it
    out = asyncio.run(main(row / "work" / "w0", items=swapped, max_tokens=16, split="dev"))
    assert len(pushed) == 12 and {r["split"] for r in out} == {"dev", "test"}
    assert len((row / "evals.jsonl").read_text().splitlines()) == 6


def test_accuracy_is_greedy_and_concurrent():
    async def main():
        fake = FakeEngine(lengths=LengthModel(mean=5, sigma=0.0), cost=CostModel(1e-3, 0, 0, 0), vocab=512)
        await fake.start()
        runner, url = await serve(fake)
        engine = VllmHttpEngine(url, "fake", max_num_seqs=64, max_model_len=4096)
        try:
            return await accuracy(engine, [((1, 2), i) for i in range(20)], lambda a, t, f: float(len(t) == 5),
                                  max_tokens=32, concurrency=8)
        finally:
            await engine.close()
            await runner.cleanup()

    r = asyncio.run(main())
    assert r["accuracy"] == 1.0 and r["n"] == 20 and r["mean_tokens"] == 5


def test_pass_at_k_and_the_keep_rule():
    from bench.prompt_filter import keep, pass_at_k

    async def main():
        fake = FakeEngine(lengths=LengthModel(mean=6, sigma=0.8), cost=CostModel(1e-3, 0, 0, 0), vocab=512)
        await fake.start()
        runner, url = await serve(fake)
        engine = VllmHttpEngine(url, "fake", max_num_seqs=64, max_model_len=4096)
        try:
            # right when the answer is long: some prompts always, some never, some sometimes
            return await pass_at_k(engine, [((1, 2), i) for i in range(10)],
                                   lambda a, t, f: float(len(t) >= 6), k=8, max_tokens=32, lengths=lengths)
        finally:
            await engine.close()
            await runner.cleanup()

    lengths = []
    rates = asyncio.run(main())
    assert len(rates) == 10 and all(r in {i / 8 for i in range(9)} for r in rates)
    assert len(lengths) == 80 and all(0 < n <= 32 for n in lengths)  # every sampled answer, for the systems map
    assert keep([0.0, 0.125, 0.5, 0.875, 1.0], 8) == [1, 2, 3]


def test_a_filter_restricts_the_task_to_the_kept_prompts(tmp_path):
    from halfwritten import tasks
    from halfwritten.config import RunConfig
    path = tmp_path / "kept.json"
    path.write_text(json.dumps({"kept": [2, 0]}))
    rows = [("a", 1), ("b", 2), ("c", 3)]
    assert tasks.training_rows(RunConfig(prompt_filter=str(path)), rows) == [("c", 3), ("a", 1)]
    assert tasks.training_rows(RunConfig(), rows[:1]) == rows[:1]


def test_the_development_split_is_held_out_of_every_pool(tmp_path, monkeypatch):
    """Decisions are read on dev, so no run may train on it, filtered or not (a lens: unfiltered runs, such as MATH's
    sync curve, would have trained on every dev item)."""
    from halfwritten import tasks
    from halfwritten.config import RunConfig
    monkeypatch.setattr(tasks, "dev_questions", lambda task: frozenset({"b"}))
    rows = [("a", 1), ("b", 2), ("c", 3)]
    path = tmp_path / "kept.json"
    path.write_text(json.dumps({"kept": [1, 2]}))
    assert tasks.training_rows(RunConfig(task="gsm8k"), rows) == [("a", 1), ("c", 3)]
    assert tasks.training_rows(RunConfig(task="gsm8k", prompt_filter=str(path)), rows) == [("c", 3)]


def test_a_trainer_starts_from_the_warm_start(tmp_path):
    import pytest
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from halfwritten.trainer.model import PolicyTrainer, PolicyTrainerSettings
    a = PolicyTrainer(PolicyTrainerSettings(model="tiny-qwen3", device="cpu", seed=1, gradient_checkpointing=False))
    save_file({k: v.detach().clone() for k, v in a.model.named_parameters()}, str(tmp_path / "model.safetensors"))
    b = PolicyTrainer(PolicyTrainerSettings(model="tiny-qwen3", device="cpu", seed=2, init_from=str(tmp_path)))
    for (k, va), (_, vb) in zip(a.model.named_parameters(), b.model.named_parameters()):
        assert torch.equal(va, vb), k


def test_calibration_recovers_the_fake_engines_cost_model():
    """In virtual time against FakeEngine itself: the fit must predict its decode and prefill steps closely."""
    from bench.calibrate import fit, measure
    from halfwritten.clock import run_sim
    true = CostModel(alpha=4e-3, beta=1e-4, gamma=2e-8, delta=2e-6)

    async def main():
        fake = FakeEngine(lengths=LengthModel(mean=5, sigma=0.0), cost=true, vocab=1000)
        await fake.start()
        try:
            return await measure(fake, widths=(1, 16, 64, 128), decode_len=48, prompt_len=16,
                                 prefill_lens=(1024, 2048), prefill_widths=(1, 2))
        finally:
            await fake.close()

    got = fit(run_sim(main()))
    for w, ctx, pre in ((64, 64 * 40, 0), (128, 128 * 40, 0), (2, 4096, 4094)):
        want = true.step_time(w, ctx, pre)
        c = got["cost"]
        assert abs(c["alpha"] + c["beta"] * w + c["gamma"] * ctx + c["delta"] * pre - want) / want < 0.15
    assert got["rel_error_median"] < 0.15


def test_noise_floor_sampling_and_statistics():
    import numpy as np

    from bench.noise_floor import mismatch_stats, sample
    calls = []

    async def main():
        fake = FakeEngine(lengths=LengthModel(mean=40, sigma=0.0), cost=CostModel(1e-3, 0, 0, 0), vocab=512)
        await fake.start()
        runner, url = await serve(fake)
        engine = VllmHttpEngine(url, "fake", max_num_seqs=64, max_model_len=4096)

        async def swap(answers):
            await fake.pause("keep", False)
            for a in answers:
                a[3] = len(a[1])
            calls.append(min(a[3] for a in answers))
            await fake.resume()

        try:
            return await sample(engine, [((1, 2), None)] * 4, max_tokens=40, stop=(), temperature=1.0, swap=swap,
                                swap_after=8)
        finally:
            await engine.close()
            await runner.cleanup()

    answers = asyncio.run(main())
    assert calls and calls[0] >= 8 and all(len(a[1]) == 40 and len(a[2]) == 40 and a[3] >= 8 for a in answers)
    s = mismatch_stats(np.array([0.0, 0.1, -0.1]))
    assert s["tokens"] == 3 and s["k3"] > 0 and 0 < s["ess"] <= 1
    assert mismatch_stats(np.zeros(0))["tokens"] == 0


def test_noise_floor_swap_waits_for_finished_answers_too():
    """An answer that ends before the swap point must not hold the swap (and the shared GPU) forever."""
    from bench.noise_floor import sample

    async def main():
        fake = FakeEngine(lengths=LengthModel(mean=40, sigma=0.0), cost=CostModel(1e-3, 0, 0, 0), vocab=512)
        await fake.start()
        runner, url = await serve(fake)
        engine = VllmHttpEngine(url, "fake", max_num_seqs=64, max_model_len=4096)
        swapped = []

        async def swap(answers):
            swapped.append([len(a[1]) for a in answers])

        try:  # one answer is capped at 5 tokens, below the swap point of 8
            items = [((1, 2), None)] * 3
            return await sample(engine, items, max_tokens=40, stop=(), temperature=1.0, swap=swap, swap_after=8,
                                timeout_s=30), swapped
        finally:
            await engine.close()
            await runner.cleanup()

    import halfwritten.engines.fake as fk
    orig = fk.FakeEngine.generate

    async def capped(self, req):  # the first request ends early, as an answer hitting a stop would
        if req.rid == "nf-0#0":
            from dataclasses import replace
            req = replace(req, max_tokens=5)
        async for c in orig(self, req):
            yield c

    fk.FakeEngine.generate = capped
    try:
        answers, swapped = asyncio.run(main())
    finally:
        fk.FakeEngine.generate = orig
    assert swapped and len(answers[0][1]) == 5 and all(len(a[1]) == 40 for a in answers[1:])


def test_calibration_requests_fit_the_engines_context():
    """vLLM rejects a prompt that leaves no room for its output: the H100's first calibration died on 2,048 + 1."""
    from bench.calibrate import measure
    from halfwritten.types import Chunk
    seen = []

    class Recorder:
        async def generate(self, req):
            seen.append(len(req.prompt_ids) + req.max_tokens)
            yield Chunk((1,), (-0.1,), "length")

    pts = asyncio.run(measure(Recorder(), widths=(1, 2), decode_len=4, prompt_len=8, max_model_len=1024))
    assert max(seen) <= 1024 and {p["kind"] for p in pts} == {"decode", "prefill"}


def test_calibration_uses_the_engines_own_step_times_when_it_has_them():
    """Client wall time adds HTTP and streaming costs: the fit must use the server's inter-token latency and prefill
    time, and a wide prefill point counts the steps the token budget forces."""
    from bench.calibrate import measure
    from halfwritten.types import Chunk
    state = {"vllm:inter_token_latency_seconds_sum": 0.0, "vllm:inter_token_latency_seconds_count": 0,
             "vllm:request_prefill_time_seconds_sum": 0.0, "vllm:request_prefill_time_seconds_count": 0}

    class Engine:
        async def generate(self, req):
            if req.max_tokens > 1:  # decode: every step 4 ms on the server
                state["vllm:inter_token_latency_seconds_sum"] += 0.004 * (req.max_tokens - 1)
                state["vllm:inter_token_latency_seconds_count"] += req.max_tokens - 1
            else:
                state["vllm:request_prefill_time_seconds_sum"] += 0.010
                state["vllm:request_prefill_time_seconds_count"] += 1
            yield Chunk((1,), (-0.1,), "length")

    async def metrics():
        return dict(state)

    pts = asyncio.run(measure(Engine(), widths=(1, 4), decode_len=8, prompt_len=8, prefill_lens=(4096,),
                              prefill_widths=(4,), max_model_len=8192, max_num_batched_tokens=8192,
                              server_metrics=metrics))
    dec = [p for p in pts if p["kind"] == "decode"]
    assert all(abs(p["seconds"] - 0.004) < 1e-12 for p in dec)
    pre = [p for p in pts if p["kind"] == "prefill"][0]
    assert pre["width"] == 2 and abs(pre["seconds"] - 0.010) < 1e-12  # 4 x 4,096 tokens: 2 per 8,192-token step


@pytest.mark.parametrize("regime,K", [("A", 0), ("E", 4), ("F", 2)])
def test_every_rows_config_becomes_a_valid_evaluation_launch(regime, K):
    """bench.evaluate forced regime D, which rejected every A row (K = 0): the pilot's baseline was never scored."""
    from bench.evaluate import eval_config
    from halfwritten.config import RunConfig, with_overrides
    cfg = with_overrides(RunConfig(model="m", regime=regime, K=K, lora=regime == "F"),
                         ["launch.head_fp32=true", "launch.engine_gpu=0", "launch.trainer_gpu=1"])
    ev = eval_config(cfg, ["launch.port=8150"]).validate()
    assert ev.regime == "A" and not ev.lora and ev.launch.weight_backend == "ipc" and ev.launch.trainer_gpu == "0"
    assert ev.launch.head_fp32  # the row's own LM head


def test_adapter_snapshots_are_evaluated_as_adapters(tmp_path):
    """External review F05 (lens: no bf16 merge): a LoRA row's snapshots are adapters. Each is loaded under a new name
    (never in place) on an engine that already serves the row's base, the previous one unloaded, and evaluated under
    its name; init is the base itself."""
    from bench.evaluate import adapter_push
    row = tmp_path / "row"
    for tag in ("t01-v4", "tend-v9"):
        (row / "work" / tag).mkdir(parents=True)
        (row / "work" / tag / "model.safetensors").write_bytes(b"")
    (row / "snapshots.json").write_text(json.dumps({"marks": [{"tag": "t01", "t": 5.0, "steps": 4},
                                                              {"tag": "tend", "t": 40.0, "steps": 9}]}))
    stream = synthetic(16, 8, vocab=512)
    items = [(p, a) for _, p, a in (next(stream) for _ in range(6))]

    async def main():
        fake = FakeEngine(lengths=LengthModel(mean=10, sigma=0.5), cost=CostModel(1e-3, 0, 0, 0), vocab=512)
        await fake.start()
        runner, url = await serve(fake)
        engine = VllmHttpEngine(url, "fake", max_num_seqs=64, max_model_len=4096)
        try:
            out = await evaluate_row(row, engine, adapter_push(engine), items, sim_reward(), max_tokens=32,
                                     init=None, log=lambda *_: None)
            return out, dict(fake.adapters)
        finally:
            await engine.close()
            await runner.cleanup()

    out, adapters = asyncio.run(main())
    assert [r["snapshot"] for r in out] == ["init", "t01-v4", "tend-v9"] and all(r["n"] == 6 for r in out)
    assert list(adapters) == ["snapshot-2"]  # the last one loaded; the first was unloaded after it


def test_the_fingerprint_is_pinned():
    """What an evaluation record measured, as a literal: a refactor that changes the fingerprint would re-evaluate
    every archived snapshot (review-2026-10-02 plan, section 1.0)."""
    from unittest import mock

    import bench.evaluate as ev
    with mock.patch.object(ev, "scorer_versions", lambda: {"scorer": "test", "math_verify": "0.9.0"}):
        assert ev.fingerprint("dev", "abc", 64, ("</s>",), None) == \
            "8636dae37ea2b0ce1e074a7966594e2a38a7e8349230521c2cb1bb3da8b45d46"


def test_pruning_keeps_the_budgets_snapshot_and_waits_for_every_mark(tmp_path):
    from bench.evaluate import prune_snapshots, snapshot_dirs
    from bench.stats import mark_problem
    row = tmp_path / "row"
    for tag in ("tend-v9", "t01-v4"):
        (row / "work" / tag).mkdir(parents=True)
        (row / "work" / tag / "model.safetensors").write_bytes(b"w" * 32)
    (row / "snapshots.json").write_text(json.dumps({
        "marks": [{"tag": "t01", "staged": 4}, {"tag": "tend", "staged": 9}],
        "written": [["t01", "work/t01-v4", 1.0], ["tend", "work/tend-v9", 1.0]]}))
    done = [{"tag": "init", "snapshot": "init", "split": "dev", "accuracy": 0.5},
            {"tag": "tend", "snapshot": "tend-v9", "split": "dev", "accuracy": 0.6}]
    (row / "evals.jsonl").write_text("".join(json.dumps(e) + "\n" for e in done))
    assert prune_snapshots(row, "dev") == []  # t01 is not evaluated yet: nothing goes
    done.insert(1, {"tag": "t01", "snapshot": "t01-v4", "split": "dev", "accuracy": 0.55})
    (row / "evals.jsonl").write_text("".join(json.dumps(e) + "\n" for e in done))
    assert prune_snapshots(row, "test") == []  # evaluated on dev, not on test
    pruned = prune_snapshots(row, "dev")
    assert [p["name"] for p in pruned] == ["t01-v4"] and pruned[0]["bytes"] == 32
    assert [d.name for d in snapshot_dirs(row)] == ["tend-v9"]
    assert mark_problem(row, done) is None  # the curve's record is complete without the weights
    assert prune_snapshots(row, "dev") == []  # nothing left to prune
