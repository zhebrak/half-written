"""The small modules nothing tested directly (review-2026-10-02 plan, section 2.10): deterministic hashes, the
pinned revisions, and the controller-worker wire protocol."""
import asyncio
import socket
import subprocess
import sys
import threading


def test_stable_keys_are_the_same_in_every_process_and_mix_spreads():
    from halfwritten.hashing import mix, stable_key
    here = stable_key("row", 3, 0.5)
    code = "from halfwritten.hashing import stable_key; print(stable_key('row', 3, 0.5))"
    there = int(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout)
    assert here == there and stable_key("row", 3, 0.5) != stable_key("row", 3, 0.6)
    assert 0 <= here < 1 << 64
    draws = {mix(i) & 0xFF for i in range(2000)}
    assert len(draws) == 256 and mix(1) != mix(2)


def test_every_task_reads_pinned_revisions():
    from halfwritten.revisions import DATASETS, MODELS, TASK_DATASETS, model_revision, task_revisions
    qwen = "Qwen/Qwen3-0.6B-Base"
    assert model_revision(qwen) == MODELS[qwen] and len(MODELS[qwen]) == 40
    assert model_revision("tiny-qwen3") is None  # a local or test model loads as it is
    for task, repos in TASK_DATASETS.items():
        assert task_revisions(task) == {r: DATASETS[r] for r in repos}
        assert all(len(v) == 40 for v in task_revisions(task).values())
    assert task_revisions("sim") == {}
    assert set(task_revisions("gsm8k")) != set(task_revisions("math"))  # re-pinning one task never touches the other


def test_the_wire_carries_frames_both_ways_in_order():
    """A blocking Channel (the worker's end) against the asyncio end (the controller's), over a socketpair: frames
    of any size, any pickle-able payload, in order, and a closed peer ends a receive with EOFError."""
    from halfwritten.trainer.wire import Channel, arecv, asend
    a, b = socket.socketpair()
    worker = Channel(b)
    big = {"tensor": list(range(50_000)), "note": "x" * 10_000}
    received = []

    def worker_side():
        for i in range(3):
            worker.send("weights_ready", i, big)
        received.append(worker.recv())
        received.append(worker.recv())
        b.close()

    async def controller_side():
        reader, writer = await asyncio.open_connection(sock=a)
        got = [await arecv(reader) for _ in range(3)]
        await asend(writer, "grant", 1)
        await asend(writer, "stop")
        writer.close()
        return got

    t = threading.Thread(target=worker_side)
    t.start()
    got = asyncio.run(controller_side())
    t.join(5)
    assert [g[:2] for g in got] == [("weights_ready", i) for i in range(3)] and got[2][2] == big
    assert received == [("grant", 1), ("stop",)]
    c, d = socket.socketpair()
    d.close()
    try:
        Channel(c).recv()
        raise AssertionError("a closed peer must end the receive")
    except EOFError:
        pass
