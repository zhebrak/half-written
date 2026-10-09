"""The trainer worker: one process on the trainer's GPU, driven by the controller over a unix socket (wire.py).

    python -m halfwritten.trainer.worker --socket PATH

Threads:
    main      ("batch", TrainBatch) -> PolicyTrainer.step -> stage the new weights -> ("weights_ready", v, stats).
              It never waits for the engine (staged publication); with publication "lockstep" it waits for its own
              push before the next batch, the systems map's sensitivity row.
    sender    ("grant", v) -> push the latest staged version >= v through the sink -> ("pushed", Pushed). The pushed
              version can be newer than the grant when the trainer staged again meanwhile: the newest wins, and the
              controller labels tokens with the version actually pushed.
    reader    socket -> the queues above.
    snapshot  ("snapshot", tag, dir): copies the latest staged version to CPU the moment the mark arrives (the same
              rule in every arm: a busy trainer does not get to finish its step first) and writes it as
              <dir>/<tag>-v<version>/model.safetensors via a temporary directory renamed into place. The worker joins
              it before saying goodbye, so no write is cut short.

Every state change is recorded on the "trainer" and "sender" lanes with time.monotonic(), the clock the controller's
event loop reads on the same host, and the events ride along with "bye" so the run's trace has both processes.
An exception in any thread is sent as ("error", traceback) and ends the process.
"""
import argparse
import queue
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

from ..trace import Tracer
from ..types import WeightsReady, snapshot_name
from .wire import Channel


class Worker:
    def __init__(self, chan: Channel, hello: dict):
        from .model import PolicyTrainer, PolicyTrainerSettings
        from .sync import make_sink
        self.chan = chan
        self.trainer = PolicyTrainer(PolicyTrainerSettings(**hello["settings"]))
        self.sink = make_sink(hello["sink"], peft_config=self.trainer.peft_config())
        self.publication = hello.get("publication", "staged")
        self.tracer = Tracer(time.monotonic, "trainer")
        self.batches: queue.Queue = queue.Queue()
        self.grants: queue.Queue = queue.Queue()
        self.snapshots: queue.Queue = queue.Queue()
        self.pushed = -1
        self._pushed_cond = threading.Condition()
        self.stopping = threading.Event()
        self.error: str | None = None

    def run(self) -> int:
        self.tracer.state("trainer", "stage")
        self.trainer.stage(0)
        self.tracer.state("trainer", "idle")
        self.tracer.state("sender", "idle")
        self.chan.send("ready", 0)
        threads = [threading.Thread(target=self._guard, args=(f,), name=f.__name__, daemon=True)
                   for f in (self._read, self._send_loop)]
        snapper = threading.Thread(target=self._guard, args=(self._snapshot_loop,), name="snapshots", daemon=True)
        for t in [*threads, snapper]:
            t.start()
        self._guard(self._train_loop)
        self.snapshots.put(None)
        snapper.join(timeout=600)  # every snapshot already asked for is on disk before "bye"
        if self.error:
            self.chan.send("error", self.error)
            return 1
        self.chan.send("bye", {"steps": self.trainer.steps, "pushed": self.pushed, "events": self.tracer.close()})
        return 0

    def _guard(self, fn):
        try:
            fn()
        except Exception:
            if not self.stopping.is_set():
                self.error = traceback.format_exc()
                self.stopping.set()
                self.batches.put(None)
                self.grants.put(None)
                self.snapshots.put(None)

    def _read(self):
        while not self.stopping.is_set():
            msg = self.chan.recv()
            kind = msg[0]
            if kind == "batch":
                self.batches.put(msg[1])
            elif kind == "grant":
                self.grants.put(msg[1])
            elif kind == "snapshot":  # (tag, dir, version): pin that version now, copy it on the snapshot thread
                tag, directory, version = msg[1:]
                try:
                    tensors, handle = self.trainer.staging.pin(version)
                except Exception:
                    self.chan.send("snapshot_failed", tag, traceback.format_exc())
                    continue
                self.snapshots.put((tag, directory, version, tensors, handle))
            elif kind == "stop":
                self.stopping.set()
                self.batches.put(None)
                self.grants.put(None)
                return
            else:
                raise RuntimeError(f"unknown message {kind!r}")

    def _train_loop(self):
        while True:
            batch = self.batches.get()
            if batch is None:
                return
            if self.publication == "lockstep":
                self.tracer.state("trainer", "wait_push")
                with self._pushed_cond:
                    self._pushed_cond.wait_for(lambda: self.pushed >= self.trainer.steps or self.stopping.is_set())
            self.tracer.state("trainer", "fwd_bwd", step=batch.step)
            stats = self.trainer.step(batch)
            self.tracer.state("trainer", "stage")
            version = batch.step + 1
            self.trainer.stage(version, waiting=lambda w: self.tracer.state("trainer", "stage_wait" if w else "stage"))
            self.tracer.state("trainer", "idle")
            self.chan.send("weights_ready", WeightsReady(version, stats))

    def _send_loop(self):
        while True:
            want = self.grants.get()
            if want is None:
                return
            version, tensors, handle = self.trainer.staging.acquire(want, timeout=600)
            try:
                self.tracer.state("sender", "send", version=version)
                pushed = self.sink.push(version, tensors)
            finally:
                self.trainer.staging.release(handle)
                self.tracer.state("sender", "idle")
            with self._pushed_cond:
                self.pushed = version
                self._pushed_cond.notify_all()
            self.chan.send("pushed", pushed)

    def _snapshot_loop(self):
        while True:
            item = self.snapshots.get()
            if item is None:
                return
            try:
                self._snapshot(*item)
            except Exception:  # reported, not swallowed: the controller invalidates a run missing a mark
                self.chan.send("snapshot_failed", item[0], traceback.format_exc())

    def _snapshot(self, tag: str, directory: str, version: int, tensors: dict, handle: int):
        t0 = time.monotonic()
        try:
            cpu = {k: v.detach().to("cpu", copy=True) for k, v in tensors.items()}
        finally:
            self.trainer.staging.release(handle)
        from safetensors.torch import save_file
        final = Path(directory) / snapshot_name(tag, version)
        tmp = Path(directory) / f".{snapshot_name(tag, version)}.tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        save_file(cpu, str(tmp / "model.safetensors"))
        if (peft := self.trainer.peft_config()) is not None:  # an adapter: loadable as one, and tied to its base
            import json
            (tmp / "adapter_config.json").write_text(json.dumps(peft, default=list))
            (tmp / "adapter_model.safetensors").symlink_to("model.safetensors")
            (tmp / "base.json").write_text(json.dumps({"model": self.trainer.s.model,
                                                        "init_from": self.trainer.s.init_from}))
        tmp.rename(final)
        self.chan.send("snapshot_done", tag, str(final), time.monotonic() - t0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--socket", required=True)
    args = p.parse_args(argv)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(args.socket)
    chan = Channel(sock)
    kind, hello = chan.recv()
    if kind != "hello":
        raise RuntimeError(f"expected hello, got {kind!r}")
    try:
        worker = Worker(chan, hello)
    except Exception:
        chan.send("error", traceback.format_exc())
        return 1
    return worker.run()


if __name__ == "__main__":
    sys.exit(main())
