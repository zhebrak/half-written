"""A job holding the host's GPU lock does not wait for the jobs queued behind it, and a killed job takes its engine
down with it."""
import signal
import subprocess
import sys
import time
from pathlib import Path

import halfwritten.gpu as gpu

ROOT = Path(__file__).resolve().parents[1]


def test_lock_holder_ignores_harnesses_waiting_on_its_lock(monkeypatch):
    monkeypatch.setattr(gpu, "compute_apps", lambda: [])
    monkeypatch.setattr(gpu, "utilization", lambda: 0.0)
    monkeypatch.setattr(gpu, "foreign_harnesses", lambda: ["python -m other.grid ... --gpu-lock /var/lock/gpu0"])
    monkeypatch.setattr(gpu.time, "sleep", lambda s: None)
    monkeypatch.delenv("HW_GPU_LOCK_HELD", raising=False)
    assert gpu.wait_for_gpu(max_wait_s=0.05, log=lambda *_: None) is False  # without the lock: busy
    clock = iter(range(0, 10_000))
    monkeypatch.setattr(gpu.time, "monotonic", lambda: float(next(clock)))
    monkeypatch.setenv("HW_GPU_LOCK_HELD", "1")
    assert gpu.wait_for_gpu(max_wait_s=600, log=lambda *_: None) is True  # with it: only the GPU counts


def test_a_killed_run_takes_its_engine_down():
    """SIGTERM (a timeout) mid-run: the engine's process group is gone when the job has exited."""
    port = 8197
    script = f"""
import sys, time
from pathlib import Path
from halfwritten.config import load, with_overrides
from halfwritten.launch import engine_process
cfg = with_overrides(load("configs/cpu/smoke.yaml"), ["launch.port={port}"])
with engine_process(cfg, Path(sys.argv[1])) as (url, served):
    print("ready", flush=True)
    time.sleep(120)
"""
    import tempfile
    logs = tempfile.mkdtemp(prefix="hw-eng-")
    p = subprocess.Popen([sys.executable, "-c", script, logs], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "ready"
        pattern = f"fake_server --port {port}"
        assert subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0
        p.send_signal(signal.SIGTERM)
        assert p.wait(30) == 128 + signal.SIGTERM
        for _ in range(50):
            if subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode != 0:
                break
            time.sleep(0.1)
        assert subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode != 0
    finally:
        if p.poll() is None:
            p.kill()
