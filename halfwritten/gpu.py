"""Wait until the GPU is free before starting a timed row.

A row's wall clock is its result, and another process's kernels on the same GPU would show up in it. So before each
row the queue waits until the GPU has had no compute process, near-zero utilisation and no other harness process
alive for a quiet period. A harness that runs its engine in Docker containers leaves gaps between runs in which the
GPU looks free, so a live harness process or GPU container counts as busy too (HW_GPU_BUSY_PATTERNS, comma-separated,
extends the patterns). Hosts without nvidia-smi (a laptop) never wait.

On a host whose jobs take a lock file around each unit (flock), the job holding the lock sets HW_GPU_LOCK_HELD=1:
then only the GPU itself is checked, over a short quiet window (our previous step letting go of its memory), so the
jobs waiting on the lock are not counted as busy and nobody waits for anybody forever.
"""
import os
import shutil
import subprocess
import time

BUSY_PATTERNS = ["--gpus"]  # other GPU jobs count as busy by command-line pattern; HW_GPU_BUSY_PATTERNS extends the list


def local_device(gpu: str) -> str:
    """Map the launcher's physical GPU selector to this process's CUDA ordinal without changing its visibility."""
    gpu = str(gpu)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [v.strip() for v in visible.split(",")]
        if gpu not in devices:
            raise ValueError(f"GPU {gpu} is not in this process's CUDA_VISIBLE_DEVICES={visible!r}")
        return f"cuda:{devices.index(gpu)}"
    if gpu.isdecimal():
        return f"cuda:{int(gpu)}"
    import torch
    for i in range(torch.cuda.device_count()):
        if str(torch.cuda.get_device_properties(i).uuid) == gpu:
            return f"cuda:{i}"
    raise ValueError(f"no visible CUDA device with UUID {gpu}")


def foreign_harnesses() -> list[str]:
    """Command lines of other jobs that use, or are about to use, the GPU (by pattern; ours never match)."""
    patterns = BUSY_PATTERNS + [p for p in os.environ.get("HW_GPU_BUSY_PATTERNS", "").split(",") if p]
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()[1:]
    me = os.getpid()
    hits = []
    for line in out:
        pid, _, args = line.strip().partition(" ")
        if pid.isdigit() and int(pid) != me and "halfwritten" not in args and any(p in args for p in patterns):
            hits.append(args[:120])
    return hits


def compute_apps() -> list[dict]:
    """[{pid, name, memory_mib}] of processes using the GPU, per nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return []
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    apps = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            apps.append({"pid": int(parts[0]), "name": parts[1], "memory_mib": float(parts[2] or 0)})
    return apps


def utilization() -> float:
    if not shutil.which("nvidia-smi"):
        return 0.0
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    return max((float(x) for x in out), default=0.0)


def wait_for_gpu(max_wait_s: float = 12 * 3600, quiet_s: float = 120, poll_s: float = 10, log=print) -> bool:
    """Blocks until the GPU has been free (no compute process, utilisation <= 5%) for quiet_s; False on timeout."""
    held = bool(os.environ.get("HW_GPU_LOCK_HELD"))
    if held:
        quiet_s, poll_s = min(quiet_s, 10.0), min(poll_s, 2.0)
    deadline, quiet_since, said = time.monotonic() + max_wait_s, None, False
    while time.monotonic() < deadline:
        busy = compute_apps() or utilization() > 5 or (not held and foreign_harnesses())
        now = time.monotonic()
        if busy:
            quiet_since = None
            if not said:
                log(f"GPU busy ({compute_apps() or foreign_harnesses()[:2]}); waiting")
                said = True
        else:
            quiet_since = quiet_since or now
            if now - quiet_since >= quiet_s:
                return True
        time.sleep(poll_s)
    return False
