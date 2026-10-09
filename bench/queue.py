"""The run queue for an unattended GPU session (PLAN.md §5.5).

    python -m bench.queue run MATRIX.yaml [--out results] [--dry-run]

A matrix names a base config, the arms, the seeds and overrides:

    base: configs/one_gpu/gsm8k_smoke.yaml
    arms: [A, B, C1, D, E]
    seeds: [0, 1]                    # every arm; or per arm: {A: [0, 1, 2], C1: [0, 1], ...}
    set: [budget_s=300]              # every row
    per_arm: {E: [snapshot_every_s=30]}     # some rows
    canary: true                     # optional: run bench.canary first (the session gate); a failure stops the queue
    evaluate: true                   # optional: evaluate each row's snapshots right after it (bench/evaluate.py)
    prune: true                      # optional: after a complete evaluation, delete the snapshots' weights but tend's
    amendment: configs/amendments/gsm8k.yaml   # optional: T, T/16 snapshots and the seeds per arm come from it
    requires_amendment: true         # a confirmatory grid: refuses to plan without the amendment file
    seeds: amendment                 # ... and takes its seeds (n per arm -> 0..n-1)
    variants:                        # optional: each variant x arm x seed is a row, tagged with the variant's name
      gsm8k-lengths: ["forced_lengths=parquet:prep/gsm8k/w0-lengths.parquet"]   # quote: specs contain commas
      lognormal-1.1: ["forced_lengths=lognormal:300,1.1"]

`plan` orders it in seed blocks: every arm with seed s runs before any arm of seed s+1, in an order shuffled per block,
so drift over a session hits all arms alike and a session cut short still leaves complete blocks. `run` executes it:
each row in a fresh `python -m halfwritten run` process (run_command: ended as a supervised step is, TERM then KILL, with
no engine left on the GPU), after waiting for the GPU to be free of other jobs (halfwritten/gpu.py). A row's status
is its metrics.json (valid: results.row_problem); the exit code only says the process crashed. A finished, valid row
of the same measurement is skipped (resumable: launch.finished); a crash is retried once, then recorded as
<out>/_queue/failed-<arm>-s<seed>.json with the log's tail, and the queue moves on. An invalid row is moved aside
to <row>.invalid-<time> and run once more, once across sessions (HYPOTHESES.md decision 7); a second one stands
and is reported, never evaluated. A valid row whose harness record trips decision 7's limits (results.harness_problem)
is kept and reported with its reason: the analysis applies the rule, the queue only records it (the owner's policy,
2026-10-02). The command's exit status is 1 if any row failed or the gate or the GPU stopped the queue, 3 if every
row ran but some stand invalid (the chain notes it and goes on), else 0. Arm A runs with K=0, S=1.
"""
import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from halfwritten.hashing import stable_key
from halfwritten.regimes import parse_arm
from halfwritten.results import Row, rerun_pending

GRACE_S = 30.0  # after a TERM: the launcher's own grace (GRACE_S) plus its engine's signalled wait, then a KILL
ROW_MARGIN_S = 20 * 60  # beyond a row's budget: engine start, warm-up, the row's write and the engine's teardown
EVALUATE_S = 2 * 3600  # a row's snapshots, evaluated: MATH's 3,000-token answers on 500 items, 17 times, fit in it
CANARY_S = 30 * 60
QUIET_S = 10.0  # the GPU's quiet window before a row on an exclusive box (gpu.py's default, 120 s, is for a box with other jobs:
# one: over the study's ~150 rows and canaries that default alone would have cost about 5 box-hours)
FAILED = ("failed", "evaluation failed", "canary failed", "gpu never free")
STANDS = ("invalid", "invalid (stands)")  # exit 3: ran, rerun, invalid again; reported, left to the analysis
RC_ROWS_STAND = 3


def plan(arms: list[str], seeds: list[int] | dict[str, list[int]], session_seed: int = 0) -> list[tuple[str, int]]:
    """(arm, seed) in seed blocks. `seeds` is one list for every arm, or a list per arm (an arm with fewer seeds, such
    as the consistency check C1, sits out the later blocks)."""
    per_arm = seeds if isinstance(seeds, dict) else {arm: seeds for arm in arms}
    out = []
    for s in sorted({s for arm in arms for s in per_arm[arm]}):
        block = [arm for arm in arms if s in per_arm[arm]]
        random.Random(stable_key(session_seed, s)).shuffle(block)
        out += [(arm, s) for arm in block]
    return out


def cells(matrix: dict) -> list[tuple[str, str | None, int]]:
    """(arm, variant, seed) in seed blocks: every arm x variant of a seed, shuffled, before the next seed."""
    variants = list(matrix.get("variants", {})) or [None]
    units = [f"{v}|{a}" if v else a for v in variants for a in matrix["arms"]]
    seeds = matrix["seeds"]
    if isinstance(seeds, dict):  # a unit takes its arm's seeds
        seeds = {u: seeds[u.rpartition("|")[2]] for u in units}
    out = []
    for unit, seed in plan(units, seeds, matrix.get("session_seed", 0)):
        variant, _, arm = unit.rpartition("|")
        out.append((arm, variant or None, seed))
    return out


def row_overrides(matrix: dict, arm: str, seed: int, variant: str | None = None) -> list[str]:
    """Overrides for one row. An arm is a regime, optionally with +lora, or a regime with a suffix naming a variant of
    it (B-lockstep: per_arm gives its overrides; the regime is B)."""
    name, lora = parse_arm(arm)
    name = name.split("-")[0]
    lora = lora or name == "F"
    sets = list(matrix.get("set", [])) + list(matrix.get("per_arm", {}).get(arm, []))
    if variant:
        sets += list(matrix["variants"][variant]) + [f"tag={variant}"]
    sets += [f"regime={name}", f"seed={seed}"]
    if lora:
        sets.append("lora=true")
    if name == "A":
        sets += ["K=0", "S=1"]
    return sets


# what the session gate's canary actually varies with (bench/canary.py): rows that agree on these share one canary;
# another value is another certificate (external review F04: the systems map's M/B = 2 variant ran at a load nobody
# had certified, and only launch.* overrides reached the gate)
CERTIFIED = ("model", "max_model_len", "max_tokens", "launch.api_servers", "launch.stream_interval",
             "launch.async_scheduling", "launch.head_fp32")


def certificate(cfg) -> dict:
    """The settings a canary certifies, as --set overrides: CERTIFIED's values and the load (M x G answers)."""
    out = {k: getattr(cfg.launch, k.split(".", 1)[1]) if k.startswith("launch.") else getattr(cfg, k)
           for k in CERTIFIED}
    return out | {"M": cfg.M, "G": cfg.G}


def run_command(cmd: list[str], timeout_s: float, port: int | None = None,
                grace_s: float = GRACE_S) -> subprocess.CompletedProcess:
    """A subprocess run to completion, or ended as a supervisor ends a step: TERM (the launcher then stops its engine and
    writes its row), `grace_s`, then KILL, and any engine still listening on `port` killed too. subprocess.run's
    timeout would KILL the launcher at once, and its engine, in a session of its own, would survive holding the GPU
    while every later row waited for it (staff engineer's review, 2026-10-02). stdout and stderr are returned merged."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout_s)
        return subprocess.CompletedProcess(cmd, p.returncode, out or "", "")
    except subprocess.TimeoutExpired:
        pass
    _signal_group(p.pid, signal.SIGTERM)
    try:
        out, _ = p.communicate(timeout=grace_s)
    except subprocess.TimeoutExpired:
        _signal_group(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
    if port is not None:
        kill_engines_on(port)
    return subprocess.CompletedProcess(cmd, -1, f"timeout after {timeout_s:.0f} s\n{out or ''}", "")


def _signal_group(pid: int, sig) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def kill_engines_on(port: int, grace_s: float = 10.0) -> list[int]:
    """Kills the process group of every vLLM server started for `port` (its argv names it). /proc is read and the
    argv matched exactly, never a `pkill -f` pattern: only engines this harness started on this port."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            argv = Path(f"/proc/{entry}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if (b"--port" in argv and argv[argv.index(b"--port") + 1] == str(port).encode()
                and any(b"vllm" in a for a in argv[:3])):
            pids.append(int(entry))
    for pid in pids:
        _signal_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while pids and time.monotonic() < deadline and any(Path(f"/proc/{p}").exists() for p in pids):
        time.sleep(0.5)
    for pid in pids:
        if Path(f"/proc/{pid}").exists():
            _signal_group(pid, signal.SIGKILL)
    return pids


def _evaluate(path: Path, cfg, done: dict, *, prune: bool = False) -> None:
    from bench.evaluate import evaluation_problem
    for attempt in (1, 2):
        ev = run_command([sys.executable, "-m", "bench.evaluate", str(path)], timeout_s=EVALUATE_S,
                         port=cfg.launch.port)
        problem = evaluation_problem(path, cfg.eval_split, current=True) if ev.returncode == 0 \
            else f"exit {ev.returncode}"
        done["evaluated"] = problem is None
        if problem is None:
            if prune:
                from bench.evaluate import prune_snapshots
                done["pruned"] = len(prune_snapshots(path, cfg.eval_split))
            return
        with (path / "evaluate-failed.log").open("a") as f:
            f.write(f"attempt {attempt}: {problem}\n{ev.stdout[-8000:]}\n")
    done.update(status="evaluation failed", problem=problem)


def apply_amendment(matrix: dict, amendment: dict | None, matrix_path: str = "") -> dict:
    """The matrix with the amendment's T, snapshot interval and seeds applied (bench.amend wrote them under the
    pre-registered rules); a matrix that requires one refuses to plan without it, so no confirmatory grid runs on
    the base config's placeholders."""
    if matrix.get("requires_amendment") and amendment is None:
        raise ValueError(f"{matrix_path or 'this matrix'} runs only under an amendment "
                         f"({matrix.get('amendment', 'none named')}): the pilot has not set T and the seeds "
                         "(python -m bench.amend)")
    if amendment is None:
        if matrix.get("seeds") == "amendment":
            raise ValueError(f"{matrix_path}: seeds come from an amendment, and none was given")
        return matrix
    matrix = dict(matrix)
    matrix["set"] = [*matrix.get("set", []), f"budget_s={amendment['T_s']}",
                     f"snapshot_every_s={amendment['snapshot_every_s']}"]
    if matrix.get("seeds") == "amendment":
        matrix["seeds"] = {arm: list(range(int(amendment["seeds"][arm]))) for arm in matrix["arms"]
                           if arm in amendment["seeds"]}
        if missing := [arm for arm in matrix["arms"] if arm not in amendment["seeds"]]:
            raise ValueError(f"the amendment names no seeds for {missing}")
    return matrix


def run_matrix(matrix_path: str, out: str, dry_run: bool = False, timeout_s: float = 6 * 3600, log=print,
               variants: list[str] | None = None, gpu_wait_s: float | None = None,
               amendment: str | None = None, deadline_h: float | None = None) -> list[dict]:
    """Runs a matrix's rows; with `variants`, only those of its variants (the rest wait for their inputs). A row's
    process gets its budget plus ROW_MARGIN_S, capped by `timeout_s`; `gpu_wait_s` bounds each wait for a free GPU
    and shortens its quiet window to QUIET_S (the study chain passes minutes: an exclusive box, whose previous row
    just let go of its memory; the default waits hours over a two-minute window, for a box with other jobs). `amendment`
    overrides the matrix's own amendment path (a dry run against a planned one)."""
    from halfwritten.config import load, with_overrides
    from halfwritten.gpu import wait_for_gpu
    from halfwritten.launch import finished, row_dir_on_host
    from halfwritten.results import harness_problem, row_problem
    matrix = yaml.safe_load(Path(matrix_path).read_text())
    amend_path = amendment or matrix.get("amendment")
    amended = yaml.safe_load(Path(amend_path).read_text()) if amend_path and Path(amend_path).exists() else None
    matrix = apply_amendment(matrix, amended, matrix_path)
    if variants:
        if unknown := set(variants) - set(matrix.get("variants", {})):
            raise ValueError(f"{matrix_path} has no variants {sorted(unknown)}")
        matrix["variants"] = {k: v for k, v in matrix["variants"].items() if k in variants}
    base = matrix["base"]
    qdir = Path(out) / "_queue"
    qdir.mkdir(parents=True, exist_ok=True)
    gpu_free = (lambda: wait_for_gpu(log=log)) if gpu_wait_s is None else (
        lambda: wait_for_gpu(max_wait_s=gpu_wait_s, quiet_s=QUIET_S, log=log))
    stopped = lambda status: [{"status": status, "arm": None, "variant": None, "seed": None, "row": None}]  # noqa
    report = []
    planned = []
    for arm, variant, seed in cells(matrix):
        sets = row_overrides(matrix, arm, seed, variant)
        planned.append((arm, variant, seed, sets, with_overrides(load(base), sets)))
    gates: dict[str, dict] = {}
    if matrix.get("canary") and not dry_run:  # the session gate: no row runs on an engine whose regimes are unproven
        if not gpu_free():
            return stopped("gpu never free")
        # one canary per distinct certificate, all before any row (the rows then keep their shuffled order), each at
        # its rows' launch and under their load: every other answer they keep in flight streams alongside
        for cfg in {json.dumps(certificate(c), sort_keys=True): c for *_, c in planned}.values():
            cert = certificate(cfg)
            where = qdir / f"canary-{int(time.time())}-{len(gates)}"
            cmd = [sys.executable, "-m", "bench.canary", "--model", cfg.model, "--config", base,
                   "--background-answers", str(cfg.M * cfg.G - 1),
                   *sum((["--set", f"{k}={json.dumps(v)}"] for k, v in cert.items() if k != "model"), []),
                   "--out", str(where)]
            gate = run_command(cmd, timeout_s=CANARY_S, port=cfg.launch.port)
            gates[json.dumps(cert, sort_keys=True)] = {"passed": gate.returncode == 0, "report": str(where),
                                                       "scope": "keep pauses, full weights", "certifies": cert}
            if gate.returncode != 0:
                log(f"canary failed for {cert}: " + gate.stdout[-2000:])
        if not any(g["passed"] for g in gates.values()):
            return stopped("canary failed")
    # the grid is cut only at a seed block's boundary (RUN.md §3): by the owner's STOP file or by `deadline_h`
    # from the queue's start; the rows not run are reported so, and the exit status is 0 (nothing failed)
    t0_queue, stop_file, last_seed, cut = time.monotonic(), qdir / "STOP", None, None
    for arm, variant, seed, sets, cfg in planned:
        if seed != last_seed:
            last_seed = seed
            if stop_file.exists():
                cut = f"{stop_file} is there"
            elif deadline_h is not None and time.monotonic() - t0_queue > deadline_h * 3600:
                cut = f"the deadline of {deadline_h} h from the queue's start passed"
            if cut:
                log(f"cut at seed {seed}: {cut}; the remaining rows are not run")
        if cut:
            report.append({"arm": arm, "variant": variant, "seed": seed, "row": None, "set": sets,
                           "status": "not run (cut)", "why": cut})
            continue
        path = row_dir_on_host(cfg, Path(out))
        entry = {"arm": arm, "variant": variant, "seed": seed, "row": str(path), "set": sets}
        gate = gates.get(json.dumps(certificate(cfg), sort_keys=True))
        if gates and not gate["passed"]:
            report.append(entry | {"status": "canary failed", "certificate": gate["report"]})
            continue
        # finished and valid, of this very measurement (an invalid one is moved aside and run again once; another
        # measurement under the same label stops the queue: external review F01)
        if finished(path, cfg, dry_run=dry_run):
            if problem := row_problem(Row(path).metrics):  # its rerun had the same problem: it stands, unevaluated
                report.append(entry | {"status": "invalid (stands)", "problem": problem})
                continue
            done = entry | {"status": "done"}
            if matrix.get("evaluate") and not dry_run:
                _evaluate(path, cfg, done, prune=matrix.get("prune", False))
            report.append(done)
            continue
        if dry_run:
            report.append(entry | {"status": "planned"})
            continue
        row_timeout = min(timeout_s, (cfg.budget_s if cfg.budget_s else timeout_s) + ROW_MARGIN_S)
        for attempt in (1, 2):
            if not gpu_free():
                report.append(entry | {"status": "gpu never free"})
                break
            t0 = time.monotonic()
            cmd = [sys.executable, "-m", "halfwritten", "run", base, "--out", out, *sum((["--set", s] for s in sets),
                                                                                       [])]
            log(f"[{arm} s{seed}] attempt {attempt}: {' '.join(cmd[3:])}")
            p = run_command(cmd, timeout_s=row_timeout, port=cfg.launch.port)
            row = Row(path)
            if not row.finished:  # no row: the process crashed or was killed before writing it
                if attempt == 2:
                    name = f"failed-{variant + '-' if variant else ''}{arm}-s{seed}.json"
                    (qdir / name).write_text(json.dumps(entry | {"log_tail": p.stdout[-4000:]}, indent=1))
                    report.append(entry | {"status": "failed"})
                continue
            problem = row_problem(row.metrics)
            if problem and attempt == 1 and not rerun_pending(path):
                # the next attempt's run_row (finished) moves it aside for the record and runs it once more; one
                # rule, in finished(), counts the reruns
                report.append(entry | {"status": "invalid, rerun", "problem": problem})
                continue
            done = entry | {"status": "valid" if problem is None else "invalid", "seconds": time.monotonic() - t0}
            if problem:
                done["problem"] = problem
            elif row.metrics.get("host") and (why := harness_problem(row.metrics)):
                done["harness"] = why  # valid, kept, flagged: decision 7's rule is applied in the analysis
            if gate:  # the canary this row ran under
                (path / "certificate.json").write_text(json.dumps(gate, indent=1))
            if matrix.get("evaluate") and problem is None:
                _evaluate(path, cfg, done, prune=matrix.get("prune", False))
            report.append(done)
            break
    (qdir / f"report-{int(time.time())}.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("matrix")
    r.add_argument("--out", default="results")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--variants", default=None, help="comma-separated: run only these of the matrix's variants")
    r.add_argument("--gpu-wait-s", type=float, default=None, help="the most to wait for a free GPU, per row")
    r.add_argument("--amendment", default=None, help="the amendment file, when not the matrix's own")
    r.add_argument("--deadline-h", type=float, default=None,
                   help="cut the grid at the next seed block once this many hours have passed (as _queue/STOP does)")
    a = ap.parse_args(argv)
    report = run_matrix(a.matrix, a.out, a.dry_run, variants=a.variants.split(",") if a.variants else None,
                        gpu_wait_s=a.gpu_wait_s, amendment=a.amendment, deadline_h=a.deadline_h)
    for e in report:
        seed = e.get("seed")
        why = e.get("problem") or (f"harness: {e['harness']}" if e.get("harness") else "")
        print(f"{e.get('status', '?'):>16}  {e.get('arm') or '-':>6} s{'-' if seed is None else seed}"
              f"  {e.get('row') or ''}{'  ' + why if why else ''}")
    if any(e.get("status") in FAILED for e in report):
        return 1
    return RC_ROWS_STAND if any(e.get("status") in STANDS for e in report) else 0


if __name__ == "__main__":
    sys.exit(main())
