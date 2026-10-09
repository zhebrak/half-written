"""The canary (PLAN.md §5.3): does a swap's boundary land where the ledger puts it, on the real engine?

    python -m bench.canary --config CONFIG --background-answers 255 --out DIR
        the session gate: exits 1 if the canary fails
    python -m bench.canary --probe --out DIR
        first, what vLLM 0.30 does (Phase 0's first GPU hour) and golden fixtures for the CPU tests; then the canary

The canary launches `vllm serve` as the runs do (with --config: that run's flags), and pushes from HF copies of the
model on the engine's GPU over IPC. The update changes every greedy token: the tied embedding's rows are permuted
and half the final norm's signs flipped (trainer/sync.py `canary_update`). It is swapped in mid-answer under keep
without and with clear_cache (D and E). The swap fences with the runs' own rule (fence.py), and `--background-
answers N` keeps N other answers streaming throughout, as a run would. HF teacher forcing then classifies each later
token as OLD, FRESH (new weights, recomputed KV) or STALE (new weights reading old-weights KV). D must be STALE, E
FRESH, and the boundary the delivered count before resume; a boundary counts as verified only where the regime's
hypothesis and OLD predict different tokens there, each read within the 2-ulp rule every match uses.

--probe first checks, recording raw SSE transcripts and /metrics as fixtures:
  stream    the chunk fields VllmHttpEngine parses (token_ids, logprobs.token_logprobs, finish_reason)
  abort     an aborted stream: its last chunk, its finish reason, the tokens it kept
  keep      a frozen request delivers nothing while paused and finishes after resume
  reset     keep + clear_cache with 4 running: does vllm:num_preemptions_total count the reset, and when
  wait      pause(wait) returns only after the running request finished
  queued    a request sent while paused waits for resume (keep and abort)
  push      an IPC push of identical weights: /weight_info names the version, greedy output unchanged
Everything lands in <out>/report.json; fixtures in <out>/fixtures/.

The CPU half, `compare`, checks a ledger against the per-token ground truth the fake engine keeps, so the
controller's boundaries are tested on every regime and delivery mode without a GPU.
"""
import argparse
import asyncio
import itertools
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from halfwritten.engines.vllm_http import (
    VllmHttpEngine,
    kv_tokens,
    parse_metrics,
    run_uvloop,
)
from halfwritten.fence import exact_cut
from halfwritten.ledger import Ledger
from halfwritten.revisions import model_revision
from halfwritten.types import CONTEXT_CODE

Truth = Callable[[str, int], list[tuple[int, bool]]]  # (sid, attempt) -> per token (version, read other-version KV)


def compare(ledger: Ledger, truth: Truth) -> dict:
    """Per closed attempt, the ledger's version and stale flag of each token against the truth. A boundary offset is
    where the truth's version changes minus where the ledger put the cut: 0 when exact, +k when the ledger cut k
    tokens early (old-weights tokens labelled new, as when outputs are held past resume)."""
    bad_tokens = bad_samples = tokens = 0
    offsets: Counter = Counter()
    examples = []
    for sid in ledger.closed():
        versions = ledger.token_versions(sid)
        stale = ledger.token_context_codes(sid) == CONTEXT_CODE["stale_kv"]
        want = truth(sid, ledger.attempt(sid))
        if len(want) != len(versions):
            bad_samples += 1
            examples.append(f"{sid}: ledger has {len(versions)} tokens, truth {len(want)}")
            continue
        tokens += len(want)
        wrong = [i for i, (v, mixed) in enumerate(want) if v != versions[i] or mixed != bool(stale[i])]
        if wrong:
            bad_samples += 1
            bad_tokens += len(wrong)
            if len(examples) < 5:
                examples.append(f"{sid}: {len(wrong)} tokens mislabelled, first at {wrong[0]}")
        cuts = [s.start for s in ledger.segments(sid)[1:]]
        changes = [i for i in range(1, len(want)) if want[i][0] != want[i - 1][0]]
        for c in cuts:
            nearest = min(changes, key=lambda x: abs(x - c), default=None)
            offsets[None if nearest is None else nearest - c] += 1
    return {"ok": bad_tokens == 0 and bad_samples == 0, "tokens": tokens, "bad_tokens": bad_tokens,
            "bad_samples": bad_samples, "boundary_offsets": dict(offsets), "examples": examples}




def canary_config(a):
    """The engine the canary talks to: with --config, launched exactly as that run's (its concurrency, scheduling,
    LM head, stream interval and API servers), so the gate certifies the launch the runs use; without, a plain
    64-answer engine. IPC either way, on the engine's GPU, as the canary's HF models share it."""
    from halfwritten.config import RunConfig, load, with_overrides
    cfg = load(a.config) if a.config else RunConfig(regime="A", K=0, S=1, model=a.model, M=8, G=8,
                                                    max_model_len=4096)
    return with_overrides(cfg, [f"model={a.model}", "lora=false", "launch.engine=vllm",
                                "launch.gpu_memory_utilization=0.6", "launch.weight_backend=ipc",
                                f"launch.trainer_gpu='{cfg.launch.engine_gpu}'", *a.set])


def body(prompt, max_tokens, ignore_eos=False, model=None):
    return {"model": model, "prompt": list(prompt), "max_tokens": max_tokens, "temperature": 0.0, "logprobs": 0,
            "stream": True, "return_token_ids": True, "skip_special_tokens": False, "ignore_eos": ignore_eos}


class Stream:
    """One streamed completion through the runs' own client: raw SSE data lines and the tokens so far."""

    def __init__(self, eng: VllmHttpEngine, b: dict, rid: str):
        self.eng, self.body, self.rid = eng, b, rid
        self.lines, self.tokens, self.finish = [], [], None
        self.task = asyncio.create_task(self._run())

    async def _run(self):
        async for payload, c in self.eng.stream(self.body, self.rid, raw=True):
            self.lines.append("data: " + payload.decode())
            self.tokens += c.token_ids
            if c.finish:
                self.finish = c.finish
        self.lines.append("data: [DONE]")

    async def until(self, n: int, timeout: float = 120):
        t0 = time.monotonic()
        while len(self.tokens) < n:
            if self.task.done():
                raise RuntimeError(f"{self.rid} ended with {len(self.tokens)} tokens")
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(self.rid)
            await asyncio.sleep(0.005)


async def preemptions(eng) -> float:
    return (await eng.metrics()).get("vllm:num_preemptions_total", 0.0)


async def pause(eng, mode, clear):
    t0 = time.monotonic()
    await eng.pause(mode, clear)
    return time.monotonic() - t0


async def get_json(eng, path):
    return json.loads(await eng.request("GET", path))


MIN_DECISIVE = 20  # positions where STALE and FRESH predict different tokens and no near-tie could flip either
MIN_MATCH = 0.98  # rates count a token within 2 bf16 ulps of the top logit; 1 in 50 may miss on kernel near-ties


def canary_failures(report: dict) -> list[str]:
    """Why the session's canary failed (empty: it passed): D must read stale KV and E recomputed KV, each with its
    boundary verified, at positions where STALE and FRESH decisively predict different tokens.

    History (HYPOTHESES.md, the gate): to 2026-10-02 the evidence was a share of positions where the hypotheses'
    argmaxes differed (50%, then 20 such positions after a 400-token canary under load failed only on the share).
    Many of those positions are near-ties the 2-ulp rule lets either hypothesis claim, so the gate now counts
    decisive positions, and asks the regime's own hypothesis to match at 98% of them. Reports from before then lack
    the counts: `python -m bench.canary --reclassify DIR` recomputes them from the transcripts."""
    bad = []
    d, e = report.get("canary_D"), report.get("canary_E")
    if not d or not e:
        return ["canary did not run"]
    if (n := d.get("preemptions", 0)) > 0:
        bad.append(f"D: vLLM preempted {n} requests during the canary (KV pressure): its answer may have been "
                   "recomputed; give the engine more memory (--gpu-memory-utilization)")
    for name in ("D", "E"):  # the exact fence cut before every token the engine generated had arrived
        if (left := report.get(f"canary_{name}_undelivered", 0)) != 0:
            bad.append(f"{name}: the engine's count and the tokens delivered differ by {left} at the cut")
        if report.get(f"canary_{name}_settled") is False:
            bad.append(f"{name}: the engine's count was still moving when the fence gave up")
    if not (d["pre_old"] == 1.0 and d["post_stale"] >= MIN_MATCH and d.get("boundary_verified_stale")):
        bad.append(f"D: pre_old {d['pre_old']}, post_stale {d['post_stale']}, "
                   f"boundary {d.get('boundary_verified_stale')}")
    if not (e["pre_old"] == 1.0 and e["post_fresh"] >= MIN_MATCH and e.get("boundary_verified_fresh")):
        bad.append(f"E: pre_old {e['pre_old']}, post_fresh {e['post_fresh']}, "
                   f"boundary {e.get('boundary_verified_fresh')}")
    for name, own, r in (("D", "stale", d), ("E", "fresh", e)):
        if r.get("boundary") is not None and r.get("total") is not None and r["boundary"] >= r["total"]:
            bad.append(f"{name}: the pause landed after the answer finished ({r['total']} tokens): nothing to test")
            continue
        n = r.get("decisive_stale_fresh")
        if n is None:
            bad.append(f"{name}: no decisive positions recorded (a report from before 2026-10-02: reclassify it)")
        elif n < MIN_DECISIVE:
            # an update under which STALE and FRESH agree cannot tell D from E: identical transcripts pass both
            bad.append(f"{name}: STALE and FRESH decisively differ at only {n} post-swap positions")
        elif (m := r[f"post_{own}_on_decisive"]) < MIN_MATCH:
            bad.append(f"{name}: matches {own.upper()} at {m:.0%} of the {n} positions that decide between STALE "
                       "and FRESH")
    return bad


async def probe(c: VllmHttpEngine, prompt: list[int], out: Path) -> dict:
    """What the engine does at the pause modes (--probe), with golden fixtures for the CPU tests."""
    model = c.model
    fx = out / "fixtures"
    fx.mkdir(parents=True, exist_ok=True)
    rep: dict = {}
    rep["weight_info_initial"] = await get_json(c, "/weight_info")
    metrics_text = (await c.request("GET", "/metrics")).decode()
    (fx / "metrics.txt").write_text(metrics_text)
    rep["metric_names"] = sorted(parse_metrics(metrics_text))
    rep["kv_tokens"] = kv_tokens(metrics_text)

    s = Stream(c, body(prompt, 12, model=model), "stream#0")
    await s.task
    (fx / "stream.sse").write_text("\n".join(s.lines) + "\n")
    rep["stream"] = {"tokens": len(s.tokens), "finish": s.finish, "chunks": len(s.lines)}

    s = Stream(c, body(prompt, 2000, True, model), "abort#0")
    await s.until(32)
    rep_abort = {"pause_s": await pause(c, "abort", True)}
    await asyncio.wait_for(s.task, 60)
    (fx / "abort.sse").write_text("\n".join(s.lines) + "\n")
    await c.resume()
    rep_abort.update(tokens=len(s.tokens), finish=s.finish, last_chunk=s.lines[-2] if len(s.lines) > 1 else None)
    rep["abort"] = rep_abort

    s = Stream(c, body(prompt, 300, True, model), "keep#0")
    await s.until(32)
    ack = await pause(c, "keep", False)
    n_ack = len(s.tokens)
    await asyncio.sleep(1.0)
    n_later = len(s.tokens)
    paused = await get_json(c, "/is_paused")
    await c.resume()
    await asyncio.wait_for(s.task, 120)
    rep["keep"] = {"pause_s": ack, "tokens_at_ack": n_ack, "tokens_1s_later": n_later, "is_paused": paused,
                   "total": len(s.tokens), "finish": s.finish}

    streams = [Stream(c, body(prompt, 400, True, model), f"reset#{i}") for i in range(4)]
    for st in streams:
        await st.until(32)
    p0 = await preemptions(c)
    await pause(c, "keep", True)
    p1 = await preemptions(c)
    await asyncio.sleep(0.5)
    p2 = await preemptions(c)
    await c.resume()
    await asyncio.sleep(0.5)
    p3 = await preemptions(c)
    await asyncio.gather(*(st.task for st in streams))
    p4 = await preemptions(c)
    rep["reset"] = {"running": 4, "before": p0, "at_ack": p1, "paused_0.5s": p2, "resumed_0.5s": p3, "end": p4,
                    "totals": [len(st.tokens) for st in streams]}

    s = Stream(c, body(prompt, 200, True, model), "wait#0")
    await s.until(16)
    t0 = time.monotonic()
    await pause(c, "wait", True)
    t_ack = time.monotonic()
    rep["wait"] = {"ack_after_s": t_ack - t0, "finished_before_ack": s.task.done() or len(s.tokens) == 200,
                   "tokens_at_ack": len(s.tokens)}
    await c.resume()
    await asyncio.wait_for(s.task, 60)

    for mode in ("keep", "abort"):
        await pause(c, mode, mode != "keep")
        q = Stream(c, body(prompt, 20, True, model), f"queued-{mode}#0")
        await asyncio.sleep(1.0)
        during = len(q.tokens)
        await c.resume()
        await asyncio.wait_for(q.task, 60)
        rep[f"queued_{mode}"] = {"tokens_while_paused": during, "total": len(q.tokens), "finish": q.finish}
    return rep


async def canary(c: VllmHttpEngine, url: str, prompt: list[int], hf, out: Path, cfg,
                 load_streams: int = 0, seed: int = 7) -> dict:
    """IPC pushes from an HF copy of the model on the same GPU, then the D/E canary with permuted weights. The canary
    cuts by the runs' own rule (fence.py: the engine's generated-token count, settled), as the controller does."""
    from halfwritten.trainer.sync import VllmSink, canary_update
    model = c.model
    (out / "fixtures").mkdir(parents=True, exist_ok=True)  # the transcripts each regime's verdict rests on
    old, new = hf
    sink = VllmSink(url, backend="ipc", packed=True)
    base = {k: v.detach() for k, v in old.named_parameters()}
    rep = {}

    s = Stream(c, body(prompt, 24, model=model), "before-push#0")
    await s.task
    pushed = await asyncio.to_thread(sink.push, 1, base)
    info = await get_json(c, "/weight_info")
    s2 = Stream(c, body(prompt, 24, model=model), "after-push#0")
    await s2.task
    rep["push"] = {"seconds": pushed.seconds, "bytes": pushed.bytes, "weight_info": info,
                   "same_greedy_output": s.tokens == s2.tokens}

    permuted = canary_update(base, seed=seed)
    version = 1
    # optional load: background answers streaming through both phases, as the runs' concurrency would be (the gate
    # must certify the boundary under the load the runs see; a backlogged server delays old tokens)
    loop = asyncio.get_running_loop()
    # the exact fence counts from here: every earlier answer has finished, so all it generated was delivered
    generated0 = (await c.counters()).generated_tokens_total
    streams: list[Stream] = []

    async def generated() -> int:
        return (await c.counters()).generated_tokens_total - generated0

    def delivered() -> int:
        return sum(len(st.tokens) for st in streams)

    async def background(i):  # one slot of the load, answer after answer, as a run keeps M*G answers in flight
        for k in itertools.count():
            st = Stream(c, body([(i * 37 + j) % 1000 + 10 for j in range(64)], cfg.max_tokens, True, model),
                        f"load-{i}-{k}#0")
            streams.append(st)
            await st.task

    load = [asyncio.create_task(background(i)) for i in range(load_streams)]

    def loaded():  # a slot ends only on an error: the gate must not certify "under load" after the load collapsed
        if dead := [t for t in load if t.done()]:
            raise RuntimeError(f"{len(dead)} background slots died, e.g. {dead[0].exception()!r}")

    try:
        for name, clear in (("D", False), ("E", True)):
            version += 1
            if load:  # weights never change under live generation: vLLM's engine dies (seen on two H100s)
                await pause(c, "keep", False)
            await asyncio.to_thread(sink.push, version, base)  # start from the original weights
            if load:
                await c.resume()
            preempted0 = await preemptions(c)
            s = Stream(c, body(prompt, CANARY_TOKENS, True, model), f"canary-{name}#0")
            streams.append(s)
            await s.until(24)
            loaded()
            # the controller's sync, step for step (controller.py _sync_push): the engine's count read alongside the
            # push, then the cut once every counted token has arrived (fence.exact_cut, the one rule both follow)
            await pause(c, "keep", clear)
            version += 1
            _, _, count, settled, _ = await exact_cut(asyncio.to_thread(sink.push, version, permuted), generated,
                                                      delivered, loop.time)
            rep[f"canary_{name}_undelivered"] = count - delivered()
            rep[f"canary_{name}_settled"] = settled
            n = len(s.tokens)
            await c.resume()
            await asyncio.wait_for(s.task, 120)
            rep[f"canary_{name}"] = classify(old, new, prompt, s.tokens, n) | {
                "boundary": n, "total": len(s.tokens), "load_streams": load_streams,
                "fence": "exact",
                # E's reset preempts every running request; any under D is KV pressure, which recomputes the answer
                "preemptions": int(await preemptions(c) - preempted0)}
            (out / "fixtures" / f"canary_{name}.json").write_text(json.dumps(
                {"prompt": list(prompt), "tokens": s.tokens, "boundary": n, "versions": [version - 1, version]}))
    finally:
        for t in load:  # cancels the stream each slot is waiting on
            t.cancel()
        await asyncio.gather(*load, return_exceptions=True)
    return rep


CANARY_TOKENS = 400  # long enough that the pause lands mid-answer even when delivery comes in large lumps


def bf16_ulp(x: float) -> float:
    """The spacing of bfloat16 numbers at x (8 significant bits): 0.125 for a logit of 20."""
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else 2.0 ** -133


def classify(old, new, prompt, tokens, n, ulps: float = 2.0) -> dict:
    """Teacher-forced argmax of three hypotheses for each engine token after the boundary n: OLD (old weights),
    FRESH (new weights on everything), STALE (old weights build the KV of prompt + tokens[:n-1], new weights continue
    from tokens[n-1]). A token matches a hypothesis when its logit is within `ulps` bf16 ulps of that hypothesis' top
    logit: the engine and HF compute bf16 logits with different kernels, so a near-tie may resolve either way (PLAN.md
    §5.3). Each hypothesis' misses are listed with their margin in ulps, so a failure shows whether it is a tie."""
    import torch
    dev = next(old.parameters()).device
    ids = torch.tensor([list(prompt) + list(tokens)], device=dev)
    p = len(prompt)
    with torch.no_grad():
        lo = old(ids).logits[0]
        lf = new(ids).logits[0]
        past = old(ids[:, :p + n - 1], use_cache=True).past_key_values
        ls = new(ids[:, p + n - 1:], past_key_values=past, use_cache=True).logits[0]

    def margin(logits_at, i):
        row = logits_at(i).float()
        top = float(row.max())
        return (top - float(row[tokens[i]])) / bf16_ulp(top)

    def rate(logits_at, positions):
        return sum(margin(logits_at, i) <= ulps for i in positions) / max(1, len(positions))

    def misses(logits_at, positions):
        return [{"i": i, "ulps": round(m, 1)} for i in positions if (m := margin(logits_at, i)) > ulps]

    def top(logits_at, i):
        return int(logits_at(i).float().argmax())

    old_at, fresh_at, stale_at = (lambda i: lo[p + i - 1]), (lambda i: lf[p + i - 1]), (lambda i: ls[i - n])
    post = range(n, len(tokens))
    pre = range(0, n)
    # positions where the hypotheses disagree: only these can tell which weights sampled a token
    disc_fresh = [i for i in post if top(old_at, i) != top(fresh_at, i)]
    disc_stale = [i for i in post if top(old_at, i) != top(stale_at, i)]
    disc_sf = [i for i in post if top(stale_at, i) != top(fresh_at, i)]  # D and E predict different tokens here

    def decisive(i):  # STALE and FRESH disagree, each by more than `ulps` over the other's token in its own row
        s_row, f_row = stale_at(i).float(), fresh_at(i).float()
        ts, tf = int(s_row.argmax()), int(f_row.argmax())
        return ts != tf and (float(s_row[ts] - s_row[tf]) / bf16_ulp(float(s_row[ts])) > ulps
                             and float(f_row[tf] - f_row[ts]) / bf16_ulp(float(f_row[tf])) > ulps)

    dec = [i for i in disc_sf if decisive(i)]
    return {"pre_old": rate(old_at, pre),
            "post_old": rate(old_at, post),
            "post_fresh": rate(fresh_at, post),
            "post_stale": rate(stale_at, post),
            "discriminating_fresh": len(disc_fresh) / max(1, len(post)),
            "discriminating_stale": len(disc_stale) / max(1, len(post)),
            "discriminating_stale_fresh": len(disc_sf) / max(1, len(post)),
            "post_fresh_where_stale_differs": rate(fresh_at, disc_sf),
            "post_stale_where_fresh_differs": rate(stale_at, disc_sf),
            "post_fresh_on_discriminating": rate(fresh_at, disc_fresh),
            "post_stale_on_discriminating": rate(stale_at, disc_stale),
            # positions no near-tie can decide either way: the gate's evidence (canary_failures)
            "decisive_stale_fresh": len(dec), "post_stale_on_decisive": rate(stale_at, dec),
            "post_fresh_on_decisive": rate(fresh_at, dec),
            # the boundary: the last pre-boundary token was the old weights' and the first post-boundary one is the
            # regime's hypothesis' and not the old weights', each within the ulps rule like every match above (an
            # exact-argmax reading failed a boundary token that was a near-tie in FRESH's own row, the H100 PCIe
            # rehearsal of 2026-10-03, with every one of 152 decisive positions after it FRESH's); a position where
            # OLD and the hypothesis agree cannot verify the boundary. `_exact` keeps the argmax reading for the record.
            "boundary_verified_fresh": 1 <= n < len(tokens) and rate(old_at, [n - 1]) == 1.0
                                       and rate(fresh_at, [n]) == 1.0 and rate(old_at, [n]) == 0.0,
            "boundary_verified_stale": 1 <= n < len(tokens) and rate(old_at, [n - 1]) == 1.0
                                       and rate(stale_at, [n]) == 1.0 and rate(old_at, [n]) == 0.0,
            "boundary_exact_fresh": n < len(tokens) and top(old_at, n) != top(fresh_at, n)
                                    and tokens[n] == top(fresh_at, n),
            "boundary_exact_stale": n < len(tokens) and top(old_at, n) != top(stale_at, n)
                                    and tokens[n] == top(stale_at, n),
            "first_post": [{"i": i, "old": rate(old_at, [i]), "fresh": rate(fresh_at, [i]),
                            "stale": rate(stale_at, [i])} for i in range(n, min(n + 4, len(tokens)))],
            "misses": {"pre_old": misses(old_at, pre), "post_fresh": misses(fresh_at, post),
                       "post_stale": misses(stale_at, post)}, "ulps": ulps}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--out", default="probe")
    ap.add_argument("--config", default=None, help="launch the engine with this run config's flags (the session gate)")
    ap.add_argument("--set", action="append", default=[],
                    help="overrides to --config, e.g. launch.head_fp32=true or launch.port=8101; vLLM's memory share "
                         "defaults to 0.6 (the rest holds the two HF copies; under load the KV must still hold M*G "
                         "answers, or vLLM preempts and the canary fails on D's recomputed requests)")
    ap.add_argument("--background-answers", type=int, default=0,
                    help="other answers streaming throughout, as a run keeps them (a run's M*G - 1)")
    ap.add_argument("--probe", action="store_true", help="first, the engine's pause semantics and golden fixtures")
    ap.add_argument("--no-weights", action="store_true", help="with --probe: skip the IPC push and the canary")
    ap.add_argument("--seed", type=int, default=None,
                    help="the update's permutation and the prompt (default: drawn, and recorded in the report)")
    ap.add_argument("--reclassify", metavar="DIR", default=None,
                    help="no engine: judge a canary's archived transcripts (DIR/fixtures) again under the current "
                         "classifier and gate, into DIR/reclassified.json")
    a = ap.parse_args(argv)
    if a.reclassify:
        return reclassify(Path(a.reclassify), a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(2), "big")
    tok = AutoTokenizer.from_pretrained(a.model, revision=model_revision(a.model))
    prompt = tok(PROMPTS[seed % len(PROMPTS)], add_special_tokens=False).input_ids
    from halfwritten.gpu import local_device
    cfg = canary_config(a)
    hf = None if a.probe and a.no_weights else hf_pair(a.model, seed, local_device(cfg.launch.engine_gpu))
    import vllm

    from halfwritten.results import host_gpu_label
    report = {"vllm": vllm.__version__, "model": a.model, "prompt_tokens": len(prompt), "seed": seed,
              "gpu": host_gpu_label()}
    report["launch"] = {k: getattr(cfg.launch, k) for k in ("async_scheduling", "head_fp32", "stream_interval",
                                                             "api_servers")}
    from halfwritten.launch import tool_engine
    with tool_engine(cfg, out / "logs") as (url, engine):
        if a.probe:
            report |= run_uvloop(probe(engine, prompt, out))
        if hf is not None:
            report |= run_uvloop(canary(engine, url, prompt, hf, out, cfg, a.background_answers, seed))
    report["canary_failures"] = canary_failures(report) if hf is not None else ["not run"]
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))
    if hf is not None and report["canary_failures"]:  # a gate must be able to stop a session
        sys.exit(1)


# the canary's prompts: one is drawn per run with the update's seed, so two canaries in a session are two tests
PROMPTS = ["The history of the Roman Empire begins", "In the morning the harbour was quiet, and",
           "A recipe for bread needs only four things:", "The first law of thermodynamics states that",
           "When the committee met on Tuesday, its chair", "Rivers carry sediment from the mountains to"]


def hf_pair(model: str, seed: int, device: str | None = None):
    """The canary's two HF models: the base (OLD) and the base under the canary update (NEW)."""
    import torch
    from transformers import AutoModelForCausalLM

    from halfwritten.trainer.sync import canary_update
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    old = AutoModelForCausalLM.from_pretrained(model, revision=model_revision(model), dtype=torch.bfloat16).to(
        device).eval()
    new = AutoModelForCausalLM.from_pretrained(model, revision=model_revision(model), dtype=torch.bfloat16).to(
        device).eval()
    with torch.no_grad():
        new.load_state_dict(canary_update({k: v.detach().clone() for k, v in new.named_parameters()}, seed=seed),
                            strict=False)
    return old, new


def reclassify(run: Path, model: str) -> list[str]:
    """Judge an archived canary again: classify its transcripts with the current classifier, keep the engine-side
    facts (preemptions, undelivered) from its report, apply the current gate. Canaries before --seed used seed 7."""
    report = json.loads((run / "report.json").read_text())
    old, new = hf_pair(model, report.get("seed", 7))
    again = {k: v for k, v in report.items() if k.endswith("_undelivered")}
    for name in ("D", "E"):
        fx = json.loads((run / "fixtures" / f"canary_{name}.json").read_text())
        engine_side = {k: v for k, v in report[f"canary_{name}"].items()
                       if k in ("preemptions", "fence", "boundary", "total", "versions")}
        again[f"canary_{name}"] = (classify(old, new, fx["prompt"], fx["tokens"], fx["boundary"])
                                   | {"boundary": fx["boundary"], "total": len(fx["tokens"])} | engine_side)
    again["canary_failures"] = canary_failures(again)
    again["canary_failures_then"] = report.get("canary_failures")
    again["reclassified_from"] = str(run)
    (run / "reclassified.json").write_text(json.dumps(again, indent=1, default=str) + "\n")
    keys = ("decisive_stale_fresh", "post_stale_on_decisive", "post_fresh_on_decisive", "post_stale", "post_fresh")
    for name in ("D", "E"):
        print(name, {k: again[f"canary_{name}"].get(k) for k in keys})
    print("then:", again["canary_failures_then"], "\nnow:", again["canary_failures"])
    return again["canary_failures"]


if __name__ == "__main__":
    main()
