"""Transcripts recorded from a real vLLM 0.30.0 server (bench/canary.py on an A100, 2026-10-01): the adapter must
parse what the engine actually sends, and the probe's findings that the design depends on stay pinned here."""
import json
from pathlib import Path

from halfwritten.engines.vllm_http import kv_tokens, parse_chunk, parse_metrics

FIX = Path(__file__).parent / "fixtures" / "vllm-0.30.0"


def chunks(name):
    out = []
    for line in (FIX / name).read_text().splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            out.append(parse_chunk(line[6:]))
    return out


def test_a_stream_parses_and_packs_several_tokens_per_chunk():
    cs = chunks("stream.sse")
    assert sum(len(c.token_ids) for c in cs) == 12 and cs[-1].finish == "length"
    assert all(len(c.token_ids) == len(c.logprobs) for c in cs)
    assert max(len(c.token_ids) for c in cs) > 1  # vLLM batches tokens into chunks; the ledger counts tokens


def test_an_abort_ends_with_an_empty_chunk_and_keeps_the_partial_answer():
    cs = chunks("abort.sse")
    assert cs[-1].finish == "abort" and cs[-1].token_ids == ()
    assert sum(len(c.token_ids) for c in cs) == 35


def test_metrics_carry_the_counters_the_controller_reads():
    text = (FIX / "metrics.txt").read_text()
    m = parse_metrics(text)
    for name in ("vllm:num_preemptions_total", "vllm:num_requests_running", "vllm:num_requests_waiting",
                 "vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:kv_cache_usage_perc"):
        assert name in m
    assert kv_tokens(text) == 86960


def test_probe_findings_the_design_depends_on():
    r = json.loads((FIX / "probe_report.json").read_text())
    assert r["vllm"] == "0.30.0" and r["weight_info_initial"]["weight_version"] == "default"
    # E's cache reset counts as preemptions, published only after resume: why I5 is cumulative
    reset = r["reset"]
    assert reset["at_ack"] == reset["paused_0.5s"] == reset["before"] and reset["resumed_0.5s"] - reset["before"] == 4
    assert r["keep"]["tokens_at_ack"] == r["keep"]["tokens_1s_later"] and r["keep"]["total"] == 300
    assert r["wait"]["finished_before_ack"]
    assert r["queued_keep"]["tokens_while_paused"] == 0 and r["queued_abort"]["tokens_while_paused"] == 0
    assert r["push"]["weight_info"]["weight_version"] == "1" and r["push"]["same_greedy_output"]
    # the mechanism: D continues on old-weights KV, E on recomputed KV, and the boundary is exact
    d, e = r["canary_D"], r["canary_E"]
    assert d["pre_old"] == 1.0 and d["post_stale"] == 1.0 and d["post_fresh"] < 0.9
    assert e["pre_old"] == 1.0 and e["post_fresh"] == 1.0 and e["post_stale"] < 0.9
    assert all(t["stale"] == 1.0 and t["fresh"] == 0.0 for t in d["first_post"])
    assert all(t["fresh"] == 1.0 and t["stale"] == 0.0 for t in e["first_post"])


def test_the_session_gate_rejects_the_first_probes_canary():
    """The first probe's E boundary was not verifiable (copying survived the vocabulary permutation): the gate built
    after the second review must reject such a canary and accept a clean one."""
    from bench.canary import canary_failures
    old = json.loads((FIX / "probe_report.json").read_text())
    assert canary_failures(old)  # no boundary_verified_* fields, no decisive positions: fails
    clean = {"canary_D": {"pre_old": 1.0, "post_stale": 1.0, "boundary_verified_stale": True,
                          "decisive_stale_fresh": 60, "post_stale_on_decisive": 1.0, "post_fresh_on_decisive": 0.0},
             "canary_E": {"pre_old": 1.0, "post_fresh": 1.0, "boundary_verified_fresh": True,
                          "decisive_stale_fresh": 60, "post_stale_on_decisive": 0.0, "post_fresh_on_decisive": 1.0}}
    assert canary_failures(clean) == []
    # the exact fence cut while tokens the engine had generated were still on the way, or the count was not ours
    assert any("differ by 3 at the cut" in x for x in canary_failures(clean | {"canary_E_undelivered": 3}))
    assert any("differ by -2 at the cut" in x for x in canary_failures(clean | {"canary_D_undelivered": -2}))
    # KV pressure preempted requests during D's canary: its answer may have been recomputed (E-like)
    pressed = clean | {"canary_D": clean["canary_D"] | {"preemptions": 2}}
    assert any("KV pressure" in x for x in canary_failures(pressed))
    # an update under which STALE and FRESH agree cannot tell D from E: identical transcripts would pass both
    blind = {k: v | {"decisive_stale_fresh": 5} for k, v in clean.items()}
    assert any("decisively differ at only 5 post-swap positions" in x for x in canary_failures(blind))
    # E's transcript scored as D: it matches STALE where STALE is also FRESH's prediction, so post_stale can stay
    # high on near-ties, but FRESH where the two decisively differ
    swapped = clean | {"canary_D": clean["canary_D"] | {"post_stale_on_decisive": 0.04}}
    assert any("D: matches STALE at 4% of the 60 positions" in x for x in canary_failures(swapped))
    # a report from before decisive positions were counted: judged again with --reclassify, never passed as it is
    legacy = {k: {f: x for f, x in v.items() if "decisive" not in f} for k, v in clean.items()}
    assert any("reclassify it" in x for x in canary_failures(legacy))


def test_bf16_ulp_sets_the_canary_tolerance():
    from bench.canary import bf16_ulp
    assert bf16_ulp(20.0) == 0.125 and bf16_ulp(-20.0) == 0.125 and bf16_ulp(1.0) == 2 ** -7
    assert bf16_ulp(0.0) > 0


def test_a_canary_whose_pause_missed_the_answer_fails_clearly():
    """Four API servers delivered D's whole 96-token canary answer before the pause landed: nothing after the swap
    to classify. That must read as an inconclusive canary, not as a regime failure or a pass."""
    from bench.canary import canary_failures
    r = {"pre_old": 1.0, "post_stale": 0.0, "post_fresh": 0.0, "boundary": 96, "total": 96,
         "discriminating_stale": 0.0, "discriminating_fresh": 0.0}
    out = canary_failures({"canary_D": r, "canary_E": r})
    assert any("pause landed after the answer finished" in x for x in out)


def test_probe_launches_the_runs_engine_or_a_plain_one():
    """The probe's engine comes from the same engine_command as the runs': with --config, the run's flags."""
    from types import SimpleNamespace

    from bench.canary import canary_config
    from halfwritten.launch import engine_command
    base = dict(model="Qwen/Qwen3-0.6B-Base", port=8100, gpu_memory_utilization=0.3, set=[])
    argv, env, _ = engine_command(canary_config(SimpleNamespace(config=None, **base)))
    assert argv[argv.index("--max-num-seqs") + 1] == "64" and argv[argv.index("--max-model-len") + 1] == "4096"
    assert '"ipc"' in argv[argv.index("--weight-transfer-config") + 1]
    a = SimpleNamespace(config="configs/two_gpu/gsm8k.yaml", **base | {"set": ["launch.api_servers=4"]})
    argv, env, _ = engine_command(canary_config(a))
    assert argv[argv.index("--api-server-count") + 1] == "4" and "--hf-overrides" in argv
    assert argv[argv.index("--max-num-seqs") + 1] == "256" and env["CUDA_VISIBLE_DEVICES"] == "0"
