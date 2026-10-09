"""Phase 1's teacher forcing on tiny random Qwen3 models: STALE is the new weights reading the old weights' KV of
the prompt and all but the last delivered token, HYBRID interpolates layer by layer, and the batched metrics agree
with the numpy references."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from bench.stale_kv import (  # noqa: E402
    alpha_rows,
    fit_alpha,
    kl,
    log_normalize,
    teacher_force,
)


def tiny(seed):
    from transformers import AutoModelForCausalLM, Qwen3Config

    from halfwritten.trainer.model import TINY
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(Qwen3Config(**TINY), dtype=torch.float32).eval()


PROMPT, ANSWER = list(range(5, 17)), list(range(40, 80))


def test_identical_weights_make_stale_kv_harmless():
    with torch.no_grad():
        m = tiny(0)
        r = teacher_force(m, m, PROMPT, ANSWER, n=10, offsets=8)
    assert np.allclose(r["kl_fresh_stale"], 0, atol=1e-6) and np.allclose(r["drift"], 0, atol=1e-6)


def test_stale_reads_old_kv_for_all_but_the_last_delivered_token():
    with torch.no_grad():
        old, new = tiny(0), tiny(1)
        r = teacher_force(old, new, PROMPT, ANSWER, n=10, offsets=6)
        ids = torch.tensor([PROMPT + ANSWER])
        p, n = len(PROMPT), 10
        cache = old(ids[:, :p + n - 1], use_cache=True).past_key_values  # old KV up to y[n-2]
        manual = new(ids[:, p + n - 1:p + n + 5], past_key_values=cache, use_cache=True).logits[0]
        fresh = torch.log_softmax(new(ids).logits[0, p + n - 1:p + n + 5], -1)
        manual = torch.log_softmax(manual, -1)
        assert np.allclose(r["kl_fresh_stale"], (fresh.exp() * (fresh - manual)).sum(-1).numpy(), atol=1e-5)
        # the off-by-one: old KV for every delivered token (y[n-1] included) is a different, wrong hypothesis
        wrong_cache = old(ids[:, :p + n], use_cache=True).past_key_values
        wrong = torch.log_softmax(new(ids[:, p + n:p + n + 5], past_key_values=wrong_cache).logits[0], -1)
        assert not torch.allclose(wrong, manual[1:], atol=1e-4)


def test_hybrid_cache_runs_from_stale_to_fresh():
    with torch.no_grad():
        old, new = tiny(0), tiny(1)
        r = teacher_force(old, new, PROMPT, ANSWER, n=10, offsets=6, fresh_layers=[1, 2])
    assert np.allclose(r["kl_fresh_stale_l2"], 0, atol=1e-5)  # every layer recomputed: FRESH
    assert r["kl_fresh_stale"].mean() > 1e-4  # no layer recomputed: STALE, which differs from FRESH


def test_batched_alpha_matches_the_reference_and_recovers_mixtures():
    rng = np.random.default_rng(0)
    lo = log_normalize(rng.normal(size=(5, 40)))
    lf = log_normalize(lo + rng.normal(scale=0.7, size=(5, 40)))
    truth = np.array([0.0, 0.3, 0.8, 1.0, 1.4])
    ls = np.stack([log_normalize((1 - a) * lo[i] + a * lf[i]) for i, a in enumerate(truth)])
    got = alpha_rows(torch.tensor(lo), torch.tensor(lf), torch.tensor(ls)).numpy()
    assert np.allclose(got, truth, atol=1e-3)
    assert fit_alpha(lo[1], lf[1], ls[1]) == pytest.approx(0.3, abs=1e-4)
    assert kl(lo[0], lo[0]) == pytest.approx(0.0)



def test_the_same_weights_agree_bit_for_bit_in_bf16():
    """OLD, FRESH and STALE share one computation, so at identical weights even bf16 gives exactly zero."""
    with torch.no_grad():
        m = tiny(0).to(torch.bfloat16)
        r = teacher_force(m, m, PROMPT, ANSWER, n=10, offsets=8)
    assert np.all(r["kl_fresh_stale"] == 0) and np.all(r["drift"] == 0)


def test_cut_at_stop_drops_the_stop_and_what_follows():
    from bench.stale_kv import cut_at_stop

    class Tok:
        def decode(self, ids):
            return "".join(chr(97 + i) if i < 26 else "\\n" for i in ids)
    ids = [0, 1, 2, 26, 16, 20, 4]  # "abc\nque"
    assert cut_at_stop(ids, Tok(), ["\\nqu"]) == [0, 1, 2]
    assert cut_at_stop(ids, Tok(), ["zz"]) == ids


def test_batched_answers_match_the_unbatched_greedy_ones():
    """Left padding must not change what each prompt's answer is (greedy, so the comparison is exact)."""
    from bench.stale_kv import sample_answers

    class Tok:
        eos_token_id, pad_token_id = 0, 0

        def decode(self, ids):
            return " ".join(map(str, ids))
    with torch.no_grad():
        m = tiny(0)
        prompts = [PROMPT, PROMPT[:5], PROMPT[3:]]
        batched = sample_answers(m, Tok(), prompts, "cpu", 12, 0.0, [])
        single = [sample_answers(m, Tok(), [p], "cpu", 12, 0.0, [])[0] for p in prompts]
    assert batched == single and all(len(a) > 0 for a in batched)


def _greedy(model, ids, past, steps):
    out = []
    for _ in range(steps):
        o = model(input_ids=ids, past_key_values=past, use_cache=True)
        past, nxt = o.past_key_values, int(o.logits[0, -1].argmax())
        out.append(nxt)
        ids = torch.tensor([[nxt]])
    return out, past


@pytest.mark.parametrize("hypothesis", ["stale", "fresh"])
def test_canary_classifies_a_known_swap(hypothesis):
    """Tokens written by OLD up to the boundary, then by STALE (D) or FRESH (E): the classifier must name the
    hypothesis that wrote them and verify the boundary. NEW is OLD under the canary's update, as on the GPU (two
    random tiny models both just repeat the prompt's last token, which tests nothing)."""
    import copy

    from bench.canary import classify
    from halfwritten.trainer.sync import canary_update
    with torch.no_grad():
        old, new = tiny(0), tiny(0)
        new.load_state_dict(canary_update({k: v.detach().clone() for k, v in new.named_parameters()}, seed=7),
                            strict=False)
        prompt, n, total = PROMPT, 12, 40
        pre, cache = _greedy(old, torch.tensor([prompt]), None, n)
        if hypothesis == "stale":
            # the engine's view: KV of prompt + pre[:n-1] from OLD; pre[n-1] is then fed to NEW
            o = old(input_ids=torch.tensor([prompt + pre[:-1]]), use_cache=True)
            post, _ = _greedy(new, torch.tensor([[pre[-1]]]), copy.deepcopy(o.past_key_values), total - n)
        else:
            post, _ = _greedy(new, torch.tensor([prompt + pre]), None, total - n)
        r = classify(old, new, prompt, pre + post, n)
    assert r["pre_old"] == 1.0 and r[f"post_{hypothesis}"] == 1.0 and not r["misses"][f"post_{hypothesis}"]
    assert r[f"boundary_verified_{hypothesis}"]
    other = "fresh" if hypothesis == "stale" else "stale"
    assert r[f"post_{other}"] < 0.6
    # at decisive positions (STALE and FRESH disagree beyond any near-tie) only the writer matches
    assert r["decisive_stale_fresh"] > 0
    assert r[f"post_{hypothesis}_on_decisive"] == 1.0 and r[f"post_{other}_on_decisive"] == 0.0


def test_residual_is_zero_on_the_line_and_large_off_it():
    """resid: what the best OLD-FRESH mixture cannot explain. FRESH itself (alpha 1) lies on the line; a STALE that
    moved orthogonally to the update does not, however its alpha comes out."""
    with torch.no_grad():
        old, new = tiny(0), tiny(1)
        r = teacher_force(old, new, PROMPT, ANSWER, n=10, offsets=6, fresh_layers=[2])
    assert np.allclose(r["resid_l2"], 0, atol=1e-4) and np.allclose(r["alpha_l2"], 1, atol=0.05)
    assert np.all(r["resid"] >= -1e-6) and r["resid"].mean() > 1e-3  # random weights: STALE is off the line


def test_phase1_table_uses_ratios_of_means(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from bench.stale_kv import markdown, phase1_table
    rows = [{"lag": k, "prompt": j % 5, "offset": j, "drift": d, "kl_fresh_old": d, "kl_fresh_stale": 0.5 * d,
             "alpha": 0.3, "resid": 0.1, "kl_fresh_stale_l7": 0.25 * d}
            for k in (0, 4) for j, d in enumerate([1e-4] * 9 + [1e-1])]  # one heavy position, as in real data
    for r in rows:
        if r["lag"] == 0:
            r |= {"drift": 0.0, "kl_fresh_old": 0.0, "kl_fresh_stale": 0.0, "kl_fresh_stale_l7": 0.0}
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "positions.parquet")
    t = phase1_table(tmp_path)
    assert t[1]["R"] == pytest.approx(0.5) and t[1]["removed_l7"] == pytest.approx(0.5)
    assert t[1]["R_ci95"] == "0.50-0.50"  # every prompt has the same ratio
    assert t[1]["drift_mean"] > 10 * t[1]["drift_median"]  # the mean sees the heavy position, the median does not
    assert "| lag |" in markdown(t)
