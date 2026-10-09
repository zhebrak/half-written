"""Phase 1: how wrong is a stale KV cache? (PLAN.md §2.1, H3)

Teacher-forced over the same tokens x + y, for checkpoints pi_old and pi_new k optimiser steps apart, with the swap
after n answer tokens delivered:
    OLD    pi_old on everything
    FRESH  pi_new on everything (arm E)
    STALE  pi_old builds the KV of the prompt and y[:n-1]; pi_new runs from y[n-1] on, reading it (arm D: the last
           delivered token's KV is computed by the new weights)
    HYBRID(l)  as STALE, but the KV of layers < l recomputed by pi_new (which layers carry the staleness)
Per answer position after the swap (offset 0, 1, ...), on full-vocabulary log-probabilities at the sampling
temperature:
    alpha  the fraction of the update the stale cache still applies: the alpha minimising
           KL(p_STALE || p_alpha), p_alpha proportional to p_OLD^(1-alpha) p_FRESH^alpha; 1 = stale KV is harmless,
           0 = it undoes the update
    R      KL(FRESH || STALE) / KL(FRESH || OLD), the brief's ratio
    drift  KL(OLD || FRESH), the x-axis that does not depend on the learning rate
The single-position helpers (fit_alpha, ratio_not_applied) are numpy references the batched torch code is tested
against.

    python -m bench.stale_kv --base Qwen/Qwen3-0.6B-Base --snapshot-dir ROW/work --lags 1,2,4,8,16,32 \\
        --prompts 32 --swap 0.25,0.5,0.75 --positions 96 --out phase1/
    python -m bench.stale_kv --table phase1/                      the per-lag table (Markdown), no GPU
"""
import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from halfwritten.types import parse_snapshot

# -- numpy references (one position)

def log_normalize(logits: np.ndarray) -> np.ndarray:
    m = logits.max(-1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))


def kl(logp: np.ndarray, logq: np.ndarray) -> np.ndarray:
    """KL(p || q) over the last axis, from log-probabilities."""
    return (np.exp(logp) * (logp - logq)).sum(-1)


def mix(log_old: np.ndarray, log_fresh: np.ndarray, alpha: float) -> np.ndarray:
    return log_normalize((1 - alpha) * log_old + alpha * log_fresh)


def fit_alpha(log_old: np.ndarray, log_fresh: np.ndarray, log_stale: np.ndarray, lo: float = -1.0,
              hi: float = 2.0, iters: int = 60) -> float:
    """alpha minimising KL(stale || mix(alpha)) for one position (golden-section search: the objective is convex in
    alpha, being linear in alpha plus the log-partition of the geometric mixture)."""
    f = lambda a: float(kl(log_stale, mix(log_old, log_fresh, a)))  # noqa: E731
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def ratio_not_applied(log_old: np.ndarray, log_fresh: np.ndarray, log_stale: np.ndarray) -> float:
    """R = KL(FRESH || STALE) / KL(FRESH || OLD); nan when the update moved nothing."""
    denom = float(kl(log_fresh, log_old))
    return float(kl(log_fresh, log_stale)) / denom if denom > 0 else float("nan")


# -- batched torch (all positions of an answer at once)

def kl_rows(lp, lq):
    """KL(p || q) per row of [rows, vocab] log-probability tensors."""
    return (lp.exp() * (lp - lq)).sum(-1)


def alpha_rows(lo, lf, ls, lo_bound: float = -1.0, hi_bound: float = 2.0, coarse: int = 61, fine: int = 41):
    """Per-row alpha minimising KL(ls || log_softmax((1-a) lo + a lf)): a coarse grid, then a fine one around each
    row's best (the objective is convex in a)."""
    import torch
    def objective(alphas):  # alphas [rows, g] -> [rows, g]
        out = []
        for j in range(alphas.shape[1]):
            a = alphas[:, j:j + 1]
            out.append(kl_rows(ls, torch.log_softmax((1 - a) * lo + a * lf, -1)))
        return torch.stack(out, 1)
    rows = lo.shape[0]
    grid = torch.linspace(lo_bound, hi_bound, coarse, device=lo.device).expand(rows, coarse)
    best = grid.gather(1, objective(grid).argmin(1, keepdim=True))
    step = (hi_bound - lo_bound) / (coarse - 1)
    fine_grid = best + torch.linspace(-step, step, fine, device=lo.device)
    return fine_grid.gather(1, objective(fine_grid).argmin(1, keepdim=True)).squeeze(1)


def hybrid_cache(old_cache, new_cache, fresh_layers: int):
    """old_cache with its first `fresh_layers` layers replaced by new_cache's (deep copies)."""
    cache = copy.deepcopy(old_cache)
    for i in range(fresh_layers):
        cache.layers[i] = copy.deepcopy(new_cache.layers[i])
    return cache


def _hidden(model, ids, past=None):
    out = model.get_decoder()(input_ids=ids, past_key_values=past, use_cache=True)
    return out.last_hidden_state[0], out.past_key_values


def _logits(model, hidden, temperature: float):
    """The FP32 head on the final hidden states, as the trainer computes it (trainer/logprobs.py)."""
    return (hidden.float() @ model.get_output_embeddings().weight.float().T) / temperature


def teacher_force(old, new, prompt: list[int], answer: list[int], n: int, offsets: int, temperature: float = 1.0,
                  fresh_layers: list[int] = ()) -> dict:
    """Log-probabilities of the next answer token at answer positions n .. n+offsets-1 under OLD, FRESH, STALE and
    each HYBRID(l); returns per-position metrics (numpy arrays). All three go through the same computation (the
    prefix's KV from a cached forward, the continuation over it, the FP32 head), so at identical weights they agree
    bit for bit and any difference is the weights' doing, not the kernels'."""
    import torch
    dev = next(old.parameters()).device
    p, m = len(prompt), min(offsets, len(answer) - n)
    if n < 1 or m < 1:
        raise ValueError("need at least one token before and after the swap")
    ids = torch.tensor([list(prompt) + list(answer[:n + m])], device=dev)
    prefix = ids[:, :p + n - 1]  # KV of the prompt and y[:n-1]; y[n-1] is fed to the weights that continue
    rest = ids[:, p + n - 1:p + n - 1 + m]
    _, old_cache = _hidden(old, prefix)
    _, new_cache = _hidden(new, prefix)
    lo_all = _logits(old, _hidden(old, rest, copy.deepcopy(old_cache))[0], temperature)
    lf_all = _logits(new, _hidden(new, rest, copy.deepcopy(new_cache))[0], temperature)
    stale = {0: _logits(new, _hidden(new, rest, copy.deepcopy(old_cache))[0], temperature)}
    for l in fresh_layers:
        stale[l] = _logits(new, _hidden(new, rest, hybrid_cache(old_cache, new_cache, l))[0], temperature)
    lo, lf = torch.log_softmax(lo_all, -1), torch.log_softmax(lf_all, -1)
    out = {}
    for l, logits in stale.items():
        ls = torch.log_softmax(logits, -1)
        kl_fs, kl_fo = kl_rows(lf, ls), kl_rows(lf, lo)
        key = "" if l == 0 else f"_l{l}"
        out[f"kl_fresh_stale{key}"] = kl_fs.cpu().numpy()
        out[f"R{key}"] = (kl_fs / kl_fo).cpu().numpy()
        a = alpha_rows(lo, lf, ls)
        out[f"alpha{key}"] = a.cpu().numpy()
        # what the best mixture leaves unexplained, relative to the update: 0 when STALE lies on the OLD-FRESH line
        # (then R = (1 - alpha)^2 to first order), about R when it sits off the line as far as FRESH is from OLD
        mix = torch.log_softmax((1 - a[:, None]) * lo + a[:, None] * lf, -1)
        out[f"resid{key}"] = (kl_rows(ls, mix) / kl_fo).cpu().numpy()
    out["kl_fresh_old"] = kl_rows(lf, lo).cpu().numpy()
    out["drift"] = kl_rows(lo, lf).cpu().numpy()
    out["offset"] = np.arange(m)
    return out


def cut_at_stop(tokens: list[int], tok, stops: list[str]) -> list[int]:
    """The answer before its first stop string, as vLLM returns it (the stop's text is not part of the answer)."""
    text = tok.decode(tokens)
    starts = [i for i in (text.find(s) for s in stops) if i >= 0]
    if not starts:
        return tokens
    cut = min(starts)
    keep = 0
    for j in range(1, len(tokens) + 1):
        if len(tok.decode(tokens[:j])) > cut:
            break
        keep = j
    return tokens[:keep]


# -- the driver over checkpoint pairs

def sample_answers(model, tok, prompts: list[list[int]], device, max_new_tokens: int, temperature: float,
                   stops: list[str]) -> list[list[int]]:
    """Answers sampled from `model` in one left-padded batch, as in-flight answers are: each ends at its first EOS
    or stop string (neither included)."""
    import torch
    width = max(map(len, prompts))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.tensor([[pad] * (width - len(p)) + list(p) for p in prompts], device=device)
    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
    kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature > 0 else dict(do_sample=False)
    if stops:
        kw |= dict(stop_strings=list(stops), tokenizer=tok)
    g = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, pad_token_id=pad, **kw)
    out = []
    for row in g[:, width:].tolist():
        if tok.eos_token_id in row:
            row = row[:row.index(tok.eos_token_id)]
        out.append(cut_at_stop(row, tok, stops))
    return out


def snapshots(directory: Path) -> dict[int, Path]:
    """{version: snapshot dir} for the trainer's snapshots (<tag>-v<version>/model.safetensors)."""
    out = {}
    for d in Path(directory).iterdir():
        _, version = parse_snapshot(d.name)
        if version is not None and (d / "model.safetensors").exists():
            out[version] = d
    return dict(sorted(out.items()))


def load_snapshot(base: str, path: Path | None, device: str, dtype):
    """The base model with a snapshot's parameters loaded over it (a tied LM head is not stored; it follows)."""
    import torch
    from transformers import AutoModelForCausalLM

    from halfwritten.revisions import model_revision
    model = AutoModelForCausalLM.from_pretrained(base, revision=model_revision(base), dtype=dtype).to(device).eval()
    if path is not None:
        from safetensors.torch import load_file
        missing, unexpected = model.load_state_dict(load_file(str(path / "model.safetensors"), device=device),
                                                    strict=False)
        if unexpected or set(missing) - {"lm_head.weight"}:
            raise ValueError(f"{path}: missing {missing[:3]}, unexpected {unexpected[:3]}")
    torch.set_grad_enabled(False)
    return model


def phase1_table(directory: Path) -> list[dict]:
    """Per lag: drift mean and median, R as a ratio of means (overall and by tokens since the swap), alpha and resid
    medians, and the share of KL(fresh||stale) that recomputing the first l layers removes (ratio of means). Every
    ratio carries a 95% bootstrap interval over prompts (`*_ci95`, "lo-hi")."""
    import pyarrow.parquet as pq
    rows = pq.read_table(directory / "positions.parquet").to_pylist()
    by = defaultdict(list)
    for r in rows:
        by[r["lag"]].append(r)
    layers = sorted({int(k.split("_l")[-1]) for k in rows[0] if k.startswith("kl_fresh_stale_l")}) if rows else []
    out = []
    for k in sorted(by):
        rs = by[k]
        col = lambda key, sel=rs: np.array([x[key] for x in sel if x.get(key) is not None], dtype=float)  # noqa
        fs, fo, off = col("kl_fresh_stale"), col("kl_fresh_old"), col("offset")
        e = {"lag": k, "positions": len(rs), "drift_mean": float(col("drift").mean()),
             "drift_median": float(np.median(col("drift")))}
        if k > 0 and fo.mean() > 0:
            e["R"] = float(fs.mean() / fo.mean())
            lo, hi = _cluster_ci(rs, "kl_fresh_stale", "kl_fresh_old")
            e["R_ci95"] = f"{lo:.2f}-{hi:.2f}"
            for lo, hi in ((0, 8), (8, 32), (32, 10 ** 9)):
                m = (off >= lo) & (off < hi)
                key = f"R_{lo}_{min(hi, 999)}"
                e[key] = float(fs[m].mean() / fo[m].mean()) if m.any() else None
                if m.any():
                    a, b = _cluster_ci([x for x in rs if lo <= x["offset"] < hi], "kl_fresh_stale", "kl_fresh_old")
                    e[key + "_ci95"] = f"{a:.2f}-{b:.2f}"
            e["alpha_median"] = float(np.nanmedian(col("alpha")))
            if "resid" in rs[0]:
                e["resid_median"] = float(np.nanmedian(col("resid")))
            for l in layers:
                e[f"removed_l{l}"] = float(1 - col(f"kl_fresh_stale_l{l}").mean() / fs.mean()) if fs.mean() else None
                if fs.mean():  # the share removed is sum(fs - fs_l) / sum(fs); the interval resamples prompts as R's
                    for x in rs:
                        x[f"_removed_l{l}"] = x["kl_fresh_stale"] - x[f"kl_fresh_stale_l{l}"]
                    a, b = _cluster_ci(rs, f"_removed_l{l}", "kl_fresh_stale")
                    e[f"removed_l{l}_ci95"] = f"{a:.2f}-{b:.2f}"
        out.append(e)
    return out


def _cluster_ci(rs: list[dict], num: str, den: str, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of sum(num) / sum(den), resampling prompts (positions of one answer are not
    independent, and a few positions carry most of the KL)."""
    sums = defaultdict(lambda: [0.0, 0.0])
    for r in rs:
        g = sums[r.get("prompt", 0)]
        g[0] += r[num]
        g[1] += r[den]
    a = np.array(list(sums.values()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    ratios = a[idx, 0].sum(1) / np.maximum(a[idx, 1].sum(1), 1e-30)
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def markdown(table: list[dict]) -> str:
    keys = list(dict.fromkeys(k for e in table for k in e))
    fmt = lambda v: "" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))  # noqa: E731
    return "\n".join(["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] +
                     ["| " + " | ".join(fmt(e.get(k)) for k in keys) + " |" for e in table])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", metavar="DIR", default=None,
                    help="no GPU: print the per-lag table of a finished study's positions.parquet, and stop")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--snapshot-dir", default=None, help="a row's work/ directory of step snapshots")
    ap.add_argument("--lags", default="1,2,4,8,16,32")
    ap.add_argument("--old-version", type=int, default=None, help="the old snapshot's version (default: the first)")
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--swap", default="0.25,0.5,0.75", help="swap points as fractions of each answer")
    ap.add_argument("--positions", type=int, default=96, help="tokens scored after each swap point")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--fresh-layers", default="",
                    help="HYBRID: counts of leading layers given fresh KV, e.g. 1,4,14,27")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out", default="phase1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.table:
        print(markdown(phase1_table(Path(a.table))))
        return
    if not a.snapshot_dir:
        ap.error("--snapshot-dir is required (or --table DIR)")
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer

    from halfwritten.tasks import STOPS, encode, gsm8k_examples
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    snaps = snapshots(Path(a.snapshot_dir))
    start = a.old_version if a.old_version is not None else min(snaps)
    from halfwritten.revisions import model_revision
    tok = AutoTokenizer.from_pretrained(a.base, revision=model_revision(a.base))
    rows = gsm8k_examples("test")[:a.prompts]
    prompts = [ids for ids, _ in encode(tok, "gsm8k", rows)]
    dtype, dev = getattr(torch, a.dtype), "cuda"
    old = load_snapshot(a.base, snaps[start], dev, dtype)
    torch.manual_seed(a.seed)
    answers = sample_answers(old, tok, prompts, dev, a.max_new_tokens, a.temperature, STOPS["gsm8k"])
    layers = [int(x) for x in a.fresh_layers.split(",") if x]
    records = []
    for k in [int(x) for x in a.lags.split(",")]:
        if start + k not in snaps:
            continue
        new = load_snapshot(a.base, snaps[start + k], dev, dtype)
        for i, (pr, ans) in enumerate(zip(prompts, answers)):
            for frac in [float(x) for x in a.swap.split(",")]:
                n = max(1, int(frac * len(ans)))
                if len(ans) - n < 1:
                    continue
                r = teacher_force(old, new, pr, ans, n, a.positions, a.temperature, layers)
                for j in range(len(r["offset"])):
                    records.append({"lag": k, "prompt": i, "swap": frac, "n": n, "prefix": len(pr) + n,
                                    **{key: float(v[j]) for key, v in r.items()}})
        del new
        torch.cuda.empty_cache()
        at = [x for x in records if x["lag"] == k]
        print(json.dumps({"lag": k, "positions": len(at),  # progress only; summary.json holds the published table
                          "drift_mean": float(np.mean([x["drift"] for x in at]))}), flush=True)
    pq.write_table(pa.Table.from_pylist(records), out / "positions.parquet")
    from halfwritten.results import host_gpu_label
    (out / "summary.json").write_text(json.dumps({"base": a.base, "start": start, "snapshots": str(a.snapshot_dir),
                                                  "gpu": host_gpu_label(),
                                                  "args": vars(a), "by_lag": phase1_table(out)}, indent=1))


if __name__ == "__main__":
    main()
