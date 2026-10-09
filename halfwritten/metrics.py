"""Per-token staleness and mismatch statistics, computed the same way in every regime (numpy only).

With r = pi/mu, pi the trainer's probability of a token and mu the engine's processed probability when it sampled it:
- truncated importance weight min(r, c) (PipelineRL eq. 5) and the fraction truncated;
- normalised effective sample size (sum w)^2 / (N sum w^2), in (0, 1] (PipelineRL eq. 6);
- KL(mu || pi) estimates over tokens sampled from mu: k1 = mean(-log r) (unbiased, may be negative) and
  k3 = mean(r - 1 - log r) (unbiased, never negative);
- lag (consuming step - sampling version) and KV depth histograms, and any statistic broken down by context.
"""
import numpy as np

from .types import CONTEXTS


def truncated_weights(log_ratio: np.ndarray, cap: float) -> tuple[np.ndarray, float]:
    w = np.exp(log_ratio)
    return np.minimum(w, cap), float((w > cap).mean()) if w.size else 0.0


def ess(w: np.ndarray) -> float:
    sq = float((w * w).sum())
    return float(w.sum()) ** 2 / (w.size * sq) if w.size and sq > 0 else float("nan")


def k1(log_ratio: np.ndarray) -> float:
    return float(-log_ratio.mean()) if log_ratio.size else float("nan")


def k3(log_ratio: np.ndarray) -> float:
    if not log_ratio.size:
        return float("nan")
    return float((np.expm1(log_ratio) - log_ratio).mean())


def histogram(values: np.ndarray, bins: int) -> list[int]:
    """Counts of 0, 1, ..., bins-2 and of >= bins-1."""
    v = np.clip(np.asarray(values, dtype=np.int64), 0, bins - 1)
    return np.bincount(v, minlength=bins).tolist()


def quantiles(values, qs=(0.5, 0.9, 0.99, 1.0)) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    if not v.size:
        return {}
    return {f"p{round(q * 100)}": float(np.quantile(v, q)) for q in qs}


def token_stats(logp: np.ndarray, mu: np.ndarray, codes: np.ndarray, cap: float) -> dict:
    """Mismatch statistics of one batch: overall and per context."""
    lr = np.asarray(logp, dtype=float) - np.asarray(mu, dtype=float)
    w, trunc = truncated_weights(lr, cap)
    out = {"tokens": int(lr.size), "k1": k1(lr), "k3": k3(lr), "ess": ess(w), "truncated": trunc}
    for i, c in enumerate(CONTEXTS):
        m = codes == i
        if m.any():
            out[f"k3_{c}"] = k3(lr[m])
            out[f"ess_{c}"] = ess(w[m])
    return out
