"""Advantages and the loss normaliser, identical in every arm (PLAN.md §4.4).

Each answer's advantage is its reward minus its group's mean, divided by the standard deviation of all rewards in
the batch (ScaleRL's batch-level normalisation). A group whose answers all scored the same has zero advantage
everywhere; it is consumed and counted, never resampled, so generation per step does not depend on accuracy. The
loss is normalised by the constant 1/(B*G*max_tokens) rather than by token counts: arms consume answers of different
lengths (lag and length are coupled), and a length-dependent normaliser would scale gradients by regime.
"""
import numpy as np


def zero_variance(rewards) -> bool:
    r = np.asarray(rewards, dtype=float)
    return bool(r.size == 0 or np.ptp(r) == 0)


def advantages(groups: list) -> list[np.ndarray]:
    """Per group, per answer: (reward - group mean) / batch std; all zeros when every reward in the batch is equal."""
    rewards = [np.asarray(g, dtype=float) for g in groups]
    std = float(np.concatenate(rewards).std()) if rewards else 0.0
    if std == 0:
        return [np.zeros_like(r) for r in rewards]
    return [(r - r.mean()) / std for r in rewards]


def loss_scale(B: int, G: int, max_tokens: int) -> float:
    return 1.0 / (B * G * max_tokens)
