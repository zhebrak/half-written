"""CISPO (ScaleRL's loss) against the engine's behaviour probabilities, identical in every arm (PLAN.md §4.4).

    loss = -scale * sum_t sg(min(pi_t / mu_t, eps_max)) * A_t * log pi_t

pi is the trainer's probability of each sampled token at the weights the step starts from, mu the engine's processed
probability when it sampled it. The weight is stop-gradient, so the gradient is a truncated-IS policy gradient (TRL's
`loss_type="cispo"` with `epsilon_high` as the cap). mu is what actually sampled the token, stale-KV tokens included,
so the ratio stays exact in arm D; D's cost shows up as more truncation and a lower ESS. `scale` is the constant
1/(B*G*max_tokens) (advantages.loss_scale), so the loss of a batch is the sum of its micro-batches' losses.
"""
import torch


def cispo_loss(logp: torch.Tensor, mu: torch.Tensor, adv: torch.Tensor, eps_max: float, scale: float,
               *, stats: bool = True):
    """logp: trainer log-probabilities of the sampled tokens (with grad), mu: engine log-probabilities, adv: each
    token's advantage (its answer's). Returns (loss, stats)."""
    ratio = torch.exp(logp.detach() - mu)
    weight = torch.clamp(ratio, max=eps_max)
    loss = -(weight * adv * logp).sum() * scale
    metrics = {"tokens": logp.numel(), "truncated": float((ratio > eps_max).float().mean()) if logp.numel() else 0.0,
               "ratio_mean": float(ratio.mean()) if logp.numel() else float("nan")} if stats else {}
    return loss, metrics
