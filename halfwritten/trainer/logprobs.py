"""Token log-probabilities from the final hidden states with an FP32 head, a chunk of tokens at a time.

Qwen3-0.6B ties its LM head to the input embedding (151,936 x 1,024). "FP32 head" means fp32 logits from the
bf16-rounded hidden states and embedding (what the engine holds), accumulated in fp32, outside autocast. A 3.5k-token
answer's fp32 logits are about 2 GB, so the softmax is computed per chunk and each chunk is recomputed in the backward
pass instead of stored (activation checkpointing). Logits are divided by the sampling temperature, as vLLM's processed
logprobs are.
"""
import torch
from torch.utils.checkpoint import checkpoint


def _chunk_logprobs(hidden: torch.Tensor, weight: torch.Tensor, tokens: torch.Tensor, temperature: float):
    # Outside autocast, which would cast this matmul's inputs back to bf16 and round the logits (up to 0.07 in log p,
    # seen as most of the first noise floor). The inputs are rounded to bf16 as the engine holds them (bf16 weights,
    # bf16 hidden states) and the products accumulate in fp32: vLLM's head_dtype=float32 head computes the same
    # function. Products of bf16 values are exact in TF32, so tensor cores lose nothing here (model.py allows TF32).
    with torch.autocast(hidden.device.type, enabled=False):
        logits = hidden.to(torch.bfloat16).float() @ weight.to(torch.bfloat16).float().T
    if temperature != 1.0:
        logits = logits / temperature
    return logits.log_softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)


def token_logprobs(hidden: torch.Tensor, weight: torch.Tensor, tokens: torch.Tensor, *, temperature: float = 1.0,
                   chunk: int = 2048, recompute: bool = True) -> torch.Tensor:
    """hidden [T, H] (the states that predict each token), weight [V, H], tokens [T] -> log p [T] in fp32."""
    out = []
    for i in range(0, hidden.shape[0], chunk):
        args = (hidden[i:i + chunk], weight, tokens[i:i + chunk], temperature)
        if recompute and torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
            out.append(checkpoint(_chunk_logprobs, *args, use_reentrant=False))
        else:
            out.append(_chunk_logprobs(*args))
    return torch.cat(out) if out else hidden.new_zeros(0, dtype=torch.float32)
