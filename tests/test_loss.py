"""The trainer's numerics on CPU: CISPO's gradient, the chunked FP32 head, and micro-batch invariance."""
import pytest

torch = pytest.importorskip("torch")

from halfwritten.advantages import loss_scale  # noqa: E402
from halfwritten.trainer.logprobs import token_logprobs  # noqa: E402
from halfwritten.trainer.loss import cispo_loss  # noqa: E402
from halfwritten.trainer.pack import pack  # noqa: E402


def test_cispo_gradient_is_the_truncated_is_policy_gradient():
    logp = torch.log(torch.tensor([0.5, 0.2, 0.9])).requires_grad_()
    mu = torch.log(torch.tensor([0.5, 0.01, 0.9]))  # the second token: ratio 20, truncated to 5
    adv = torch.tensor([1.0, 2.0, -1.0])
    loss, stats = cispo_loss(logp, mu, adv, eps_max=5.0, scale=0.5)
    loss.backward()
    assert logp.grad.tolist() == pytest.approx([-0.5 * 1 * 1.0, -0.5 * 5 * 2.0, -0.5 * 1 * -1.0])
    assert stats["truncated"] == pytest.approx(1 / 3)


def test_cispo_matches_trls_formula():
    """TRL v1.14.1 _compute_loss, loss_type="cispo": -clamp(exp(logp - old), max=epsilon_high).detach() * A * logp."""
    g = torch.Generator().manual_seed(0)
    logp, mu, adv = -torch.rand(50, generator=g) * 3, -torch.rand(50, generator=g) * 3, torch.randn(50, generator=g)
    ours, _ = cispo_loss(logp, mu, adv, eps_max=5.0, scale=1.0)
    trl = (-torch.clamp(torch.exp(logp - mu), max=5.0).detach() * adv * logp).sum()
    assert ours.item() == pytest.approx(trl.item())


def test_chunked_fp32_head_equals_the_full_computation_with_gradients():
    g = torch.Generator().manual_seed(1)
    hidden = torch.randn(37, 16, generator=g, dtype=torch.bfloat16).requires_grad_()
    weight = torch.randn(101, 16, generator=g, dtype=torch.bfloat16).requires_grad_()
    tokens = torch.randint(0, 101, (37,), generator=g)
    full = (hidden.float() @ weight.float().T / 0.7).log_softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)
    chunked = token_logprobs(hidden, weight, tokens, temperature=0.7, chunk=8)
    assert chunked.dtype == torch.float32
    assert torch.allclose(chunked, full, atol=1e-6)
    g_full = torch.autograd.grad(full.sum(), (hidden, weight))
    g_chunk = torch.autograd.grad(chunked.sum(), (hidden, weight))
    for a, b in zip(g_full, g_chunk):  # bf16 gradients, accumulated over 5 chunks instead of once
        assert torch.allclose(a.float(), b.float(), rtol=3e-2, atol=1e-2)


def test_the_loss_of_a_batch_is_the_sum_of_its_micro_batches():
    g = torch.Generator().manual_seed(2)
    lengths = [7, 3, 12, 5, 9, 1]
    logps = [(-torch.rand(n, generator=g)).requires_grad_() for n in lengths]
    mus = [-torch.rand(n, generator=g) for n in lengths]
    advs = [torch.full((n,), float(a)) for n, a in zip(lengths, torch.randn(len(lengths), generator=g))]
    scale = loss_scale(B=3, G=2, max_tokens=12)
    whole, _ = cispo_loss(torch.cat(logps), torch.cat(mus), torch.cat(advs), 5.0, scale)
    parts = sum(cispo_loss(torch.cat([logps[i] for i in b]), torch.cat([mus[i] for i in b]),
                           torch.cat([advs[i] for i in b]), 5.0, scale)[0] for b in pack(lengths, budget=16))
    assert whole.item() == pytest.approx(parts.item())


def test_pack_places_every_sequence_once_within_the_budget():
    lengths = [900, 30, 400, 400, 2000, 120, 64, 64, 64]
    bins = pack(lengths, budget=1024)
    assert sorted(i for b in bins for i in b) == list(range(len(lengths)))
    for b in bins:
        assert len(b) * max(lengths[i] for i in b) <= 1024 or len(b) == 1
    assert pack(lengths, 1024) == bins


def test_head_is_fp32_even_under_autocast_and_matches_the_engines_rounding():
    """Autocast casts a matmul's inputs to bf16 and rounds its output: the head must opt out. Its inputs are rounded
    to bf16 as the engine holds them, its products accumulated in fp32 (vLLM's head_dtype=float32)."""
    import torch

    from halfwritten.trainer.logprobs import token_logprobs
    torch.manual_seed(0)
    h = torch.randn(6, 64, requires_grad=True)
    w = torch.randn(1000, 64, requires_grad=True)
    t = torch.randint(0, 1000, (6,))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = token_logprobs(h, w, t, chunk=4)
    ref = (h.detach().bfloat16().double() @ w.detach().bfloat16().double().T).log_softmax(-1)
    ref = ref.gather(-1, t[:, None]).squeeze(-1)
    assert got.dtype == torch.float32 and float((got.double() - ref).abs().max()) < 1e-5
    got.sum().backward()  # the rounding passes gradients straight through to the fp32 master
    assert h.grad is not None and w.grad is not None and w.grad.dtype == torch.float32 and w.grad.abs().sum() > 0
