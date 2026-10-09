"""Phase 1's statistics without a model: alpha and the share of the update a stale cache does not apply."""
import numpy as np
import pytest

from bench.stale_kv import fit_alpha, kl, log_normalize, mix, ratio_not_applied


def _dists(seed=0, vocab=50):
    rng = np.random.default_rng(seed)
    old = log_normalize(rng.normal(size=vocab))
    fresh = log_normalize(old + rng.normal(scale=0.5, size=vocab))
    return old, fresh


def test_alpha_is_1_when_stale_kv_is_harmless_and_0_when_it_undoes_the_update():
    old, fresh = _dists()
    assert fit_alpha(old, fresh, fresh) == pytest.approx(1.0, abs=1e-4)
    assert fit_alpha(old, fresh, old) == pytest.approx(0.0, abs=1e-4)
    assert ratio_not_applied(old, fresh, fresh) == pytest.approx(0.0)
    assert ratio_not_applied(old, fresh, old) == pytest.approx(1.0)


def test_alpha_recovers_a_geometric_mixture():
    old, fresh = _dists(1)
    for a in (0.3, 0.8, 1.2):
        assert fit_alpha(old, fresh, mix(old, fresh, a)) == pytest.approx(a, abs=1e-4)
    assert kl(old, old) == pytest.approx(0.0)
