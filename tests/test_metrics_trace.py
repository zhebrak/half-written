import json

import numpy as np
import pytest

from halfwritten import metrics
from halfwritten.advantages import advantages, loss_scale, zero_variance
from halfwritten.trace import (
    Tracer,
    check_lanes,
    lane_fractions,
    lane_totals,
    to_chrome,
    to_table,
)
from halfwritten.types import CONTEXT_CODE


def test_equal_weights_have_full_ess_and_skew_lowers_it():
    assert metrics.ess(np.ones(10)) == pytest.approx(1.0)
    assert metrics.ess(np.array([1.0] + [0.0] * 9)) == pytest.approx(0.1)


def test_truncation_caps_weights():
    w, frac = metrics.truncated_weights(np.log(np.array([0.5, 1.0, 10.0])), cap=5.0)
    assert w.tolist() == pytest.approx([0.5, 1.0, 5.0]) and frac == pytest.approx(1 / 3)


def test_kl_estimators():
    assert metrics.k3(np.zeros(5)) == 0.0 and metrics.k1(np.zeros(5)) == 0.0
    lr = np.array([-0.2, 0.1, 0.3])
    assert metrics.k3(lr) == pytest.approx(np.mean(np.exp(lr) - 1 - lr)) and metrics.k3(lr) > 0


def test_histogram_clips_the_tail_into_the_last_bin():
    assert metrics.histogram(np.array([0, 1, 1, 7, 9]), 4) == [1, 2, 0, 2]


def test_token_stats_break_down_by_context():
    codes = np.array([CONTEXT_CODE["pure"]] * 3 + [CONTEXT_CODE["stale_kv"]] * 2)
    mu = np.full(5, -1.0)
    out = metrics.token_stats(mu + np.array([0, 0, 0, 0.5, 0.5]), mu, codes, cap=5.0)
    assert out["k3_pure"] == 0 and out["k3_stale_kv"] > 0 and out["tokens"] == 5


def test_advantages_center_per_group_and_scale_by_batch_std():
    a = advantages([[1, 0], [1, 1]])
    std = np.std([1, 0, 1, 1])
    assert a[0].tolist() == pytest.approx([0.5 / std, -0.5 / std]) and a[1].tolist() == [0, 0]
    assert all((x == 0).all() for x in advantages([[1, 1], [1, 1]]))
    assert zero_variance([1, 1]) and not zero_variance([0, 1])
    assert loss_scale(32, 8, 1024) == 1 / (32 * 8 * 1024)


def test_lanes_are_gap_free_and_export_to_chrome_and_arrow():
    t = [0.0]
    tr = Tracer(lambda: t[0], "controller")
    tr.state("engine", "generating")
    t[0] = 2.0
    tr.state("engine", "paused")
    tr.state("engine", "paused")  # repeating the current state is a no-op
    tr.counter("inflight", 3)
    tr.instant("sync", "push", version=1)
    t[0] = 3.0
    events = tr.close()
    assert lane_totals(events, "engine") == {"generating": 2.0, "paused": 1.0}
    assert lane_fractions(events, "engine")["paused"] == pytest.approx(1 / 3)
    assert check_lanes(events) == []
    chrome = to_chrome(events)
    json.dumps(chrome)
    assert {e["ph"] for e in chrome["traceEvents"]} == {"M", "X", "C", "i"}
    assert to_table(events).num_rows == len(events)


def test_check_lanes_reports_a_gap():
    from halfwritten.trace import Event
    events = [Event("state", "p", "engine", "a", 0, 1), Event("state", "p", "engine", "b", 2, 3)]
    assert check_lanes(events) and "ends at 1" in check_lanes(events)[0]
