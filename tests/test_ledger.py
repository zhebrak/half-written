import numpy as np
import pyarrow as pa
import pytest

from halfwritten.ledger import Ledger, LedgerError
from halfwritten.types import CONTEXT_CODE, Segment


def test_segments_tile_the_answer():
    led = Ledger()
    led.open("s", 3)
    led.extend("s", 5)
    led.cut("s", 5, 4, "stale_kv")
    led.extend("s", 2)
    led.cut("s", 7, 6, "stale_kv")
    led.extend("s", 3)
    assert led.close("s", "stop") == (Segment(0, 5, 3, "pure"), Segment(5, 7, 4, "stale_kv"),
                                      Segment(7, 10, 6, "stale_kv"))
    assert led.token_versions("s").tolist() == [3] * 5 + [4] * 2 + [6] * 3


def test_cut_before_any_token_relabels_and_stays_pure():
    """An answer that has not started when the weights change (queued in B, or admitted just before a D pause with
    no token delivered) is entirely on the new version."""
    led = Ledger()
    led.open("s", 0)
    led.cut("s", 0, 1, "stale_kv")
    led.extend("s", 4)
    assert led.close("s", "stop") == (Segment(0, 4, 1, "pure"),)


def test_second_cut_without_new_tokens_relabels_keeps_context():
    led = Ledger()
    led.open("s", 0)
    led.extend("s", 3)
    led.cut("s", 3, 1, "stale_kv")
    led.cut("s", 3, 2, "stale_kv")
    led.extend("s", 1)
    assert led.close("s", "stop") == (Segment(0, 3, 0, "pure"), Segment(3, 4, 2, "stale_kv"))


def test_cut_only_at_the_delivered_length():
    led = Ledger()
    led.open("s", 0)
    led.extend("s", 3)
    with pytest.raises(LedgerError, match="cut at 2 but 3"):
        led.cut("s", 2, 1, "stale_kv")


def test_versions_never_decrease():
    led = Ledger()
    led.open("s", 5)
    led.extend("s", 1)
    with pytest.raises(LedgerError, match="version 4 after 5"):
        led.cut("s", 1, 4, "recomputed")


def test_trailing_empty_segment_is_dropped():
    led = Ledger()
    led.open("s", 0)
    led.extend("s", 3)
    led.cut("s", 3, 1, "recomputed")
    assert led.close("s", "stop") == (Segment(0, 3, 0, "pure"),)


def test_discard_counts_waste_and_opens_the_next_attempt():
    led = Ledger()
    assert led.open("s", 0, group="g") == 0
    led.extend("s", 7)
    assert led.discard("s", "abort") == 7
    assert led.open("s", 1, group="g") == 1
    led.extend("s", 2)
    led.close("s", "stop")
    assert led.attempt("s") == 1
    assert led.discards()[0] | {"t_open": 0} == {"sid": "s", "group": "g", "attempt": 0, "reason": "abort",
                                                 "tokens": 7, "versions": [0], "t_open": 0, "t_discard": 0.0}
    assert led.counts() == {"open": 0, "closed": 1, "consumed": 0, "discarded": 1, "attempts": 2,
                            "discarded_tokens": 7}


def test_consume_returns_lag_once():
    led = Ledger()
    led.open("s", 2)
    led.extend("s", 2)
    led.cut("s", 2, 3, "recomputed")
    led.extend("s", 1)
    led.close("s", "stop")
    assert led.consume("s", 4).tolist() == [2, 2, 1]
    with pytest.raises(LedgerError, match="consumed twice"):
        led.consume("s", 5)


def test_negative_lag_is_an_error():
    led = Ledger()
    led.open("s", 3)
    led.extend("s", 1)
    led.close("s", "stop")
    with pytest.raises(LedgerError, match="negative lag"):
        led.consume("s", 2)


def test_kv_depth_counts_only_stale_tokens_from_the_prompt_version():
    led = Ledger()
    led.open("d", 1)
    led.extend("d", 2)
    led.cut("d", 2, 2, "stale_kv")
    led.extend("d", 1)
    led.cut("d", 3, 4, "stale_kv")
    led.extend("d", 1)
    led.close("d", "stop")
    assert led.kv_depth("d").tolist() == [0, 0, 1, 3]
    led.open("e", 1)
    led.extend("e", 2)
    led.cut("e", 2, 2, "recomputed")
    led.extend("e", 1)
    led.close("e", "stop")
    assert led.kv_depth("e").tolist() == [0, 0, 0]
    assert led.token_context_codes("d").tolist() == [CONTEXT_CODE["pure"]] * 2 + [CONTEXT_CODE["stale_kv"]] * 2


def test_rfc_spans():
    led = Ledger()
    led.open("s", 6)
    led.extend("s", 57)
    led.cut("s", 57, 7, "stale_kv")
    led.extend("s", 3)
    led.close("s", "stop")
    assert led.rfc_spans("s") == [{"version": "v6", "start": 0, "end": 57}, {"version": "v7", "start": 57, "end": 60}]


def test_table_roundtrips_through_parquet(tmp_path):
    import pyarrow.parquet as pq
    led = Ledger()
    led.open("s", 0, group="g")
    led.extend("s", 2)
    led.cut("s", 2, 1, "stale_kv")
    led.extend("s", 1)
    led.close("s", "length", t=1.5)
    led.annotate("s", reward=1.0)
    pq.write_table(led.table(), tmp_path / "ledger.parquet")
    row = pq.read_table(tmp_path / "ledger.parquet").to_pylist()[0]
    assert row["finish"] == "length" and row["reward"] == 1.0 and row["tokens"] == 3
    assert row["segments"] == [{"start": 0, "end": 2, "version": 0, "context": "pure"},
                               {"start": 2, "end": 3, "version": 1, "context": "stale_kv"}]
    assert isinstance(led.discards_table(), pa.Table)


def test_misuse_raises():
    led = Ledger()
    with pytest.raises(LedgerError):
        led.extend("nope", 1)
    led.open("s", 0)
    with pytest.raises(LedgerError, match="already open"):
        led.open("s", 0)
    with pytest.raises(LedgerError, match="unknown context"):
        led.cut("s", 0, 1, "fresh")
    assert np.array_equal(led.token_versions("s"), np.zeros(0))
