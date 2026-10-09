import pytest

from halfwritten.ledger import Ledger
from halfwritten.regimes import REGIMES, check_ledger, resolve


def test_each_regime_makes_the_vllm_call_the_plan_names():
    calls = {name: (r.pause, r.clear_cache, r.on_abort, r.post_swap_context)
             for name, r in REGIMES.items() if r.implemented}
    assert calls == {
        "A": (None, False, None, None),
        "B": ("wait", True, None, None),
        "C1": ("abort", True, "resume_partial", "resubmitted"),
        "C2": ("abort", True, "restart", None),
        "D": ("keep", False, None, "stale_kv"),  # clear_cache must be passed as False: vLLM's default makes it E
        "E": ("keep", True, None, "recomputed"),
        "F": (None, False, None, None),
    }


def test_resolve_lora():
    assert resolve("D").weights == "full"
    assert resolve("D", lora=True).weights == "lora_inplace"
    assert resolve("D+lora").label == "D+lora"
    assert resolve("F", lora=True).weights == "lora_versioned"
    with pytest.raises(ValueError, match="lora=True"):
        resolve("F")
    with pytest.raises(ValueError, match="emulated"):
        resolve("A2")
    with pytest.raises(ValueError, match="unknown"):
        resolve("G")


def _ledger(*cuts):
    led = Ledger()
    led.open("s", 0)
    led.extend("s", 2)
    for at, v, ctx in cuts:
        led.cut("s", at, v, ctx)
        led.extend("s", 1)
    led.close("s", "stop")
    return led


def test_check_ledger_accepts_each_regimes_shape():
    assert check_ledger(resolve("B"), _ledger()) == []
    assert check_ledger(resolve("D"), _ledger((2, 1, "stale_kv"), (3, 2, "stale_kv"))) == []
    assert check_ledger(resolve("E"), _ledger((2, 1, "recomputed"))) == []
    assert check_ledger(resolve("C1"), _ledger((2, 1, "resubmitted"))) == []


def test_check_ledger_flags_the_wrong_shape():
    assert check_ledger(resolve("B"), _ledger((2, 1, "recomputed")))  # B never cuts an answer
    assert check_ledger(resolve("E"), _ledger((2, 1, "stale_kv")))  # E never reads old KV
    assert check_ledger(resolve("D"), _ledger((2, 1, "recomputed")))
