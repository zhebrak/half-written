"""Rewards: a function (answer, generated token ids, finish) -> float. Text tasks wrap a string scorer with the
tokenizer's decode. No format reward anywhere; an answer cut by max_tokens scores what its text scores.
"""
import re
from collections.abc import Callable, Sequence
from typing import Any

from .hashing import stable_key as _key

Reward = Callable[[Any, Sequence[int], str], float]


def sim_reward(p: float = 0.5, seed: int = 0, *, forced_lengths: bool = False) -> Reward:
    """For the simulator: correct with probability p, deterministically in the tokens, so some groups have zero
    variance; an answer cut at max_tokens is wrong unless its length was prescribed by the systems map."""
    def reward(answer, token_ids, finish) -> float:
        if finish == "length" and not forced_lengths:
            return 0.0
        return float((_key(seed, answer, len(token_ids), *token_ids[:8]) % 10_000) < p * 10_000)
    return reward


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gsm8k_score(answer: str, text: str) -> float:
    """Exact numeric match of the final answer: the number after '####' if the model wrote one, else its last
    number. `answer` is GSM8K's reference solution (its own '#### <n>') or the bare number."""
    def final(s: str) -> float | None:
        tail = s.rsplit("####", 1)[1] if "####" in s else s
        nums = _NUMBER.findall(tail)
        if not nums:
            return None
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    want, got = final(str(answer)), final(text)
    return float(want is not None and got is not None and abs(want - got) < 1e-6)


def math_score(answer: str, text: str) -> float:
    """MATH: math-verify's equivalence of the answer's last \\boxed{} with the reference's. math-verify bounds its
    own parsing with signals, which work only in a process's main thread, so runs score MATH in a process pool
    (launch.py) with a timeout that scores 0 and is counted."""
    from math_verify import parse, verify
    return float(verify(parse("\\boxed{" + answer + "}"), parse(text)))

