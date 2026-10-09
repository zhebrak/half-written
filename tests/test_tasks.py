"""Tasks: seeded prompt streams and the GSM8K, MATH and simulator rewards."""
import pytest

from halfwritten.rewards import gsm8k_score, sim_reward
from halfwritten.tasks import PromptStream, synthetic


def test_prompt_stream_is_seeded_and_reshuffles_each_epoch():
    s1, s2 = synthetic(8, 4, seed=1), synthetic(8, 4, seed=1)
    first = [next(s1)[2] for _ in range(16)]
    assert first == [next(s2)[2] for _ in range(16)]  # the same seed asks every arm the same prompts in order
    assert sorted(first[:8]) == list(range(8)) and first[:8] != first[8:]
    with pytest.raises(ValueError):
        PromptStream([])


def test_gsm8k_score():
    assert gsm8k_score("Natalia sold ... #### 72", "so the answer is 72.") == 1.0
    assert gsm8k_score("72", "#### 1,234") == 0.0
    assert gsm8k_score("#### 1234", "Total: 1,234") == 1.0
    assert gsm8k_score("#### -3", "it is -3") == 1.0
    assert gsm8k_score("#### 5", "no number here") == 0.0


def test_sim_reward_is_deterministic_and_length_cut_is_wrong():
    r = sim_reward(0.5)
    draws = [r(i, (i, i + 1), "stop") for i in range(200)]
    assert draws == [r(i, (i, i + 1), "stop") for i in range(200)]
    assert 0.3 < sum(draws) / 200 < 0.7
    assert r(1, (1, 2), "length") == 0.0


def test_math_answers_and_scoring():
    from halfwritten.tasks import TEMPLATES, last_boxed
    assert last_boxed(r"so $x = \boxed{\frac{1}{2}}$ and \boxed{3}") == "3"
    assert last_boxed(r"\boxed{\left( 3, \frac{\pi}{2} \right)}") == r"\left( 3, \frac{\pi}{2} \right)"
    assert last_boxed("no box") is None
    rendered = TEMPLATES["math"].format(question="q")
    assert "{question}" not in rendered and r"\boxed{}" in rendered
    pytest.importorskip("math_verify")
    from halfwritten.rewards import math_score
    assert math_score(r"\frac{1}{2}", r"The answer is $\boxed{0.5}$.") == 1.0
    assert math_score("3", r"so \boxed{4}") == 0.0


def test_prompts_that_cannot_fit_are_excluded_not_cut():
    """vLLM 0.30 refuses prompt + max_tokens > max_model_len. MATH at 4,096 - 3,000 had 4 such training prompts
    (external review F06): training, the pass@k filter and evaluation exclude them by the same rule."""
    from halfwritten.tasks import too_long
    items = [((1,) * 1096, "a"), ((1,) * 1097, "b"), ((1,) * 1675, "c"), ((1,) * 10, "d")]
    assert too_long(items, max_tokens=3000, max_model_len=4096) == [1, 2]
    assert too_long(items, max_tokens=1000, max_model_len=4096) == []


def test_a_request_that_cannot_fit_stops_the_run_before_reaching_the_engine():
    from tests.simutil import quick_sim
    summary, *_ = quick_sim("D", budget_s=5.0, max_model_len=40, max_tokens=30)  # 16-token prompts + 30 > 40
    assert not summary.valid and "leaves no room" in summary.error


def test_math_dev_holds_only_problems_that_fit(monkeypatch):
    """A MATH dev item too long for 4,096 - 3,000 made every MATH dev evaluation refuse to start (code review of
    2026-10-02): dev is drawn only from problems that fit, at MATH-500's level counts."""
    import transformers

    from halfwritten import tasks

    class Tok:  # one token per word
        def __call__(self, text, add_special_tokens=False):
            return type("E", (), {"input_ids": text.split()})()

    def rows(split="train", levels=(3, 4, 5)):
        out = []
        for level in levels:
            out += [(f"L{level} problem {i}" + (" word" * 2000 if i % 7 == 0 else ""), "1") for i in range(300)]
        return out
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tok())
    monkeypatch.setattr(tasks, "math_examples", rows)
    tasks.dev_questions.cache_clear()
    try:
        dev = tasks.dev_questions("math")
    finally:
        tasks.dev_questions.cache_clear()
    assert len(dev) == 500 and not any("word" in q for q in dev)
    assert {lvl: sum(q.startswith(f"L{lvl} ") for q in dev) for lvl in (1, 2, 3, 4, 5)} == tasks.MATH500_LEVELS
