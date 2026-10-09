"""Tasks: the prompt stream and reward a run uses (PLAN.md §4.3).

A prompt stream yields (group id, prompt token ids, answer) in a seeded order, identical across the arms of a seed
block (PLAN.md §4.6), so arms differ in what they generate, never in what they are asked.

    sim     synthetic token-id prompts and a seeded random reward (the systems map, CPU tests)
    gsm8k   GSM8K from the Hugging Face parquet, rendered with a base-model template; exact numeric match
    math    MATH levels 3-5 (train) with math-verify; MATH-500 for evaluation (no overlap with train, checked)

Every task holds a development split out of training (dev_questions): decisions (the pilot's horizon, the seed count,
learning rates) are read on dev, and test is kept for the confirmatory grid (external review: the first decisions had
been read on GSM8K test).

Prompts are tokenised once with the run model's tokenizer. Rewards decode the answer's token ids and score the text;
TextReward is picklable (it reloads the tokenizer in a worker process), so a slow scorer can run in a process pool.
"""
import asyncio
import functools
import hashlib
import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from .hashing import stable_key as _key
from .revisions import dataset_revision, model_revision
from .rewards import gsm8k_score, math_score, sim_reward


class PromptStream:
    """Cycles through `items` (prompt ids, answer), reshuffled each epoch with a seed derived from (seed, epoch)."""

    def __init__(self, items: Sequence[tuple[tuple[int, ...], Any]], seed: int = 0):
        if not items:
            raise ValueError("no prompts")
        self.items, self.seed = list(items), seed
        self._n = 0

    def __iter__(self) -> Iterator[tuple[str, tuple[int, ...], Any]]:
        return self

    def __next__(self) -> tuple[str, tuple[int, ...], Any]:
        epoch, i = divmod(self._n, len(self.items))
        if i == 0:
            self._order = list(range(len(self.items)))
            random.Random(_key(self.seed, epoch)).shuffle(self._order)
        gid = f"g{self._n}"
        self._n += 1
        prompt, answer = self.items[self._order[i]]
        return gid, tuple(prompt), answer


def synthetic(n: int = 1024, prompt_len: int = 64, seed: int = 0, vocab: int = 151_936) -> PromptStream:
    """Prompts for the simulator: random token ids; the answer is the prompt's index (sim_reward hashes it)."""
    items = [(tuple(_key(seed, i, j) % vocab for j in range(prompt_len)), i) for i in range(n)]
    return PromptStream(items, seed)


TEMPLATES = {"gsm8k": "Question: {question}\nAnswer: Let's think step by step.",
             "math": "Problem: {question}\nSolution: Let's think step by step, and put the final answer in "
                     "\\boxed{{}}.\n"}
STOPS = {"gsm8k": ["\nQuestion:", "\n\n\n"], "math": ["\nProblem:", "\n\n\n\n"]}
SCORES = {"gsm8k": gsm8k_score, "math": math_score}


class TextReward:
    def __init__(self, tokenizer: str, task: str):
        self.tokenizer, self.task = tokenizer, task
        self._tok = None

    def __getstate__(self):
        return {"tokenizer": self.tokenizer, "task": self.task, "_tok": None}

    def __call__(self, answer, token_ids, finish) -> float:
        self.warm()
        return SCORES[self.task](answer, self._tok.decode(list(token_ids), skip_special_tokens=True))

    def warm(self):
        if self._tok is None:
            self._tok = reward_tokenizer(self.tokenizer)


@functools.lru_cache(maxsize=4)
def reward_tokenizer(model: str):
    """One tokenizer per process/model, including when a caller pickles TextReward per submission."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, revision=model_revision(model))


def _init_reward(reward, ready):
    global _worker_reward
    _worker_reward = reward
    reward.warm()
    ready.wait(timeout=120)  # no timed dispatch until every worker's tokenizer is ready


def _reward_ready():
    return True


def worker_reward(answer, token_ids, finish):
    """Process-pool entry point: only answer data crosses the queue."""
    return _worker_reward(answer, token_ids, finish)


async def score(reward, pool, answer, token_ids, finish, timeout_s: float | None = None) -> float:
    """One answer's reward: inline, or on `pool`'s warmed workers (scoring(); only answer data crosses the queue).
    With `timeout_s` a late score raises TimeoutError to the caller (the controller counts it and scores 0; the
    evaluator and the filter wait). The one dispatch for all three."""
    if pool is None:
        return float(reward(answer, token_ids, finish))
    fut = asyncio.get_running_loop().run_in_executor(pool, reward, answer, tuple(token_ids), finish)
    return float(await (asyncio.wait_for(fut, timeout_s) if timeout_s else fut))


@contextmanager
def scoring(reward, workers: int = 8):
    """MATH's signal-bound verifier runs on warmed process main threads, never the event loop."""
    if not isinstance(reward, TextReward) or reward.task != "math":
        yield reward, None
        return
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    ctx = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_reward,
                               initargs=(reward, ctx.Barrier(workers)))
    try:
        ready = [pool.submit(_reward_ready) for _ in range(workers)]
        for f in ready:
            f.result(timeout=180)
        yield worker_reward, pool
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def gsm8k_examples(split: str = "train") -> list[tuple[str, str]]:
    """(question, final answer) from openai/gsm8k's main config."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset",
                           revision=dataset_revision("openai/gsm8k"))
    t = pq.read_table(path).to_pylist()
    return [(r["question"], r["answer"].rsplit("####", 1)[1].strip()) for r in t]


def last_boxed(text: str) -> str | None:
    """The content of the last \\boxed{...} (braces balanced), as MATH solutions give their answers."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def math_examples(split: str = "train", levels: tuple[int, ...] = (3, 4, 5)) -> list[tuple[str, str]]:
    """(problem, boxed answer) from MATH (DigitalLearningGmbH/MATH-lighteval), at the given levels."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("DigitalLearningGmbH/MATH-lighteval", f"data/{split}-00000-of-00001.parquet",
                           repo_type="dataset",
                           revision=dataset_revision("DigitalLearningGmbH/MATH-lighteval"))
    keep = {f"Level {n}" for n in levels}
    out = []
    for r in pq.read_table(path).to_pylist():
        answer = last_boxed(r["solution"])
        if r["level"] in keep and answer:
            out.append((r["problem"], answer))
    return out


def math500_examples() -> list[tuple[str, str]]:
    import json as _json

    from huggingface_hub import hf_hub_download
    path = hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset",
                           revision=dataset_revision("HuggingFaceH4/MATH-500"))
    return [(r["problem"], r["answer"]) for r in map(_json.loads, open(path))]


DEV_SEED, DEV_N = 20261002, 500
# W0 is S1's step 50: arm A, B = 32, seed 0, unfiltered, so it had been sent the first (50 + 0 + 1) x 32 prompts of
# PromptStream(GSM8K train, seed 0). GSM8K dev is drawn outside them, so it is unseen by W0, as test is.
W0_SEEN = {"gsm8k": (50 + 1) * 32}
MATH500_LEVELS = {1: 43, 2: 90, 3: 105, 4: 128, 5: 134}  # MATH dev mirrors the endpoint's level mix
# MATH dev is drawn only from problems whose prompt leaves room for the MATH configs' 3,000 tokens within 4,096 (as
# rendered for the study's model): evaluation refuses an item that cannot fit (too_long), and four do not (review)
MATH_DEV_FIT = ("Qwen/Qwen3-0.6B-Base", 3000, 4096)


@functools.cache
def dev_questions(task: str) -> frozenset[str]:
    """The development split's questions, held out of every training pool (load) and evaluated as split "dev".
    GSM8K: 500 train questions outside those W0 was trained on. MATH: 500 train problems stratified to MATH-500's
    levels (levels 1-2 are outside the training pool, which loses only the level 3-5 ones), among those that fit
    the context (MATH_DEV_FIT)."""
    rng = random.Random(DEV_SEED)
    if task == "gsm8k":
        rows = gsm8k_examples("train")
        order = list(range(len(rows)))
        random.Random(_key(0, 0)).shuffle(order)  # PromptStream's first epoch at seed 0: S1's order
        seen = set(order[:W0_SEEN["gsm8k"]])
        return frozenset(rows[i][0] for i in rng.sample([i for i in range(len(rows)) if i not in seen], DEV_N))
    if task == "math":
        from transformers import AutoTokenizer
        model, max_tokens, max_model_len = MATH_DEV_FIT
        tok = AutoTokenizer.from_pretrained(model, revision=model_revision(model))
        picked = []
        for level, n in MATH500_LEVELS.items():
            rows = math_examples("train", levels=(level,))
            long = set(too_long(encode(tok, "math", rows), max_tokens, max_model_len))
            picked += [rows[i][0] for i in rng.sample([i for i in range(len(rows)) if i not in long], n)]
        return frozenset(picked)
    return frozenset()


def dev_digest(task: str) -> str | None:
    """sha256 of the dev split's sorted questions: rows and evaluations record it, and evaluation on dev checks it."""
    qs = dev_questions(task)
    return hashlib.sha256("\n".join(sorted(qs)).encode()).hexdigest() if qs else None


def dev_examples(task: str) -> list[tuple[str, str]]:
    dev = dev_questions(task)
    rows = gsm8k_examples("train") if task == "gsm8k" else math_examples("train", levels=tuple(MATH500_LEVELS))
    return [r for r in rows if r[0] in dev]


def eval_examples(task: str, split: str) -> list[tuple[str, str]]:
    """The evaluation items: split "dev" (held out of training; decisions) or "test" (GSM8K test, MATH-500; the
    confirmatory grid)."""
    if split == "dev":
        return dev_examples(task)
    if split != "test":
        raise ValueError(f"split {split!r}: dev or test")
    return gsm8k_examples("test") if task == "gsm8k" else math500_examples()


def too_long(items, max_tokens: int, max_model_len: int) -> list[int]:
    """Indices of the items whose prompt leaves no room for max_tokens within max_model_len. vLLM rejects such a
    request (prompt + max_tokens > max_model_len) rather than truncating it, so they are excluded, never cut: 4 of
    the 5,582 MATH level 3-5 training prompts (1,191-1,675 tokens against 4,096 - 3,000; external review F06)."""
    return [i for i, (prompt, _) in enumerate(items) if len(prompt) + max_tokens > max_model_len]


def training_rows(cfg, rows: list) -> list:
    """The task's training rows a run draws from: those bench.prompt_filter's pass@k filter kept (by index into the
    training rows), less the development split, held out of every pool, filtered or not."""
    keep = range(len(rows))
    if cfg.prompt_filter:
        import json as _json

        from .revisions import task_revisions
        kept = _json.loads(open(cfg.prompt_filter).read())
        if (made := kept.get("dataset_revisions")) is not None and made != (now := task_revisions(cfg.task)):
            raise ValueError(f"{cfg.prompt_filter} indexes dataset revisions {made}, not {now}")
        keep = kept["kept"]
    dev = dev_questions(cfg.task)
    return [rows[i] for i in keep if rows[i][0] not in dev]


def encode(tokenizer, task: str, rows) -> list[tuple[list[int], str]]:
    """[(prompt token ids, reference answer)] for (question, answer) rows: the one place a task's prompt is rendered,
    so training, the pass@k filter, evaluation and the noise floor all see the same prompts."""
    return [(tokenizer(TEMPLATES[task].format(question=q), add_special_tokens=False).input_ids, a) for q, a in rows]


def load(cfg, tokenizer=None) -> tuple[Any, Any, list[str]]:
    """(prompt stream, reward, stop strings) for a RunConfig."""
    if cfg.task == "sim":
        vocab = cfg.launch.fake_vocab if cfg.launch.engine == "fake" else 151_936
        return synthetic(4096, cfg.sim.prompt_len, seed=cfg.seed, vocab=vocab), sim_reward(cfg.sim.p_correct,
                    cfg.seed, forced_lengths=bool(cfg.forced_lengths)), []
    if cfg.task in ("gsm8k", "math"):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=model_revision(cfg.model))
        rows = training_rows(cfg, gsm8k_examples("train") if cfg.task == "gsm8k" else math_examples("train"))
        items = [(tuple(ids), a) for ids, a in encode(tokenizer, cfg.task, rows)]
        if long := set(too_long(items, cfg.max_tokens, cfg.max_model_len)):
            print(f"{len(long)} of {len(items)} prompts leave no room for {cfg.max_tokens} tokens within "
                  f"{cfg.max_model_len}: excluded", flush=True)
            items = [x for i, x in enumerate(items) if i not in long]
        stream = PromptStream(items, cfg.seed)
        return stream, TextReward(cfg.model, cfg.task), list(STOPS[cfg.task])
    raise NotImplementedError(f"task {cfg.task!r}")
