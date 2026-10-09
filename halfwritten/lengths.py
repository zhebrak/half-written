"""Answer lengths an attempt draws, as a deterministic function of (sample id, attempt).

Two users: FakeEngine draws a natural length for each answer (it stops there unless told to ignore EOS), and the
controller forces lengths on a real engine for the systems map (ignore_eos with max_tokens set to the drawn length,
PLAN.md §4.5). Keyed by attempt, so a C1 continuation keeps its attempt's length and a C2 restart draws
anew.

    parse("lognormal:300,1.1")   lognormal with mean 300 tokens and sigma 1.1
    parse("fixed:512")           every answer 512 tokens
    parse("parquet:<path>")      lengths drawn from a parquet's `tokens` column (an empirical distribution: a row's
                                 ledger.parquet, or bench.prompt_filter's --lengths-out)
"""
import math
from dataclasses import dataclass
from statistics import NormalDist

from .hashing import stable_key


@dataclass(frozen=True)
class LengthModel:
    mean: float = 200.0
    sigma: float = 0.8
    seed: int = 0
    empirical: tuple[int, ...] | None = None

    def __call__(self, sid: str, attempt: int) -> int:
        u = (stable_key(self.seed, sid, attempt) + 0.5) / 2 ** 64
        if self.empirical:
            return self.empirical[int(u * len(self.empirical))]
        if self.sigma == 0:
            return max(1, round(self.mean))
        mu = math.log(self.mean) - self.sigma ** 2 / 2
        return max(1, round(math.exp(mu + self.sigma * NormalDist().inv_cdf(u))))


def parse(spec: str, seed: int = 0) -> LengthModel:
    kind, _, arg = spec.partition(":")
    if kind == "lognormal":
        mean, sigma = map(float, arg.split(","))
        return LengthModel(mean, sigma, seed)
    if kind == "fixed":
        return LengthModel(float(int(arg)), 0.0, seed)
    if kind == "parquet":
        import pyarrow.parquet as pq
        lengths = tuple(int(n) for n in pq.read_table(arg, columns=["tokens"])["tokens"].to_pylist() if n)
        if not lengths:
            raise ValueError(f"{arg}: no answers")
        return LengthModel(seed=seed, empirical=lengths)
    raise ValueError(f"lengths: lognormal:<mean>,<sigma> | fixed:<n> | parquet:<path>, not {spec!r}")


def sim_fields(spec: str) -> dict:
    """The SimSettings fields that give the fake engine these natural lengths (the regime calculator's --lengths)."""
    m = parse(spec)
    return {"empirical_lengths": list(m.empirical)} if m.empirical else {"length_mean": m.mean, "length_sigma": m.sigma}
