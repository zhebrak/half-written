"""The regime calculator: which regime wins for a given length distribution and trainer speed, predicted by running
the real controller in virtual time against a FakeEngine whose step-time model a GPU session measured.

    python -m bench.calculator --lengths lognormal:300,1.1 --regimes A,B,C1,C2,D,E --B 16 --G 8 --K 4 --budget 300
    python -m bench.calculator --lengths parquet:results/.../ledger.parquet --cost alpha=2.1e-3,beta=1.8e-5,... \
        --train 0.3,2.9e-5

Columns: goodput counts consumed tokens only (C2's discarded attempts and unconsumed answers are not throughput);
"in flight" is the time-weighted number of answers generating, the engine's occupancy, drains included.

The CostModel comes from bench.calibrate (vLLM's own step times, fitted in two stages); bench.replay checks it
against real rows, and its error on held-out systems-map cells is hypothesis H1's first test (PLAN.md §3).
"""
import argparse

from halfwritten.config import RunConfig, SimSettings
from halfwritten.lengths import sim_fields
from halfwritten.simulate import configure, simulate, table, table_row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--regimes", default="A,B,C1,C2,D,E")
    p.add_argument("--lengths", default="lognormal:250,1.0")
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--M", type=int, default=None, help="groups in flight (default B)")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--S", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--budget", type=float, default=60.0, help="virtual seconds per regime")
    p.add_argument("--cost", default="", help="CostModel overrides, e.g. alpha=2e-3,beta=2e-5")
    p.add_argument("--train", default="0.2,3e-5", help="trainer seconds per step: fixed,per_token")
    a = p.parse_args(argv)
    cost = {k: float(v) for k, v in (kv.split("=") for kv in a.cost.split(",") if kv)}
    fixed, per_token = map(float, a.train.split(","))
    sim = SimSettings(cost=cost, train_s_fixed=fixed, train_s_per_token=per_token, **sim_fields(a.lengths))
    base = RunConfig(B=a.B, G=a.G, M=a.M or a.B, K=a.K, S=a.S, max_tokens=a.max_tokens, budget_s=a.budget, sim=sim)
    rows = [table_row(r, simulate(configure(base, r))) for r in a.regimes.split(",")]
    print(table(rows))


if __name__ == "__main__":
    main()
