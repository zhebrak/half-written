"""python -m halfwritten sim CONFIG [--regimes A,B,...] [--out DIR]   every regime in virtual time, one row each
python -m halfwritten run CONFIG                                     a GPU run: stock `vllm serve` + trainer worker
python -m halfwritten where CONFIG [--set ...]                       the row directory that run would write
python -m halfwritten fetch gsm8k [--to prep/gsm8k]                  W0 (by hash) and the frozen filter, for a box
"""
import argparse
import sys
from pathlib import Path


def sim(args) -> int:
    from halfwritten.config import load
    from halfwritten.results import row_dir, write_row
    from halfwritten.simulate import configure, simulate, table, table_row
    base = load(args.config)
    regimes = args.regimes.split(",") if args.regimes else [base.regime + ("+lora" if base.lora else "")]
    rows, failed = [], 0
    for regime in regimes:
        cfg = configure(base, regime)
        result = simulate(cfg)
        rows.append(table_row(regime, result))
        if not result.summary.valid:
            failed += 1
            print(f"{regime}: INVALID {result.summary.error or ''} {result.summary.violations[:3]}", file=sys.stderr)
        if args.out:
            write_row(row_dir(cfg, "sim", Path(args.out)), cfg, result.summary, result.controller.ledger,
                      result.events)
    print(table(rows))
    return 1 if failed else 0


def run(args) -> int:
    import json

    from halfwritten.config import load, with_overrides
    from halfwritten.launch import run_row
    cfg = with_overrides(load(args.config), args.set or [])
    path = run_row(cfg, Path(args.out))
    metrics = json.loads((path / "metrics.json").read_text())
    print(f"{path}: valid={metrics['valid']} steps={metrics['steps']} "
          f"goodput={metrics['goodput_tokens_per_s']:.0f} tok/s {metrics['violations'][:3]}")
    return 0 if metrics["valid"] else 1


def where(args) -> int:
    from halfwritten.config import load, with_overrides
    from halfwritten.launch import row_dir_on_host
    print(row_dir_on_host(with_overrides(load(args.config), args.set or []), Path(args.out)))
    return 0


def fetch(args) -> int:
    import json

    from halfwritten.revisions import fetch_warm_start
    out = fetch_warm_start(args.task, args.to, from_local=args.from_local)
    print(json.dumps(out, indent=1))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m halfwritten", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim", help="run regimes against the fake engine in virtual time")
    s.add_argument("config")
    s.add_argument("--regimes", help="comma-separated, e.g. A,B,C1,C2,D,E,F,D+lora (default: the config's)")
    s.add_argument("--out", default="results", help="results root; '' to write nothing")
    r = sub.add_parser("run", help="run one row: engine server, trainer worker, controller")
    r.add_argument("config")
    r.add_argument("--set", action="append", help="override, e.g. --set K=2 --set launch.engine=fake")
    r.add_argument("--out", default="results")
    w = sub.add_parser("where", help="print the row directory a run of CONFIG would write")
    w.add_argument("config")
    w.add_argument("--set", action="append")
    w.add_argument("--out", default="results")
    f = sub.add_parser("fetch", help="the pre-registered inputs of a task (W0 by hash, the frozen filter) into a dir")
    f.add_argument("task", choices=("gsm8k",))
    f.add_argument("--to", default=None, help="default: prep/<task>")
    f.add_argument("--from-local", default=None, help="a snapshot directory on this machine instead of the Hub")
    args = p.parse_args(argv)
    if args.cmd == "fetch" and args.to is None:
        args.to = f"prep/{args.task}"
    return {"sim": sim, "run": run, "where": where, "fetch": fetch}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
