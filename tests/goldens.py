"""Golden fixtures: a value's canonical JSON (floats at 9 decimal places, keys sorted) against tests/fixtures/<name>.
The certified behaviour is pinned this way before refactors (the review-2026-10-02 plan, section 1.0); a fixture is
regenerated only with HW_UPDATE_GOLDEN=1, and its diff reviewed."""
import json
import math
import os
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def canonical(value):
    """JSON-serialisable with floats rounded, so a digest is stable across platforms' last bits."""
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 9)
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if hasattr(value, "tolist"):  # numpy scalars and arrays
        return canonical(value.tolist())
    if hasattr(value, "to_pylist"):  # pyarrow tables
        return canonical(value.to_pylist())
    return value


def check_golden(name: str, value) -> None:
    """Compares `value` with the fixture, or writes the fixture under HW_UPDATE_GOLDEN=1."""
    path = FIXTURES / name
    text = json.dumps(canonical(value), sort_keys=True, indent=1) + "\n"
    if os.environ.get("HW_UPDATE_GOLDEN"):
        path.write_text(text)
        return
    assert path.exists(), f"{path} is missing: run with HW_UPDATE_GOLDEN=1 once and review the file"
    expected = json.loads(path.read_text())
    actual = json.loads(text)
    if expected != actual:
        diff = _first_difference(expected, actual)
        raise AssertionError(f"{name} changed at {diff}: regenerate only if the change is intended "
                             "(HW_UPDATE_GOLDEN=1) and review the diff")


def check_digests(name: str, cases: dict, summary=lambda v: {}) -> None:
    """Pins large values by their sha256 (one per case) beside a few readable numbers from `summary(value)`, so the
    fixture stays small and its diffs legible. HW_UPDATE_GOLDEN=1 regenerates it; HW_DUMP_GOLDEN=DIR also writes
    every case's full canonical JSON there, so two commits' outputs can be diffed file by file."""
    import hashlib
    path = FIXTURES / name
    actual = {}
    for case, value in cases.items():
        text = json.dumps(canonical(value), sort_keys=True, separators=(",", ":"))
        actual[case] = {"sha256": hashlib.sha256(text.encode()).hexdigest(), **canonical(summary(value))}
        if dump := os.environ.get("HW_DUMP_GOLDEN"):
            Path(dump).mkdir(parents=True, exist_ok=True)
            (Path(dump) / f"{name}.{case.replace('/', '_')}.json").write_text(
                json.dumps(canonical(value), sort_keys=True, indent=1) + "\n")
    if os.environ.get("HW_UPDATE_GOLDEN"):
        path.write_text(json.dumps(actual, sort_keys=True, indent=1) + "\n")
        return
    assert path.exists(), f"{path} is missing: run with HW_UPDATE_GOLDEN=1 once and review the file"
    expected = json.loads(path.read_text())
    changed = sorted(c for c in set(expected) | set(actual) if expected.get(c) != actual.get(c))
    assert not changed, (f"{name}: {len(changed)} case(s) changed ({', '.join(changed[:6])}): regenerate only if "
                         "the change is intended (HW_UPDATE_GOLDEN=1), and diff the full records "
                         "(HW_DUMP_GOLDEN=DIR on both commits)")


def _first_difference(a, b, path="$"):
    if type(a) is not type(b):
        return f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: {'missing' if k not in a else 'extra'} in the new value"
            if a[k] != b[k]:
                return _first_difference(a[k], b[k], f"{path}.{k}")
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return _first_difference(x, y, f"{path}[{i}]")
    return f"{path}: {a!r} != {b!r}"
