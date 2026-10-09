"""CPU modules never import vLLM, and the controller side never imports torch: the loop runs on a laptop, and a
GPU-only import hiding in it would only fail on a pod."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_SIDE = ["halfwritten.controller", "halfwritten.engines.fake", "halfwritten.engines.vllm_http",
                   "halfwritten.ledger", "halfwritten.regimes", "halfwritten.trace", "halfwritten.metrics",
                   "halfwritten.clock", "halfwritten.config", "halfwritten.tasks", "halfwritten.rewards",
                   "halfwritten.trainer.client", "halfwritten.__main__", "halfwritten.results", "halfwritten.gpu",
                   "halfwritten.simulate", "bench.canary", "bench.calculator"]


def test_controller_side_imports_neither_vllm_nor_torch():
    code = ("import importlib, json, sys\n"
            f"for m in {CONTROLLER_SIDE!r}: importlib.import_module(m)\n"
            "print(json.dumps(sorted(k for k in sys.modules if k.split('.')[0] in ('vllm', 'torch'))))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == []


def test_the_core_package_never_imports_the_study_harness():
    """halfwritten/ is the loop others read; bench/ is this study's harness. The dependency runs one way (review
    2026-10-02 plan, section 2.5): a reader can take the package without the harness."""
    import ast
    offenders = []
    for f in sorted((ROOT / "halfwritten").rglob("*.py")):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "bench":
                offenders.append(f"{f.relative_to(ROOT)}:{node.lineno}")
            if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "bench" for a in node.names):
                offenders.append(f"{f.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []
