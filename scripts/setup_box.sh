#!/usr/bin/env bash
# A fresh GPU box, once, after the code is on it: uv when the box has none, the venv
# from the GPU lock when there is one (requirements-gpu.lock, a uv pip freeze of the last box) else requirements.txt,
# a check that the stack imports and sees every GPU, the model and datasets cached (so no stage spends its time
# downloading), the CPU tests that touch the engine adapter, and the topology: the two-GPU layout needs NVLink
# between GPU0 and GPU1 (HW_REQUIRE_NVLINK=0 lets a box without it through, with a warning). Writes
# requirements-gpu.lock and gpu-environment.json, which the session archive keeps.
#   ssh BOX 'cd <checkout> && setsid nohup scripts/setup_box.sh > setup.log 2>&1 < /dev/null &'
set -ex
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"  # where uv's installer puts it
if ! command -v uv > /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if [ ! -x .venv/bin/python ]; then  # a second run (study.sh start after prepare) keeps the venv and checks it
    uv venv --python 3.12 .venv
fi
# The pinned stack as PyPI serves it is built for CUDA 13 (torch 2.13.0+cu130), which needs driver 580 or later; a
# box on driver 570 (CUDA 12.8) runs the same versions in their cu129
# builds (CUDA minor-version compatibility covers 12.8). The lock then records which; gpu-environment.json too.
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ -f requirements-gpu.lock ]; then
    uv pip install --python .venv/bin/python -r requirements-gpu.lock \
        --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
elif [ "${driver:-0}" -ge 580 ]; then
    uv pip install --python .venv/bin/python -r requirements.txt
elif [ "${driver:-0}" -ge 570 ]; then
    echo "driver $driver serves CUDA 12.x: the pinned versions in their cu129 builds"
    # torchvision and torchaudio must follow torch's build: a cu130 torchvision beside a cu129 torch loads no
    # operators, and transformers then cannot import the model class (2026-10-03)
    # torchcodec (imported by vllm.multimodal) has no cu129 build of 0.17: its CPU build imports and is never used
    uv pip install --python .venv/bin/python -r requirements.txt "torch==2.13.0+cu129" \
        "torchvision==0.28.0+cu129" "torchaudio==2.11.0+cu129" "torchcodec==0.17.0+cpu" \
        "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
        --extra-index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match
else
    echo "driver ${driver:-unknown} is older than 570: no build of the pinned stack runs on it"; exit 1
fi
PY=.venv/bin/python
$PY -c 'import torch, vllm, transformers; print(torch.__version__, vllm.__version__, transformers.__version__,
        torch.cuda.device_count(), "GPUs")'
$PY -c 'from huggingface_hub import snapshot_download; from halfwritten.revisions import model_revision as r
m = "Qwen/Qwen3-0.6B-Base"; print(snapshot_download(m, revision=r(m)))'  # the pinned files every load asks for
$PY -c 'from halfwritten.tasks import gsm8k_examples, math_examples, math500_examples
print(len(gsm8k_examples("train")), len(gsm8k_examples("test")), len(math_examples("train")), len(math500_examples()))'
$PY -m pytest -q -x tests/test_imports.py tests/test_golden_vllm.py tests/test_vllm_http.py
nvidia-smi topo -m
# GPU0 and GPU1 must be joined by NVLink (NV#): the weights go over NCCL between them every sync
nvidia-smi topo -m | awk 'NR > 1 && $1 == "GPU0" {print $3}' | grep -q '^NV' || {
    if [ "${HW_REQUIRE_NVLINK:-1}" = 0 ]; then
        echo "WARNING: no NVLink between GPU0 and GPU1 (nvidia-smi topo -m): a rehearsal box, not the study's layout"
    else
        echo "no NVLink between GPU0 and GPU1 (nvidia-smi topo -m): not the study's layout"; exit 1
    fi; }
uv pip freeze --python .venv/bin/python > requirements-gpu.lock
$PY - <<'PY'
import json
from pathlib import Path
import torch
from halfwritten.results import environment
record = environment() | {"cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version()}
Path("gpu-environment.json").write_text(json.dumps(record, indent=1) + "\n")
PY
echo "setup done"
