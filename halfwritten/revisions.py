"""The exact upstream revisions the study reads (external review F12): the same name on the Hub can point at other
files tomorrow, so every model, tokenizer and dataset load names its commit. These are the snapshots the GPU sessions
of 2026-10-01/02 used (the model as served by vLLM on the two-H100 box, the datasets as cached there and here).
A name without an entry (a local path, the tiny test model) loads as it is."""
MODELS = {"Qwen/Qwen3-0.6B-Base": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"}
DATASETS = {"openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
            "DigitalLearningGmbH/MATH-lighteval": "0530c78699ea5e8eb5530600900e1f328b48acad",
            "HuggingFaceH4/MATH-500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"}


TASK_DATASETS = {"gsm8k": ("openai/gsm8k",), "math": ("DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500")}


# The warm starts (HYPOTHESES.md decision 8): W0 is S1's step 50, in the model repository below at a pinned revision,
# with the sha256 of its weights pre-registered. `python -m halfwritten fetch gsm8k` restores it to prep/gsm8k/ on a
# box and verifies the hash; the revision is pinned once the owner has uploaded it.
WARM_STARTS = {"gsm8k": {"repo": "zhebrak/halfwritten-warm-starts", "revision": "TODO-pin-after-upload",
                         "path": "gsm8k/w0",
                         "sha256": "568716f7029a4be7f0f4e582b91ed0f450274a1d7a1ad29e40b243ff9b168ea8",
                         "prep": "data/two_gpu-2026-10-02/prep/gsm8k"}}


def fetch_warm_start(task: str, to, from_local=None, prep_source=None) -> dict:
    """Puts the task's pre-registered inputs under `to`: W0's weights at to/w0/model.safetensors (from the Hub
    repository at its pinned revision, or copied from `from_local`, a snapshot directory on this machine), verified
    against the pre-registered sha256, and the frozen prompt filter and W0's answer lengths copied from the archive
    (`prep_source`, default the session's prep directory in data/). Returns the files and their hashes."""
    import json
    import shutil
    from pathlib import Path

    from .results import file_sha256
    spec = WARM_STARTS[task]
    to = Path(to)
    (to / "w0").mkdir(parents=True, exist_ok=True)
    weights, record = to / "w0" / "model.safetensors", to / "w0.json"
    previous = json.loads(record.read_text()) if record.exists() else {}
    # the provenance names what placed the file, by kind and hash, never by a path (the record is archived and
    # published); a fetch that finds the verified weights in place keeps the record of the fetch that placed them
    if from_local is not None:
        source_weights = Path(from_local) / "model.safetensors"
        if file_sha256(source_weights) != spec["sha256"]:
            raise ValueError(f"{source_weights}: sha256 is not the pre-registered W0 ({spec['sha256']})")
        if source_weights.resolve() != weights.resolve():
            shutil.copy2(source_weights, weights)
        provenance = "local snapshot"
    elif not weights.exists():
        if spec["revision"].startswith("TODO"):
            raise ValueError("W0 has no published immutable revision yet: use --from-local W0_DIR "
                             "(study.sh start: HW_W0_DIR); the registered sha256 is still required")
        from huggingface_hub import snapshot_download
        got = Path(snapshot_download(spec["repo"], revision=spec["revision"], allow_patterns=[f"{spec['path']}/*"]))
        shutil.copy2(got / spec["path"] / "model.safetensors", weights)
        provenance = f"hub@{spec['revision']}"
    else:
        provenance = previous.get("source") or "already in place"
    digest = file_sha256(weights)
    if digest != spec["sha256"]:
        raise ValueError(f"{weights}: sha256 {digest} is not the pre-registered W0 ({spec['sha256']})")
    out = {"w0": str(weights.relative_to(to)), "sha256": digest}  # relative to the record's directory: no host path
    prep = Path(prep_source) if prep_source is not None else Path(spec["prep"])
    for name in ("pass8.json", "w0-lengths.parquet"):
        if (prep / name).exists():
            shutil.copy2(prep / name, to / name)
            out[name] = file_sha256(to / name)
        else:
            out[name] = None
    record.write_text(json.dumps(out | {"repo": spec["repo"], "revision": spec["revision"], "source": provenance,
                                        "verified": True}, indent=1))
    return out


def task_revisions(task: str) -> dict[str, str]:
    """The datasets a task reads, at their revisions: what a row or a filter of that task records, so re-pinning one
    task's data never makes another task's rows look like another measurement."""
    return {repo: DATASETS[repo] for repo in TASK_DATASETS.get(task, ())}


def model_revision(name: str) -> str | None:
    return MODELS.get(name)


def dataset_revision(repo: str) -> str:
    return DATASETS[repo]
