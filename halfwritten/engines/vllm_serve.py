"""`vllm serve` with one change, for the in-place LoRA arms (launch.lora_inplace_fix): after an in-place adapter load,
the record vLLM keeps for the adapter has its load_inplace flag cleared.

vLLM 0.30 attaches that record to every later request, and each engine step reloads from disk any adapter whose
record says load_inplace, from the path of the batch's first request: after a swap, a request from before it drags
the whole batch back to its older adapter (NOTEBOOK.md 2026-10-02, bench/lora_probe.py). SkyRL and
prime-rl, which reload LoRA in place in production, clear the flag the same way in their own servers
(skyrl/.../vllm_server_actor.py, prime_rl/inference/vllm/server.py). With it cleared, the reload happens once, at
the load; later steps only find the adapter in its slot, holding the new weights, which is the regime: requests in
flight continue on the new adapter over their old KV. The engine's code paths are untouched.

    python -m halfwritten.engines.vllm_serve serve MODEL [vllm serve options]  (one API server: the patch is
    applied in this process, where a single API server runs)
"""
import functools
import sys


def clear_inplace_flag():
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    load = OpenAIServingModels.load_lora_adapter
    if getattr(load, "_clears_inplace", False):
        return

    @functools.wraps(load)
    async def load_lora_adapter(self, request, base_model_name=None):
        out = await load(self, request, base_model_name)
        if (stored := self.lora_requests.get(request.lora_name)) is not None:
            stored.load_inplace = False
        return out

    load_lora_adapter._clears_inplace = True
    OpenAIServingModels.load_lora_adapter = load_lora_adapter


def main(argv: list[str]) -> None:
    if "--api-server-count" in argv:  # further API server processes would not carry the patch
        raise SystemExit("vllm_serve: one API server only (LoRA arms run with one: config)")
    clear_inplace_flag()
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv = ["vllm", *argv]
    vllm_main()


if __name__ == "__main__":
    main(sys.argv[1:])
