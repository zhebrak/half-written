"""halfwritten: what an async RL loop does with half-written answers when new weights arrive.

One controller (controller.py) drives a generation engine and a trainer through every weight-update regime
(regimes.py) and records which weight version sampled each token (ledger.py). The engine is stock `vllm serve`
(engines/vllm_http.py) on a GPU, or FakeEngine (engines/fake.py) in virtual time (clock.py) for the tests and the
regime calculator. PLAN.md describes the study.
"""
