#!/usr/bin/env python3
"""Quickstart: instantiate the 5 V1-V5 decision-engine variants.

Demonstrates the public API surface in <60 lines of code with no external
data dependencies. Runs in <10 seconds.

This script is the Tier 2 reproducibility check — if this works, your
``agent-urban-planning`` install is healthy and the 5 paper variants are
configurable via the kwargs documented at:

    https://agent-urban-planning.readthedocs.io/api/decisions.html

End-to-end Berlin reproduction with bundled data is in
``examples/03_berlin_replication/``.
"""
from __future__ import annotations

import agent_urban_planning as aup
from agent_urban_planning.data.loaders import AhlfeldtParams


def _stub_llm_client():
    """Minimal stub for the LLM-using variants (V4, V5.0, V5)."""

    class _StubLLMClient:
        total_concurrency = 1

        def complete(self, user, system=""):
            return '{"top_5":[],"scores":[]}'

    return _StubLLMClient()


def main() -> int:
    # The Ahlfeldt 2015 structural parameters (Berlin defaults).
    params = AhlfeldtParams(
        kappa_eps=0.0987,
        epsilon=6.6941,
        lambda_=0.071,
        delta=0.362,
        eta=0.155,
        rho=0.759,
    )

    # V1 — Baseline-softmax (closed-form, deterministic).
    v1 = aup.UtilityEngine(params, mode="softmax")

    # V2 — Baseline-ABM argmax with Fréchet idiosyncratic shocks.
    v2 = aup.UtilityEngine(params, mode="argmax", noise="frechet",
                           num_agents=10, batch_size=10, seed=42)

    # V3 — Normal-ABM argmax with Gaussian shocks.
    v3 = aup.UtilityEngine(params, mode="argmax", noise="normal",
                           num_agents=10, batch_size=10, seed=42)

    # V5.0 — Top-5 hierarchical LLM (legacy).
    v5_0 = aup.LLMDecisionEngine(params, llm_client=_stub_llm_client(),
                                 response_format="top5",
                                 num_agents=10, batch_size=10, seed=42, cluster_k=2)

    # V5 — Score-all-96 + rebalance + stage-2 cap (paper headline).
    v5 = aup.LLMDecisionEngine(params, llm_client=_stub_llm_client(),
                                 response_format="score_all",
                                 rebalance_instruction=True,
                                 stage2_top_k_residences=10,
                                 num_agents=10, batch_size=10, seed=42, cluster_k=2)

    print(f"agent-urban-planning version: {aup.__version__}")
    print()
    for label, engine in (("V1   Baseline-softmax    ", v1),
                          ("V2   Baseline-ABM argmax ", v2),
                          ("V3   Normal-ABM argmax   ", v3),
                          ("V5.0 LLM top-5           ", v5_0),
                          ("V5 LLM score-all-96    ", v5)):
        print(f"  {label}: {engine!r}")
    print()
    print("All 5 paper variants instantiated successfully.")
    print("For end-to-end Berlin reproduction, see examples/03_berlin_replication/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
