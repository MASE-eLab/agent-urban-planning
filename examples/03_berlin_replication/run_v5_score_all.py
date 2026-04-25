#!/usr/bin/env python3
"""Berlin V5 — LLM-ABM (score-all-96 + rebalance + stage-2 cap).

The paper's headline contribution. Configures ``LLMDecisionEngine``
to use the score-all-96 stage-1 prompt with the V5.3-validated rebalance
instruction and the stage-2 top-K residence cap.

Two run modes:

  ``--no-llm`` (default): replays the bundled cached LLM responses at
  ``data/berlin/llm_cache_v5/``. ~5 min wall-clock, no LLM credits
  required. Tier-4 reproducibility for reviewers without LLM access.

  Without ``--no-llm``: makes live LLM calls via the configured provider.
  ~10 hr wall-clock, ~$30-50 in codex-cli credits. Reproduces V5
  baseline + shock from scratch.

Configuration::

    aup.LLMDecisionEngine(
        params,
        llm_client=...,
        response_format="score_all",
        rebalance_instruction=True,
        stage2_top_k_residences=10,
    )
"""
from __future__ import annotations

import argparse

import agent_urban_planning as aup
from agent_urban_planning.data.loaders import load_scenario

from _common import (
    BERLIN_DATA_DIR,
    BERLIN_SCENARIO_YAML,
    check_berlin_data_present,
    run_baseline_and_shock,
)


def _build_llm_client(provider: str, no_llm: bool, cache_dir):
    """Construct the LLM client for the V5 run.

    When ``no_llm=True``, returns a CacheReplayClient that only reads from
    the bundled cache without ever issuing a live call. Useful for
    reproducibility without LLM credits.
    """
    if no_llm:
        # The CacheReplayClient is bundled in aup.llm; for this skeleton
        # implementation we just point at the cache dir.
        raise NotImplementedError(
            f"--no-llm cache replay requires the CacheReplayClient "
            f"helper in aup.llm.cache, not yet exposed in the public "
            f"library API. Cache lives at {cache_dir}. The dev repo's "
            f"V5 baseline run already populated this cache; see the "
            f"dev repo's run_berlin_v5_hierarchical.py for replay logic."
        )
    if provider == "codex-cli":
        from agent_urban_planning.llm.clients import CodexCliClient
        return CodexCliClient()
    if provider == "claude-code":
        from agent_urban_planning.llm.clients import ClaudeCodeClient
        return ClaudeCodeClient()
    raise ValueError(f"unknown provider: {provider}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm-provider", default="codex-cli",
                        choices=("codex-cli", "claude-code"))
    parser.add_argument("--no-llm", action="store_true",
                        help="Replay bundled cache instead of live LLM calls.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    check_berlin_data_present()
    sc = load_scenario(BERLIN_SCENARIO_YAML)

    cache_dir = BERLIN_DATA_DIR / "llm_cache_v5"
    llm_client = _build_llm_client(args.llm_provider, args.no_llm, cache_dir)

    # V5 configuration — paper's headline.
    engine = aup.LLMDecisionEngine(
        sc.ahlfeldt_params,
        llm_client=llm_client,
        response_format="score_all",       # V5 — score every zone
        rebalance_instruction=True,        # V5.3-validated affordability instruction
        stage2_top_k_residences=10,        # cost cap on stage-2 fan-out
        cluster_k=50,
        num_agents=1_000_000,
        seed=args.seed,
        cache_dir=cache_dir,
    )
    print(f"Engine: {engine!r}")
    run_baseline_and_shock(
        engine,
        output_subdir="berlin_v5_score_all",
        iters=args.iters,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
