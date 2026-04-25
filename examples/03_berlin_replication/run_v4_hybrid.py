#!/usr/bin/env python3
"""Berlin V4 — Hybrid-ABM (LLM-elicited preferences + closed-form choice).

The V4 pattern: an LLM elicits per-agent preference weights (β, κ);
mixed-logit then computes discrete zone choice deterministically.

Requires an LLM provider (codex-cli recommended; ~$5 in API credits).
"""
from __future__ import annotations

import argparse

import agent_urban_planning as aup
from agent_urban_planning.data.loaders import load_scenario

from _common import (
    BERLIN_SCENARIO_YAML,
    check_berlin_data_present,
    run_baseline_and_shock,
)


def _build_elicitor(provider: str):
    """Build an LLM-based preference elicitor for the V4 pattern.

    NOTE: The full elicitor wiring (LLM client → caching → per-type β/κ
    extraction) lives in the dev repo's ``simulator.decisions.elicitation``
    module. The public library exposes ``aup.HybridDecisionEngine(elicitor=...)``
    but does NOT yet provide a high-level "build elicitor from llm_client"
    helper. For end-to-end V4 reproduction, see:

        multi_agent_simulator/scripts/run_berlin_v4_b_shock_argmax_hybrid.py

    This script is a documentation reference; full integration is tracked
    in Phase 4 of the extract-library-agent-urban-planning change.
    """
    raise NotImplementedError(
        "V4 elicitor builder not yet exposed in the public library. "
        "See examples/03_berlin_replication/README.md for the dev-repo "
        "fallback path."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm-provider", default="codex-cli",
                        choices=("codex-cli", "claude-code", "anthropic", "openai"))
    args = parser.parse_args()

    check_berlin_data_present()
    sc = load_scenario(BERLIN_SCENARIO_YAML)
    elicitor = _build_elicitor(args.llm_provider)
    engine = aup.HybridDecisionEngine(
        sc.ahlfeldt_params,
        elicitor=elicitor,
        num_agents=1_000_000,
        cluster_k=50,
        seed=42,
    )
    print(f"Engine: {engine!r}")
    run_baseline_and_shock(engine, output_subdir="berlin_v4_hybrid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
