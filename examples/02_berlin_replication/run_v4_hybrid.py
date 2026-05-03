#!/usr/bin/env python3
"""Berlin V4 — Hybrid-ABM (LLM-elicited preferences + closed-form choice).

The V4 pattern: an LLM elicits per-agent preference weights (β, κ);
mixed-logit then computes discrete zone choice deterministically.

Requires an LLM provider (codex-cli recommended; ~$5 in API credits).
For smoke-testing the pipeline without LLM credits, use
``--llm-provider stub-uniform``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Picks up ANTHROPIC_API_KEY / OPENAI_API_KEY / ZAI_API_KEY etc.

import agent_urban_planning as aup
from agent_urban_planning.llm.clients import LLMPreferenceElicitor

from _common import REPO_ROOT, run_baseline_and_shock


class _StubUniformClient:
    """A stand-in LLM client for smoke testing without API credits.

    Returns a uniform preference response for every prompt. The hybrid
    engine's elicitation pipeline accepts this and produces uniform
    per-type weights, exercising the full V4 plumbing without any
    network access.
    """

    total_concurrency = 1

    def complete(self, prompt: str, system: str = "") -> str:
        del prompt, system
        return '{"housing": 5, "commute": 5, "services": 5, "amenities": 5}'


def _build_elicitor(provider: str, cache_dir: Path) -> LLMPreferenceElicitor:
    """Build an LLM-based preference elicitor for the V4 pattern.

    Returns an :class:`LLMPreferenceElicitor` that can be passed as
    ``elicitor=`` to :class:`aup.HybridDecisionEngine`.
    """
    from agent_urban_planning.llm import clients as _clients
    if provider == "stub-uniform":
        client: object = _StubUniformClient()
    elif provider == "codex-cli":
        client = _clients.CodexCliClient()
    elif provider == "claude-code":
        client = _clients.ClaudeCodeClient()
    elif provider == "zai-coding":
        client = _clients.ZaiCodingClient()
    elif provider == "anthropic":
        client = _clients.AnthropicClient()
    elif provider == "openai":
        client = _clients.OpenAIClient()
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return LLMPreferenceElicitor(client, cache_dir=str(cache_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm-provider", default="stub-uniform",
                        choices=("stub-uniform", "codex-cli", "claude-code",
                                 "zai-coding", "anthropic", "openai"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--preference-cache-dir", type=Path,
                        default=Path(".cache/llm_preferences_berlin_v4"))
    args = parser.parse_args()

    cache_dir = (
        args.preference_cache_dir
        if args.preference_cache_dir.is_absolute()
        else REPO_ROOT / args.preference_cache_dir
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    elicitor = _build_elicitor(args.llm_provider, cache_dir)

    def engine_factory(sc, seed, iters):
        return aup.HybridDecisionEngine(
            sc.ahlfeldt_params,
            elicitor=elicitor,
            preference_cache_dir=cache_dir,
            seed=seed,
            dtype=getattr(sc.ahlfeldt_params, "dtype", "float64"),
        )

    def pre_run_hook(eng, engine):
        # Elicit preferences once before market clearing kicks off.
        engine.ensure_elicitation(list(eng.population))

    run_baseline_and_shock(
        engine_factory,
        output_subdir="berlin_v4_hybrid",
        iters=args.iters,
        seed=args.seed,
        engine_name_for_seed_json="AhlfeldtArgmaxHybridEngine",
        shock_distribution="none",
        pre_run_hook=pre_run_hook,
        extra_seed_fields={"llm_provider": args.llm_provider},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
