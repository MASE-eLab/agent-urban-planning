#!/usr/bin/env python3
"""Berlin V4 — Hybrid-ABM (LLM-elicited preferences + closed-form choice).

The V4 pattern: an LLM elicits per-agent factor weights phi (rent, commute,
amenity, wage); mixed-logit then computes discrete zone choice
deterministically.

Two run modes:

  ``--no-llm``: replays the bundled per-agent preference cache at
  ``data/berlin/llm_cache_v4/``. Bit-identical Tier 3c reproduction
  with no LLM credits required. Fails fast on a cache miss.

  Without ``--no-llm``: makes live LLM calls via the configured
  provider. ~$5 in codex-cli credits. Reproduces V4 baseline + shock
  from scratch, populating the cache as it goes.

For smoke-testing the pipeline without LLM credits or the bundled
cache, use ``--llm-provider stub-uniform``.

Forcing fully-fresh elicitation (no cache reuse)
------------------------------------------------
Live runs read the bundled cache at ``data/berlin/llm_cache_v4/`` first
and only call the LLM on cache miss. To force a clean live elicitation
with no cache reuse, point ``--preference-cache-dir`` at an empty
directory:

    python examples/02_berlin_replication/run_v4_hybrid.py \\
        --llm-provider codex-cli \\
        --preference-cache-dir /tmp/v4_fresh_cache

This makes every prompt a cache miss → every prompt is dispatched to
the LLM → all 504 elicitation calls hit the provider, billing the
subscription / credits accordingly. Useful for verifying the LLM's
behaviour or comparing two providers on the same demographic prompts.
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


class _CacheReplayClient:
    """LLM client that hard-fails on any live call.

    Used with ``--no-llm`` to force the elicitor onto the bundled disk
    cache. Every cache hit short-circuits before reaching ``.complete``;
    an actual call here means the bundled cache is missing an entry the
    pipeline needs, which is a misconfiguration.
    """

    total_concurrency = 1

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def complete(self, prompt: str, system: str = "") -> str:
        del prompt, system
        raise RuntimeError(
            "CacheReplayClient.complete called: the V4 preference cache at "
            f"{self.cache_dir}/ is missing an entry needed by this run. "
            "Likely causes: (1) you ran with seed/scenario params that "
            "don't match the paper config (seed=42, iters=50, "
            "config/scenarios/berlin_2006_ortsteile.yaml) — the bundled "
            "cache only covers the paper config; (2) you downloaded an "
            "incomplete cache. Either re-run without --no-llm to populate "
            "the cache, or verify the cache was downloaded correctly. "
            "See data/README.md."
        )


def _build_elicitor(
    provider: str, cache_dir: Path, no_llm: bool = False,
) -> LLMPreferenceElicitor:
    """Build an LLM-based preference elicitor for the V4 pattern.

    Returns an :class:`LLMPreferenceElicitor` that can be passed as
    ``elicitor=`` to :class:`aup.HybridDecisionEngine`. When ``no_llm``
    is True, the inner client is a :class:`_CacheReplayClient` that
    fails fast on any live call.
    """
    from agent_urban_planning.llm import clients as _clients
    if no_llm:
        return LLMPreferenceElicitor(
            _CacheReplayClient(cache_dir), cache_dir=str(cache_dir),
        )
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
    parser.add_argument("--no-llm", action="store_true",
                        help="Replay the bundled preference cache at "
                             "data/berlin/llm_cache_v4/ without making any "
                             "live LLM calls. Fails fast on cache miss.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--num-agents", type=int, default=1_000_000,
                        help="Monte Carlo replicates for the per-agent "
                             "Frechet shock + argmax (default 1M, paper config).")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--preference-cache-dir", type=Path,
                        default=Path("data/berlin/llm_cache_v4"))
    args = parser.parse_args()

    cache_dir = (
        args.preference_cache_dir
        if args.preference_cache_dir.is_absolute()
        else REPO_ROOT / args.preference_cache_dir
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.no_llm:
        n_cached = sum(1 for _ in cache_dir.glob("*.json"))
        if n_cached == 0:
            raise SystemExit(
                f"--no-llm requires a populated cache at {cache_dir}. "
                f"None found. See data/README.md for cache hosting."
            )
        print(f"[berlin_v4_hybrid] --no-llm replay mode, "
              f"cache: {cache_dir} ({n_cached} entries)", flush=True)

    elicitor = _build_elicitor(args.llm_provider, cache_dir, no_llm=args.no_llm)

    def engine_factory(sc, seed, iters):
        return aup.HybridDecisionEngine(
            sc.ahlfeldt_params,
            elicitor=elicitor,
            preference_cache_dir=cache_dir,
            seed=seed,
            shock_distribution="frechet",
            num_agents=args.num_agents,
            batch_size=args.batch_size,
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
        engine_name_for_seed_json="AhlfeldtShockArgmaxHybridEngine",
        shock_distribution="none",
        pre_run_hook=pre_run_hook,
        extra_seed_fields={
            "llm_provider": args.llm_provider,
            "no_llm": args.no_llm,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
