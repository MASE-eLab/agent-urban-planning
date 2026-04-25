#!/usr/bin/env python3
"""Berlin V5 — LLM-ABM (score-all-96 + rebalance + stage-2 cap).

The paper's headline contribution. Configures ``LLMDecisionEngine``
to use the score-all-96 stage-1 prompt with the V5.3-validated
rebalance instruction and the stage-2 top-K residence cap.

Two run modes:

  ``--no-llm`` (default): replays the bundled cached LLM responses at
  ``data/berlin/llm_cache_v5/``. ~5 min wall-clock, no LLM credits
  required. Tier-4 reproducibility for reviewers without LLM access.
  Requires the cache directory to exist; raises a clear error if it
  doesn't.

  Without ``--no-llm``: makes live LLM calls via the configured
  provider. ~10 hr wall-clock, ~$30-50 in codex-cli credits.
  Reproduces V5 baseline + shock from scratch.

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
import json
from pathlib import Path

import agent_urban_planning as aup

from _common import (
    BERLIN_DATA_DIR,
    BERLIN_ZONE_NAMES,
    REPO_ROOT,
    run_baseline_and_shock,
)


class _CacheReplayClient:
    """LLM client that hard-fails on any live call.

    Used with ``--no-llm`` to force the engine onto the disk cache.
    Every cache hit short-circuits before reaching ``.complete``; an
    actual call here means the engine wanted a response not in cache,
    which is a misconfiguration.
    """

    total_concurrency = 1

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def complete(self, user: str, system: str = "") -> str:
        raise RuntimeError(
            "CacheReplayClient.complete called: the LLM cache at "
            f"{self.cache_dir} is missing an entry needed by this run. "
            "Either re-run without --no-llm to populate the cache, or "
            "verify the cache was downloaded correctly. See data/README.md."
        )


class _StubScoreAllClient:
    """Stub that returns uniform stage-1 score-all and stage-2 top-5 responses.

    Used for smoke testing the V5 plumbing without any cache or live
    LLM access. Detects the prompt's expected schema (score-all stage-1
    vs top-5 stage-2) by looking for the format hint in the prompt body
    and returns a uniform response over the zone names extracted from
    the bullet-listed prompt entries.
    """

    total_concurrency = 1

    def complete(self, user: str, system: str = "") -> str:
        import re
        names = re.findall(r"^- ([^\s:][^:\n]*?)(?:\s*:|$)",
                           user, flags=re.MULTILINE)
        seen: set[str] = set()
        zones: list[str] = []
        for n in names:
            n = n.strip()
            if n and n not in seen:
                seen.add(n)
                zones.append(n)

        # Stage-1 score-all wants {"scores": [...]}; stage-2 wants
        # {"top_5": [...]}. Detect by the JSON schema the prompt advertises.
        wants_top5 = '"top_5"' in user
        if wants_top5:
            picks = zones[:5] if zones else []
            if not picks:
                return '{"top_5": []}'
            score = 1.0 / len(picks)
            return json.dumps({
                "top_5": [{"zone": z, "score": score} for z in picks],
            })
        # Stage-1: score-all-96 schema.
        if not zones:
            return '{"scores": []}'
        score = 1.0 / len(zones)
        return json.dumps({
            "scores": [{"zone": z, "score": score} for z in zones],
        })


def _load_zone_name_map(path: Path) -> dict[str, str]:
    """Load synthetic_id → ortsteile_name crosswalk."""
    import csv
    mapping: dict[str, str] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            mapping[row["synthetic_id"]] = row["ortsteile_name"]
    return mapping


def _build_llm_client(provider: str, no_llm: bool, cache_dir: Path):
    """Construct the LLM client for the V5 run.

    With ``no_llm=True`` returns a :class:`_CacheReplayClient` that
    deliberately fails on any cache miss. The cache directory must
    already exist; a clear error is raised if it does not.
    """
    if no_llm:
        if not cache_dir.exists() or not any(cache_dir.glob("*.json")):
            raise FileNotFoundError(
                f"--no-llm cache replay requires {cache_dir} to contain "
                "LLM cache JSON files. The bundled V5 cache (~320 MB) is "
                "hosted as a GitHub release asset; see data/README.md for "
                "download instructions. Without the cache the script "
                "cannot replay LLM responses."
            )
        return _CacheReplayClient(cache_dir)
    if provider == "stub-score-all":
        return _StubScoreAllClient()
    if provider == "codex-cli":
        from agent_urban_planning.llm.clients import CodexCliClient
        return CodexCliClient()
    if provider == "claude-code":
        from agent_urban_planning.llm.clients import ClaudeCodeClient
        return ClaudeCodeClient()
    raise ValueError(f"unknown provider: {provider}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm-provider", default="stub-score-all",
                        choices=("stub-score-all", "codex-cli", "claude-code"))
    parser.add_argument("--no-llm", action="store_true",
                        help="Replay bundled cache instead of live LLM calls.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--num-agents", type=int, default=1_000_000,
                        help="Monte Carlo replicates (default 1M).")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--cluster-k", type=int, default=50)
    parser.add_argument("--llm-concurrency", type=int, default=15)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Override the LLM cache directory. Default: "
                             "data/berlin/llm_cache_v5/")
    args = parser.parse_args()

    cache_dir = args.cache_dir or (BERLIN_DATA_DIR / "llm_cache_v5")
    cache_dir = cache_dir if cache_dir.is_absolute() else REPO_ROOT / cache_dir

    llm_client = _build_llm_client(args.llm_provider, args.no_llm, cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Zone-name crosswalk so the LLM sees real Ortsteile names.
    zone_name_map = _load_zone_name_map(BERLIN_ZONE_NAMES)

    def engine_factory(sc, seed, iters):
        return aup.LLMDecisionEngine(
            sc.ahlfeldt_params,
            llm_client=llm_client,
            response_format="score_all",       # paper's V5 headline
            rebalance_instruction=True,        # V5.3 affordability instruction
            stage2_top_k_residences=10,        # cost cap on stage-2 fan-out
            cluster_k=args.cluster_k,
            zone_name_map=zone_name_map,
            cache_dir=cache_dir,
            num_agents=args.num_agents,
            batch_size=args.batch_size,
            llm_concurrency=args.llm_concurrency,
            seed=seed,
            dtype=getattr(sc.ahlfeldt_params, "dtype", "float64"),
        )

    def pre_run_hook(eng, engine):
        # Pre-cluster so cluster personas are committed before the market
        # loop fires the first stage-1 / stage-2 LLM batch.
        engine.ensure_clustering(list(eng.population))

    run_baseline_and_shock(
        engine_factory,
        output_subdir="berlin_v5_score_all",
        iters=args.iters,
        seed=args.seed,
        engine_name_for_seed_json="AhlfeldtHierarchicalLLMEngine",
        shock_distribution="none",
        pre_run_hook=pre_run_hook,
        extra_seed_fields={
            "llm_provider": args.llm_provider,
            "no_llm": args.no_llm,
            "cache_dir": str(cache_dir),
            "cluster_k": args.cluster_k,
            "response_format": "score_all",
            "rebalance_instruction": True,
            "stage2_top_k_residences": 10,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
