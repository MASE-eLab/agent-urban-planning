"""Shared helpers for the Berlin V1-V5.4 replication scripts.

Loads the bundled Berlin scenario + agents, applies the East-West
Express τ shock, and writes per_zone.csv outputs in the format consumed
by the comparison + plotting tools.

Each ``run_v*.py`` script is a thin wrapper around :func:`run_baseline_and_shock`
that supplies a different ``aup.UtilityEngine`` / ``HybridDecisionEngine`` /
``LLMDecisionEngine`` configuration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import agent_urban_planning as aup


# Repo root (3 levels up from this file).
REPO_ROOT = Path(__file__).resolve().parents[2]
BERLIN_DATA_DIR = REPO_ROOT / "data" / "berlin"
BERLIN_SCENARIO_YAML = BERLIN_DATA_DIR / "scenarios" / "berlin_2006_ortsteile.yaml"
BERLIN_AGENTS_YAML = BERLIN_DATA_DIR / "agents" / "berlin_ortsteile_richer_10k.yaml"
BERLIN_TT_NPZ = BERLIN_DATA_DIR / "ortsteile" / "tt_2006_actual.npz"
BERLIN_ZONE_NAMES = BERLIN_DATA_DIR / "ortsteile" / "zone_names.csv"
BERLIN_SHOCK_YAML = REPO_ROOT / "data" / "berlin" / "shocks" / "east_west_express.yaml"

OUTPUT_BASE = REPO_ROOT / "output"


def check_berlin_data_present() -> None:
    """Verify the bundled Berlin data is on disk; raise if missing."""
    missing = []
    for p in (BERLIN_SCENARIO_YAML, BERLIN_AGENTS_YAML, BERLIN_TT_NPZ,
              BERLIN_ZONE_NAMES, BERLIN_SHOCK_YAML):
        if not p.exists():
            missing.append(str(p))
    if missing:
        msg = (
            "Bundled Berlin data missing. The Berlin V1-V5.4 replication "
            "requires the bundled NPZ + YAML files which ship in the git "
            "repo but NOT in the PyPI sdist. Either:\n"
            "  1. git clone the repo (gives you the data), OR\n"
            "  2. download the data separately (see data/README.md)\n\n"
            "Missing paths:\n" + "\n".join(f"  - {p}" for p in missing)
        )
        raise FileNotFoundError(msg)


def run_baseline_and_shock(
    engine: aup.DecisionEngine,
    output_subdir: str,
    *,
    iters: int = 50,
    seed: int = 42,
) -> dict[str, Path]:
    """Run baseline + East-West Express shock and write per_zone.csv outputs.

    Args:
        engine: Pre-instantiated decision engine (e.g., ``aup.UtilityEngine``).
        output_subdir: Subdirectory under ``output/`` to write results to.
            E.g., ``"berlin_v1_softmax"``.
        iters: Number of market-clearing iterations. Default 50 (paper setting).
        seed: Random seed. Default 42 (paper setting).

    Returns:
        Dictionary with keys ``"baseline"`` and ``"shock"`` mapping to the
        per_zone.csv paths.

    Raises:
        FileNotFoundError: If the bundled Berlin data is not on disk.

    Examples:
        >>> import agent_urban_planning as aup
        >>> from agent_urban_planning.data.loaders import load_scenario
        >>> sc = load_scenario("data/berlin/scenarios/berlin_2006_ortsteile.yaml")
        >>> engine = aup.UtilityEngine(sc.ahlfeldt_params, mode="softmax")
        >>> out = run_baseline_and_shock(engine, "berlin_v1_softmax")
        >>> # out["baseline"] and out["shock"] are now paths to per_zone.csv files

    Notes:
        This is a placeholder implementation. The full pipeline that
        reproduces the dev repo's outputs lives at
        ``multi_agent_simulator/scripts/run_berlin_v*.py``; consolidating
        it into the public library is tracked in the
        ``extract-library-agent-urban-planning`` change's Phase 4 work.
        For now, treat this script as a documentation reference for the
        intended replication workflow.
    """
    raise NotImplementedError(
        "Berlin replication pipeline not yet wired. See README.md in this "
        "directory for the intended workflow. The dev repo at "
        "multi_agent_simulator/scripts/run_berlin_v*.py contains the "
        "current working implementation."
    )
