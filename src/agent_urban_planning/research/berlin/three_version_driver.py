"""Shared driver logic for the V1/V2/V3 Berlin comparison.

Each of the V1/V2/V3 example scripts is a thin wrapper over the
functions here. Keeping the three scripts simple at the file level
while sharing the boilerplate (scenario loading, mass scaling, I/O).

Ported from ``simulator.research.three_version_driver`` in the dev repo.
The only differences from the dev version are import paths
(``simulator.*`` → ``agent_urban_planning.*``).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


PACK_WORKER_MASS = 1_770_666.0


def add_common_cli(parser: argparse.ArgumentParser) -> None:
    """Add CLI flags shared by all three drivers."""
    parser.add_argument(
        "--scenario", type=Path,
        default=Path("data/berlin/scenarios/berlin_2006_ortsteile.yaml"),
        help="Scenario YAML (default: 96-zone Ortsteile)",
    )
    parser.add_argument(
        "--agents", type=Path,
        default=Path("data/berlin/agents/berlin_ortsteile_richer_10k.yaml"),
        help="Agent YAML (default: richer 10k Berlin config)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master seed (default 42).",
    )
    parser.add_argument(
        "--iters", type=int, default=50,
        help="Max market iterations.",
    )
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parents[4],
    )


def prepare_scenario(args, version_label: str):
    """Load scenario + agents, freeze agglomeration to Run 1 settings,
    apply the pack's 1.77M worker mass scaling.
    """
    root = args.project_root
    from agent_urban_planning.data.loaders import load_scenario, load_agents
    scenario_path = args.scenario if args.scenario.is_absolute() else root / args.scenario
    agents_path = args.agents if args.agents.is_absolute() else root / args.agents

    sc = load_scenario(scenario_path)
    # Pack settings across all three versions: deterministic continuum path
    # for V1 (closed-form softmax); V2/V3 instantiate AhlfeldtABMEngine
    # explicitly and do not use these flags.
    sc.ahlfeldt_params.deterministic = True
    sc.ahlfeldt_params.endogenous_agglomeration = False
    sc.simulation.market_max_iterations = args.iters
    ag = load_agents(agents_path)
    return sc, ag, root


def scale_agent_mass(population, target_mass: float = PACK_WORKER_MASS) -> None:
    """Scale agent weights so the sum matches pack's worker mass. Same trick
    as Run 1: agent count is still len(population), but each carries a scaled
    weight so the market-clearing aggregates operate at Berlin population scale.
    """
    if len(population) == 0:
        return
    for a in population:
        a.weight = a.weight * target_mass


def write_seed_json(
    output_dir: Path,
    *,
    seed: int,
    engine_name: str,
    shock_distribution: Optional[str],
    scenario: str,
    num_agents: Optional[int] = None,
    extra: Optional[dict] = None,
) -> None:
    """Write reproducibility metadata to ``seed.json``."""
    payload: dict[str, Any] = {
        "seed": int(seed),
        "engine": engine_name,
        "shock_distribution": shock_distribution,
        "scenario": scenario,
    }
    if num_agents is not None:
        payload["num_agents"] = int(num_agents)
    if extra:
        payload.update(extra)
    with open(output_dir / "seed.json", "w") as f:
        json.dump(payload, f, indent=2)


def write_per_zone_csv(
    output_dir: Path,
    zone_index: list[str],
    sim_Q: np.ndarray,
    sim_HR: np.ndarray,
    sim_HM: np.ndarray,
    sim_wage: np.ndarray,
    obs_Q: Optional[np.ndarray] = None,
    obs_HR: Optional[np.ndarray] = None,
    obs_HM: Optional[np.ndarray] = None,
    obs_wage: Optional[np.ndarray] = None,
) -> Path:
    """Emit a per-zone CSV. Observed columns may be omitted (filled empty)."""
    path = output_dir / "per_zone.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "zone_id",
            "Q_sim", "HR_sim", "HM_sim", "wage_sim",
            "Q_obs", "HR_obs", "HM_obs", "wage_obs",
        ])
        for i, zid in enumerate(zone_index):
            row = [
                zid,
                f"{sim_Q[i]:.6f}", f"{sim_HR[i]:.2f}", f"{sim_HM[i]:.2f}",
                f"{sim_wage[i]:.6f}",
                f"{obs_Q[i]:.6f}" if obs_Q is not None else "",
                f"{obs_HR[i]:.2f}" if obs_HR is not None else "",
                f"{obs_HM[i]:.2f}" if obs_HM is not None else "",
                f"{obs_wage[i]:.6f}" if obs_wage is not None else "",
            ]
            w.writerow(row)
    return path


def load_obs_fundamentals(
    data_root: Path, unit: str, zone_index: list[str],
) -> dict[str, np.ndarray]:
    """Load pack-aggregated observed fundamentals for the given unit.

    Returns ``{"Q": obs_Q, "HR": obs_HR, "HM": obs_HM, "wage": obs_wage}``
    aligned to ``zone_index`` order. The NPZ at
    ``data/berlin/<unit>/fund_2006.npz`` stores ``floor_06, emprsd_06,
    empwpl_06, wage_06`` with an ``aggregate_id`` array indicating which
    zone each row belongs to.
    """
    path = data_root / "data" / "berlin" / unit / "fund_2006.npz"
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    # The aggregator writes zone IDs under ``block_id`` (legacy name from
    # before generalization to arbitrary aggregation units). Accept either.
    id_key = "aggregate_id" if "aggregate_id" in z.files else "block_id"
    if id_key not in z.files:
        return {}
    agg_ids = [str(a) for a in z[id_key]]
    idx_by_zone = {zid: i for i, zid in enumerate(agg_ids)}
    n = len(zone_index)

    def _pick(arr: np.ndarray) -> np.ndarray:
        out = np.zeros(n, dtype=np.float64)
        for i, zid in enumerate(zone_index):
            out[i] = float(arr[idx_by_zone[zid]]) if zid in idx_by_zone else 0.0
        return out

    return {
        "Q":    _pick(np.asarray(z["floor_06"], dtype=np.float64)),
        "HR":   _pick(np.asarray(z["emprsd_06"], dtype=np.float64)),
        "HM":   _pick(np.asarray(z["empwpl_06"], dtype=np.float64)),
        "wage": _pick(np.asarray(z["wage_06"], dtype=np.float64)),
    }


def apply_pack_L_override(scenario, data_root: Path, unit: str) -> float:
    """Override each zone's ``total_floor_area`` with pack's ``L = V × area^0.75``.

    Ahlfeldt's closed-form Q FOC — ``Q · L = (1−β) · vv`` — assumes ``L`` is
    the construction-cost-weighted floor supply ``V × K^0.75`` **at block
    level**, not the raw area aggregate. To reproduce pack's observed Q at
    Ortsteile / Bezirke scale we must compute L per block and sum into
    each zone via the same crosswalk used for fundamentals aggregation.

    Returns the total L summed across zones (for logging).
    """
    blocks_path = data_root / "data/berlin/blocks/fund_2006.npz"
    crosswalk_path = data_root / "data/berlin/crosswalks" / f"blocks_to_{unit}.csv"
    if not blocks_path.exists() or not crosswalk_path.exists():
        return 0.0

    fund = np.load(blocks_path, allow_pickle=True)
    block_ids = [str(b) for b in fund["block_id"]]
    L_per_block = np.asarray(fund["V_06"], dtype=np.float64) * np.power(
        np.asarray(fund["area_06"], dtype=np.float64), 0.75,
    )

    # Load crosswalk (skip ``#`` comment lines).
    import csv as _csv
    block_to_zone: dict[str, str] = {}
    with open(crosswalk_path) as f:
        non_comment = (line for line in f if not line.lstrip().startswith("#"))
        reader = _csv.DictReader(non_comment)
        for row in reader:
            block_to_zone[row["block_id"]] = row["aggregate_id"]

    # Sum L per zone.
    zone_L: dict[str, float] = {}
    for i, bid in enumerate(block_ids):
        z = block_to_zone.get(bid)
        if z is None:
            continue
        zone_L[z] = zone_L.get(z, 0.0) + float(L_per_block[i])

    # Override on scenario.
    applied = 0
    for z in scenario.zones:
        if z.name in zone_L:
            z.total_floor_area = zone_L[z.name]
            applied += 1

    return float(sum(zone_L.values()))


def infer_unit_from_scenario_path(path: Path) -> str:
    """Best-effort: ``berlin_2006_ortsteile.yaml`` → ``ortsteile``."""
    name = path.name
    if "ortsteile" in name:
        return "ortsteile"
    if "bezirke" in name:
        return "bezirke"
    if "block" in name:
        return "blocks"
    return "ortsteile"


def extract_results(result, engine, zone_index: list[str], total_mass: float):
    """Pull out sim_Q, sim_HR, sim_HM, sim_wage from a completed SimulationResults.

    sim_HR, sim_HM come from the engine's ``last_choice_probabilities`` if
    present (continuum path). Otherwise they're zero vectors.
    """
    sim_Q = np.array(
        [result.metrics.zone_prices[z] for z in zone_index], dtype=np.float64,
    )
    P = getattr(engine, "last_choice_probabilities", None)
    if P is not None:
        sim_HR = np.asarray(P, dtype=np.float64).sum(axis=1) * total_mass
        sim_HM = np.asarray(P, dtype=np.float64).sum(axis=0) * total_mass
    else:
        sim_HR = np.zeros(len(zone_index), dtype=np.float64)
        sim_HM = np.zeros(len(zone_index), dtype=np.float64)

    sim_wage_dict: dict[str, float] = {}
    if getattr(result, "price_history", None):
        last_snap = result.price_history[-1]
        sim_wage_dict = last_snap.get("wages", {}) if isinstance(last_snap, dict) else {}
    sim_wage = np.array(
        [float(sim_wage_dict.get(z, 0.0)) for z in zone_index], dtype=np.float64,
    )
    return sim_Q, sim_HR, sim_HM, sim_wage


def write_diagnostics(
    output_dir: Path,
    *,
    version_label: str,
    runtime_s: float,
    iters: int,
    scenario: str,
    num_agents: Optional[int] = None,
    extra: Optional[dict] = None,
) -> None:
    payload: dict[str, Any] = {
        "version": version_label,
        "runtime_s": float(runtime_s),
        "iterations_cap": int(iters),
        "scenario": scenario,
    }
    if num_agents is not None:
        payload["num_agents"] = int(num_agents)
    if extra:
        payload.update(extra)
    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(payload, f, indent=2, default=float)
