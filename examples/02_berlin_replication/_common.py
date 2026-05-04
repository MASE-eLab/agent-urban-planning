"""Shared helpers for the Berlin V1-V5 replication scripts.

Loads the bundled Berlin scenario + agents, applies the East-West
Express τ shock, and writes per_zone.csv outputs in the format consumed
by the comparison + plotting tools.

Each ``run_v*.py`` script is a thin wrapper around
:func:`run_baseline_and_shock` that supplies a different engine factory.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import agent_urban_planning as aup
from agent_urban_planning.core.engine import SimulationEngine
from agent_urban_planning.research.berlin import (
    PACK_WORKER_MASS,
    apply_pack_L_override,
    extract_results,
    infer_unit_from_scenario_path,
    load_obs_fundamentals,
    prepare_scenario,
    scale_agent_mass,
    write_diagnostics,
    write_per_zone_csv,
    write_seed_json,
)
from agent_urban_planning.research.berlin.shock_pipeline import (
    run_shock_pipeline,
)


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
            "Bundled Berlin data missing. The Berlin V1-V5 replication "
            "requires the bundled NPZ + YAML files which ship in the git "
            "repo but NOT in the PyPI sdist. Either:\n"
            "  1. git clone the repo (gives you the data), OR\n"
            "  2. download the data separately (see data/README.md)\n\n"
            "Missing paths:\n" + "\n".join(f"  - {p}" for p in missing)
        )
        raise FileNotFoundError(msg)


def _run_baseline(
    *,
    engine_factory: Callable[..., Any],
    output_dir: Path,
    variant_name: str,
    seed: int,
    iters: int,
    engine_name_for_seed_json: str,
    shock_distribution: str | None,
    extra_seed_fields: dict | None = None,
    pre_run_hook: Callable | None = None,
) -> Path:
    """Run baseline market clearing and write per_zone.csv.

    Returns the output_dir for chaining into the shock run.
    """
    from types import SimpleNamespace
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{variant_name}] start at {time.strftime('%H:%M:%S')} seed={seed}",
          flush=True)

    args_ns = SimpleNamespace(
        scenario=BERLIN_SCENARIO_YAML,
        agents=BERLIN_AGENTS_YAML,
        seed=seed,
        iters=iters,
        project_root=REPO_ROOT,
    )
    sc, ag, root = prepare_scenario(args_ns, version_label=variant_name)
    unit = infer_unit_from_scenario_path(BERLIN_SCENARIO_YAML)
    L_total = apply_pack_L_override(sc, root, unit)
    print(f"[{variant_name}] L = V × K^0.75 applied, Σ = {L_total:.0f}", flush=True)

    engine = _call_engine_factory(engine_factory, sc, seed, iters, phase="baseline")
    eng = SimulationEngine(
        scenario=sc, agent_config=ag, engine=engine,
        seed=seed, verbose=True,
    )
    scale_agent_mass(eng.population, PACK_WORKER_MASS)
    print(f"[{variant_name}] zones={len(eng.base_env.zone_names)}, "
          f"agents={len(eng.population)}, mass={PACK_WORKER_MASS:.0f}",
          flush=True)

    if pre_run_hook is not None:
        pre_run_hook(eng, engine)

    t0 = time.time()
    result = eng.run(policy=None)
    runtime_s = time.time() - t0
    print(f"[{variant_name}] done in {runtime_s:.1f}s", flush=True)

    zone_index = list(eng.base_env.zone_names)
    sim_Q, sim_HR, sim_HM, sim_wage = extract_results(
        result, engine, zone_index, PACK_WORKER_MASS,
    )
    obs = load_obs_fundamentals(root, unit, zone_index)

    seed_extra = {"variant_name": variant_name}
    if extra_seed_fields:
        seed_extra.update(extra_seed_fields)

    write_seed_json(
        output_dir, seed=seed, engine_name=engine_name_for_seed_json,
        shock_distribution=shock_distribution,
        scenario=str(BERLIN_SCENARIO_YAML),
        num_agents=getattr(engine, "num_agents", len(eng.population)),
        extra=seed_extra,
    )
    write_per_zone_csv(
        output_dir, zone_index, sim_Q, sim_HR, sim_HM, sim_wage,
        obs_Q=obs.get("Q"), obs_HR=obs.get("HR"),
        obs_HM=obs.get("HM"), obs_wage=obs.get("wage"),
    )
    write_diagnostics(
        output_dir, version_label=variant_name, runtime_s=runtime_s,
        iters=iters, scenario=str(BERLIN_SCENARIO_YAML),
        extra={
            "n_zones": len(zone_index),
            "agent_mass": PACK_WORKER_MASS,
            "abm_diagnostics": getattr(engine, "last_abm_diagnostics", {}) or {},
        },
    )
    print(f"[{variant_name}] artifacts written to {output_dir}")
    return output_dir


def _call_engine_factory(engine_factory, sc, seed, iters, *, phase):
    """Invoke ``engine_factory`` with optional phase awareness.

    Older factories take ``(sc, seed, iters)``; newer ones can opt into a
    ``phase="baseline"`` / ``"shock"`` keyword for cache isolation. We
    introspect the signature so V1/V2/V3 factories continue to work
    unchanged while V5 can route per-phase cache dirs.
    """
    import inspect
    try:
        sig = inspect.signature(engine_factory)
        if "phase" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            return engine_factory(sc, seed, iters, phase=phase)
    except (TypeError, ValueError):
        pass
    return engine_factory(sc, seed, iters)


def run_baseline_and_shock(
    engine_factory: Callable[..., Any],
    output_subdir: str,
    *,
    iters: int = 50,
    seed: int = 42,
    engine_name_for_seed_json: str = "DecisionEngine",
    shock_distribution: str | None = None,
    pre_run_hook: Callable | None = None,
    extra_seed_fields: dict | None = None,
) -> dict[str, Path]:
    """Run baseline + East-West Express shock and write per_zone.csv outputs.

    Args:
        engine_factory: Callable ``(scenario, seed, iters) -> engine``
            invoked once for the baseline and once for the shocked run.
        output_subdir: Subdirectory under ``output/`` to write baseline
            results to. The shocked run writes to
            ``output/{output_subdir}_shock_east_west/``.
        iters: Number of market-clearing iterations. Default 50.
        seed: Random seed. Default 42.
        engine_name_for_seed_json: String written to ``seed.json`` for
            metadata tracking.
        shock_distribution: Stored in ``seed.json``; e.g. ``"frechet"``,
            ``"normal"``, ``"none"``, or ``"softmax"``.
        pre_run_hook: Optional callable ``(SimulationEngine, engine) -> None``
            invoked after engine construction but before market clearing.
            Used by V4/V5 to pre-elicit preferences or pre-cluster
            agents.
        extra_seed_fields: Additional metadata to record in
            ``seed.json``.

    Returns:
        Dictionary with keys ``"baseline"`` and ``"shock"`` mapping to
        the output directories.

    Raises:
        FileNotFoundError: If the bundled Berlin data is not on disk.
    """
    check_berlin_data_present()

    baseline_dir = OUTPUT_BASE / output_subdir
    shock_dir = OUTPUT_BASE / f"{output_subdir}_shock_east_west"

    # ---- Phase 1: baseline -------------------------------------------------
    _run_baseline(
        engine_factory=engine_factory,
        output_dir=baseline_dir,
        variant_name=output_subdir,
        seed=seed,
        iters=iters,
        engine_name_for_seed_json=engine_name_for_seed_json,
        shock_distribution=shock_distribution,
        extra_seed_fields=extra_seed_fields,
        pre_run_hook=pre_run_hook,
    )

    # ---- Phase 2: shock ----------------------------------------------------
    run_shock_pipeline(
        scenario_path=BERLIN_SCENARIO_YAML,
        agents_path=BERLIN_AGENTS_YAML,
        shock_config_path=BERLIN_SHOCK_YAML,
        zone_names_csv=BERLIN_ZONE_NAMES,
        baseline_dir=baseline_dir,
        output_dir=shock_dir,
        project_root=REPO_ROOT,
        seed=seed,
        iters=iters,
        variant_name=output_subdir,
        engine_factory=engine_factory,
        engine_name_for_seed_json=engine_name_for_seed_json,
        shock_distribution_label=(shock_distribution or "shock-east-west-express"),
        extra_seed_fields=extra_seed_fields,
        pre_run_hook=pre_run_hook,
    )

    return {"baseline": baseline_dir, "shock": shock_dir}
