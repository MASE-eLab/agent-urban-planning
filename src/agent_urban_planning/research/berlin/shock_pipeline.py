"""Shared pipeline for Berlin policy-shock runs across V1..V5 variants.

Each variant has a slightly different engine construction, but the
surrounding plumbing (load scenario → load shock config → warm-start
from baseline → patch τ matrix → run SimulationEngine → write outputs)
is identical. This module provides the shared helper; per-variant
drivers supply only an ``engine_factory`` callable.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

from agent_urban_planning.research.berlin.railway_shock import (
    apply_railway_shock_with_diagnostics,
)
from agent_urban_planning.research.berlin.shock_config import load_shock_config
from agent_urban_planning.research.berlin.warm_start import (
    load_baseline_prices,
    inject_initial_prices,
)


logger = logging.getLogger(__name__)


def run_shock_pipeline(
    *,
    scenario_path: Path,
    agents_path: Path,
    shock_config_path: Path,
    zone_names_csv: Path,
    baseline_dir: Path,
    output_dir: Path,
    project_root: Path,
    seed: int,
    iters: int,
    variant_name: str,
    engine_factory: Callable,
    engine_name_for_seed_json: str,
    shock_distribution_label: str = "shock-east-west-express",
    extra_seed_fields: dict | None = None,
    agent_mass_scale: bool = True,
    L_override: bool = True,
    pre_run_hook: Callable | None = None,
    post_run_hook: Callable | None = None,
) -> None:
    """Run a single-variant shock experiment end-to-end.

    Args:
        scenario_path: baseline scenario YAML.
        agents_path: agents YAML.
        shock_config_path: path to shock YAML
            (e.g. ``east_west_express.yaml``).
        zone_names_csv: path to ``zone_names`` crosswalk.
        baseline_dir: directory containing the variant's baseline
            ``per_zone.csv``.
        output_dir: where to write shocked outputs.
        project_root: absolute path to repo root for path resolution.
        seed: RNG seed passed to ``SimulationEngine``.
        iters: max market-iteration cap.
        variant_name: short label like "V1", "V2", "V4-B", "V5".
        engine_factory: callable with signature
            ``(scenario, seed, iters) -> engine_instance``.
        engine_name_for_seed_json: stored in seed.json
            ("AhlfeldtABMEngine" etc.).
        shock_distribution_label: stored in seed.json
            ``shock_distribution`` field.
        extra_seed_fields: additional key/value pairs to record in
            seed.json.
        agent_mass_scale: whether to call ``scale_agent_mass`` on the
            population.
        L_override: whether to apply ``apply_pack_L_override``.

    Emits standard baseline-driver output files (``seed.json``,
    ``per_zone.csv``, ``diagnostics.json``) plus a ``shock_config.json``
    capturing the applied shock parameters + diagnostics.
    """
    from agent_urban_planning.research.berlin.three_version_driver import (
        prepare_scenario, scale_agent_mass, extract_results,
        write_seed_json, write_per_zone_csv,
        load_obs_fundamentals, infer_unit_from_scenario_path,
        apply_pack_L_override, PACK_WORKER_MASS,
    )
    from agent_urban_planning.core.engine import SimulationEngine

    output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = baseline_dir if baseline_dir.is_absolute() else project_root / baseline_dir
    baseline_per_zone = baseline_dir / "per_zone.csv"
    if not baseline_per_zone.exists():
        print(f"ERROR: baseline per_zone.csv not found at {baseline_per_zone}",
              file=sys.stderr)
        raise SystemExit(2)

    print(f"[{variant_name}-shock] start at {time.strftime('%H:%M:%S')} "
          f"seed={seed} variant={variant_name}", flush=True)
    print(f"[{variant_name}-shock] baseline: {baseline_per_zone}", flush=True)
    print(f"[{variant_name}-shock] shock:    {shock_config_path}", flush=True)

    # ---- Load scenario + agents (baseline setup) ------------------------
    from types import SimpleNamespace
    args_ns = SimpleNamespace(
        scenario=scenario_path, agents=agents_path, seed=seed, iters=iters,
        project_root=project_root,
    )
    sc, ag, root = prepare_scenario(args_ns, version_label=variant_name)
    unit = infer_unit_from_scenario_path(scenario_path)
    if L_override:
        L_total = apply_pack_L_override(sc, root, unit)
        print(f"[{variant_name}-shock] L = V × K^0.75 applied, Σ = {L_total:.0f}",
              flush=True)

    # ---- Warm-start: inject baseline Q, w ------------------------------
    zone_index = [z.name for z in sc.zones]
    Q_init, w_init = load_baseline_prices(baseline_per_zone, zone_index)
    sc = inject_initial_prices(sc, Q_init, w_init)
    print(f"[{variant_name}-shock] warm-started {len(Q_init)} zones from baseline",
          flush=True)

    # ---- Load shock config ---------------------------------------------
    shock_cfg = load_shock_config(shock_config_path, zone_names_csv)
    print(f"[{variant_name}-shock] shock: {shock_cfg.name} — "
          f"{len(shock_cfg.stations)} stations "
          f"({', '.join(shock_cfg.stations)}); "
          f"intra={shock_cfg.intra_station_min} min", flush=True)

    # ---- Build engine (variant-specific) + SimulationEngine ------------
    # Phase-aware factory call so V5 can route per-phase cache dirs.
    import inspect
    try:
        _sig = inspect.signature(engine_factory)
        if "phase" in _sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values()
        ):
            engine = engine_factory(sc, seed, iters, phase="shock")
        else:
            engine = engine_factory(sc, seed, iters)
    except (TypeError, ValueError):
        engine = engine_factory(sc, seed, iters)
    eng = SimulationEngine(
        scenario=sc, agent_config=ag, engine=engine,
        seed=seed, verbose=False,
    )
    if agent_mass_scale:
        scale_agent_mass(eng.population, PACK_WORKER_MASS)

    # Variant-specific pre-run work (LLM elicitation, clustering, etc.)
    # fires here so it happens AFTER scale_agent_mass but BEFORE shock
    # application + market clearing.
    if pre_run_hook is not None:
        pre_run_hook(eng, engine)

    # ---- Apply Route-C shock to env.transport_matrix -------------------
    if eng.base_env.transport_matrix is None:
        raise RuntimeError(
            f"{variant_name}-shock: base_env.transport_matrix is None — "
            f"shock cannot be applied."
        )
    baseline_tt = eng.base_env.transport_matrix.copy()
    tt_zone_index = eng.base_env.transport_matrix_index
    shocked_tt, tt_diag = apply_railway_shock_with_diagnostics(
        baseline_tt, tt_zone_index,
        shock_cfg.station_synthetic_ids, shock_cfg.intra_station_min,
    )
    eng.base_env.transport_matrix = shocked_tt.astype(
        baseline_tt.dtype, copy=False
    )
    print(f"[{variant_name}-shock] τ shock: "
          f"{tt_diag['n_pairs_reduced']}/{tt_diag['n_pairs_total']} pairs reduced "
          f"(mean {tt_diag['mean_reduction_min']:.2f} min, "
          f"max {tt_diag['max_reduction_min']:.2f} min)", flush=True)

    # ---- Run shocked market --------------------------------------------
    if post_run_hook is not None:
        post_run_hook(eng, engine)
    t_market = time.time()
    result = eng.run(policy=None)
    runtime_market = time.time() - t_market
    print(f"[{variant_name}-shock] market loop: {runtime_market:.1f}s",
          flush=True)

    # ---- Emit outputs ---------------------------------------------------
    sim_Q, sim_HR, sim_HM, sim_wage = extract_results(
        result, engine, zone_index, PACK_WORKER_MASS,
    )
    obs = load_obs_fundamentals(root, unit, zone_index)

    seed_extra = {
        "shock_config_path": str(shock_config_path),
        "baseline_dir": str(baseline_dir),
        "variant_name": variant_name,
        "shock_name": shock_cfg.name,
        "shock_stations": shock_cfg.stations,
        "shock_intra_station_min": shock_cfg.intra_station_min,
    }
    if extra_seed_fields:
        seed_extra.update(extra_seed_fields)

    write_seed_json(
        output_dir, seed=seed, engine_name=engine_name_for_seed_json,
        shock_distribution=shock_distribution_label,
        scenario=str(scenario_path),
        num_agents=getattr(engine, "num_agents", len(eng.population)),
        extra=seed_extra,
    )
    write_per_zone_csv(
        output_dir, zone_index, sim_Q, sim_HR, sim_HM, sim_wage,
        obs_Q=obs.get("Q"), obs_HR=obs.get("HR"),
        obs_HM=obs.get("HM"), obs_wage=obs.get("wage"),
    )
    abm_diag = getattr(engine, "last_abm_diagnostics", {}) or {}
    diag_payload = {
        "version": f"{variant_name}-shock-east-west-express",
        "runtime_market_s": runtime_market,
        "iterations_cap": iters,
        "scenario": str(scenario_path),
        "num_types": len(eng.population),
        "agent_mass": PACK_WORKER_MASS,
        "abm_diagnostics": abm_diag,
        "shock_tt_diagnostics": tt_diag,
    }
    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diag_payload, f, indent=2, default=float)
    with open(output_dir / "shock_config.json", "w") as f:
        json.dump({
            "name": shock_cfg.name,
            "description": shock_cfg.description,
            "intra_station_min": shock_cfg.intra_station_min,
            "stations": shock_cfg.stations,
            "station_synthetic_ids": shock_cfg.station_synthetic_ids,
            "station_roles": shock_cfg.station_roles,
            "tt_diagnostics": tt_diag,
        }, f, indent=2, default=float)

    print(f"[{variant_name}-shock] artifacts written to {output_dir}",
          flush=True)
