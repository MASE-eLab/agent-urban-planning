"""Berlin replication helpers (Ahlfeldt et al. 2015).

This subpackage contains paper-specific implementations:
  * Ahlfeldt-style closed-form market clearing (Cobb-Douglas + Fréchet).
  * East-West Express τ shock application (Route-C min-of-paths).
  * Block-to-zone aggregation utilities.
  * Calibration helpers.

These modules are imported by the
:class:`agent_urban_planning.UtilityEngine`,
:class:`HybridDecisionEngine`, and :class:`LLMDecisionEngine` public
classes to provide Berlin-specific behavior. End users typically do not
import them directly; the example scripts at
``examples/02_berlin_replication/`` use them via
:mod:`agent_urban_planning.research.berlin.shock_pipeline`.
"""
from agent_urban_planning.research.berlin.three_version_driver import (
    PACK_WORKER_MASS,
    add_common_cli,
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
from agent_urban_planning.research.berlin.railway_shock import (
    apply_railway_shock,
    apply_railway_shock_with_diagnostics,
)
from agent_urban_planning.research.berlin.shock_config import (
    RailwayShockConfig,
    load_shock_config,
    load_zone_name_map,
)
from agent_urban_planning.research.berlin.warm_start import (
    inject_initial_prices,
    load_baseline_prices,
)
from agent_urban_planning.research.berlin.shock_pipeline import (
    run_shock_pipeline,
)

__all__ = [
    "PACK_WORKER_MASS",
    "add_common_cli",
    "apply_pack_L_override",
    "apply_railway_shock",
    "apply_railway_shock_with_diagnostics",
    "extract_results",
    "infer_unit_from_scenario_path",
    "inject_initial_prices",
    "load_baseline_prices",
    "load_obs_fundamentals",
    "load_shock_config",
    "load_zone_name_map",
    "prepare_scenario",
    "RailwayShockConfig",
    "run_shock_pipeline",
    "scale_agent_mass",
    "write_diagnostics",
    "write_per_zone_csv",
    "write_seed_json",
]
