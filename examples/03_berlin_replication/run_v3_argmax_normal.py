#!/usr/bin/env python3
"""Berlin V3 — Normal-ABM argmax (Gaussian) replication.

Configuration::

    aup.UtilityEngine(params, mode="argmax", noise="normal")

See ``run_v1_softmax.py`` for the shared workflow.
"""
from __future__ import annotations

import agent_urban_planning as aup
from agent_urban_planning.data.loaders import load_scenario

from _common import (
    BERLIN_SCENARIO_YAML,
    check_berlin_data_present,
    run_baseline_and_shock,
)


def main() -> int:
    check_berlin_data_present()
    sc = load_scenario(BERLIN_SCENARIO_YAML)
    engine = aup.UtilityEngine(
        sc.ahlfeldt_params,
        mode="argmax",
        noise="normal",
        num_agents=1_000_000,
        seed=42,
    )
    print(f"Engine: {engine!r}")
    run_baseline_and_shock(engine, output_subdir="berlin_v3_argmax_normal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
