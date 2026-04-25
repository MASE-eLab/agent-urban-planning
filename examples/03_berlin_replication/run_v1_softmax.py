#!/usr/bin/env python3
"""Berlin V1 — Baseline-softmax replication.

Reproduces the V1 baseline (closed-form Cobb-Douglas + Fréchet softmax over
all 96 zones) + East-West Express shock at seed 42.

Expected wall-clock: ~3 hr.
Expected output: ``output/berlin_v1_softmax/per_zone.csv`` and
``output/berlin_v1_shock_east_west/per_zone.csv``.

Configuration::

    aup.UtilityEngine(params, mode="softmax")
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
    engine = aup.UtilityEngine(sc.ahlfeldt_params, mode="softmax")
    print(f"Engine: {engine!r}")
    run_baseline_and_shock(engine, output_subdir="berlin_v1_softmax")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
