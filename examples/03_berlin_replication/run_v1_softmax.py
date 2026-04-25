#!/usr/bin/env python3
"""Berlin V1 — Baseline-softmax replication.

Reproduces the V1 baseline (closed-form Cobb-Douglas + Fréchet softmax over
all 96 zones) + East-West Express shock at seed 42.

Expected wall-clock: ~3 hr.
Expected output: ``output/berlin_v1_softmax/per_zone.csv`` and
``output/berlin_v1_softmax_shock_east_west/per_zone.csv``.

Configuration::

    aup.UtilityEngine(params, mode="softmax", deterministic=True)
"""
from __future__ import annotations

import argparse

import agent_urban_planning as aup

from _common import run_baseline_and_shock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    def engine_factory(sc, seed, iters):
        # ``deterministic=True`` matches the V1 paper run — the
        # continuum-limit softmax path that populates
        # ``last_choice_probabilities`` so HR/HM extraction works.
        return aup.UtilityEngine(
            sc.ahlfeldt_params,
            mode="softmax",
            seed=seed,
            deterministic=True,
            dtype=getattr(sc.ahlfeldt_params, "dtype", "float64"),
        )

    run_baseline_and_shock(
        engine_factory,
        output_subdir="berlin_v1_softmax",
        iters=args.iters,
        seed=args.seed,
        engine_name_for_seed_json="AhlfeldtUtilityEngine",
        shock_distribution="softmax",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
