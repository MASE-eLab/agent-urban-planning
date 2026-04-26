#!/usr/bin/env python3
"""Berlin V2 — Baseline-ABM argmax (Fréchet) replication.

Configuration::

    aup.UtilityEngine(params, mode="argmax", noise="frechet")

See ``run_v1_softmax.py`` for the shared workflow.
"""
from __future__ import annotations

import argparse

import agent_urban_planning as aup

from _common import run_baseline_and_shock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--num-agents", type=int, default=1_000_000,
                        help="Monte Carlo replicate count (default 1,000,000).")
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()

    def engine_factory(sc, seed, iters):
        return aup.UtilityEngine(
            sc.ahlfeldt_params,
            mode="argmax",
            noise="frechet",
            num_agents=args.num_agents,
            batch_size=args.batch_size,
            seed=seed,
            dtype=getattr(sc.ahlfeldt_params, "dtype", "float64"),
        )

    run_baseline_and_shock(
        engine_factory,
        output_subdir="berlin_v2_argmax_frechet",
        iters=args.iters,
        seed=args.seed,
        engine_name_for_seed_json="AhlfeldtABMEngine",
        shock_distribution="frechet",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
