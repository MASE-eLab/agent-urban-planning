#!/usr/bin/env python3
"""Quickstart: full simulation pipeline with V1 (Baseline-softmax).

Walks through the five workflow stages from the paper's Figure 1
(Environment → Agents → Decision → Market → Equilibrium) and runs a
single baseline simulation with no policy shock.

This is the **Tier 2 reproducibility check**: if this script runs
green, your install is healthy and the paper's pipeline is end-to-end
usable from ``agent_urban_planning`` as a library.

Wall-clock: ~10 seconds. Uses the bundled Berlin Ortsteile scenario
(96 zones), heavily downsized for fast iteration:

  - num_agents: 1,000 (paper uses 1,000,000)
  - market iterations: 5 (paper uses 50)
  - decision engine: V1 closed-form softmax (deterministic)

For the paper-faithful Berlin V1-V5 reproduction, see
``examples/02_berlin_replication/``.
"""
from __future__ import annotations

from pathlib import Path

import agent_urban_planning as aup
from agent_urban_planning.data.loaders import load_scenario, load_agents


REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_YAML = REPO_ROOT / "data" / "berlin" / "scenarios" / "berlin_2006_ortsteile.yaml"
AGENTS_YAML = REPO_ROOT / "data" / "berlin" / "agents" / "berlin_ortsteile_richer_10k.yaml"


def main() -> int:
    print(f"agent-urban-planning version: {aup.__version__}")
    print()

    # ── Stage 1: Environment ────────────────────────────────────────────
    # The Environment holds zones (housing supply, amenities, productivity,
    # observed wages/floor prices) and the transport network. We load it
    # from a scenario YAML — same one used by the paper's V1-V5 runs.
    print("[1/5] Loading Environment from scenario YAML...")
    scenario = load_scenario(SCENARIO_YAML)
    print(f"      zones: {len(scenario.zones)}, scenario: {scenario.name}")

    # ── Stage 2: Agents ─────────────────────────────────────────────────
    # The AgentPopulation is a heterogeneous household population sampled
    # from per-zone demographic distributions (income, age, household
    # size, etc.). We downsize to 1,000 types for the quickstart; the
    # paper uses 1,000,000.
    print("[2/5] Loading AgentPopulation (downsized for quickstart)...")
    agent_config = load_agents(AGENTS_YAML)
    agent_config.num_types = 1_000
    print(f"      agent types: {agent_config.num_types}")

    # ── Stage 3: Decision engine ────────────────────────────────────────
    # V1 = closed-form Cobb-Douglas + Fréchet softmax (deterministic,
    # fastest of the 5 paper variants). Configured via constructor kwargs;
    # no subclassing needed. See examples/02_berlin_replication/ for the
    # other variants (V2/V3 argmax, V4 hybrid LLM-elicited, V5 full LLM).
    print("[3/5] Constructing DecisionEngine (V1 — closed-form softmax)...")
    engine = aup.UtilityEngine(scenario.ahlfeldt_params, mode="softmax")
    print(f"      engine: {engine!r}")

    # ── Stage 4: SimulationEngine + market clearing ─────────────────────
    # The SimulationEngine orchestrates the pipeline: it builds the
    # market clearer (AhlfeldtMarket for Berlin scenarios — joint
    # tatonnement on housing prices Q and wages w), invokes the engine
    # to compute choice probabilities, and iterates until convergence.
    # `policy=None` runs the baseline (no policy shock).
    print("[4/5] Running SimulationEngine baseline (5 market iterations)...")
    scenario.simulation.market_max_iterations = 5
    sim = aup.SimulationEngine(
        scenario=scenario,
        agent_config=agent_config,
        engine=engine,
        seed=42,
    )
    result = sim.run(policy=None)

    # ── Stage 5: Equilibrium results ────────────────────────────────────
    # `result` is a SimulationResults containing welfare metrics, per-agent
    # traces, the price-trajectory history, and run metadata.
    # NOTE: 5 iterations is too few for the market to converge — this run
    # only illustrates the pipeline shape, not paper-faithful numbers.
    # For real reproduction see examples/02_berlin_replication/.
    print("[5/5] Pipeline complete. Welfare metrics from this short run:")
    m = result.metrics
    print(f"      avg utility:           {m.avg_utility:+.4f}")
    print(f"      Gini (utility):         {m.gini_coefficient:.4f}")
    print()
    sample_zones = sorted(m.zone_prices.items(), key=lambda kv: -kv[1])[:3]
    print("      top 3 zones by clearing price (Q):")
    for zone, price in sample_zones:
        pop_share = m.zone_populations.get(zone, 0.0)
        print(f"        {zone}: Q={price:.4f}, pop share={pop_share:.4f}")

    print()
    print("This is an API-surface walkthrough; numbers are NOT paper-faithful")
    print("(only 5 market iterations, 1k agents, no L-override).")
    print("For paper-faithful V1-V5 reproduction at full scale, see")
    print("examples/02_berlin_replication/ (Tier 3+4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
