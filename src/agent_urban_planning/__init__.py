"""agent-urban-planning — agent-based urban planning simulator.

Companion to the NeurIPS Datasets & Benchmarks 2026 paper. Provides closed-form,
hybrid, and full-LLM decision engines for spatial-equilibrium ABM simulation.

Quickstart::

    import agent_urban_planning as aup

    scenario = aup.data.builtin.load("singapore_real_v2")
    agents = aup.data.builtin.load_agents("singapore_real_v2")
    engine = aup.UtilityEngine(scenario.ahlfeldt_params, mode="softmax")
    sim = aup.SimulationEngine(scenario=scenario, agent_config=agents, engine=engine)
    results = sim.run(policy=None)

See the documentation at https://agent-urban-planning.readthedocs.io for full API
reference, tutorials, and reproducibility instructions for the paper's V1-V5.4
results.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "__version__",
    # Core re-exports — populated in Phase 2 as core/ modules are ported.
    # "Environment",
    # "AgentPopulation",
    # "Market",
    # "SimulationEngine",
    # "WelfareMetrics",
    # "Results",
    # Decision-engine re-exports — populated in Phase 3.
    # "DecisionEngine",
    # "UtilityEngine",
    # "HybridDecisionEngine",
    # "LLMDecisionEngine",
]
