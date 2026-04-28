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

See the documentation at https://anonymous.4open.science/r/agent-urban-planning-4B4D for full API
reference, tutorials, and reproducibility instructions for the paper's V1-V5
results.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Core re-exports.
from agent_urban_planning.core.engine import SimulationEngine
from agent_urban_planning.core.environment import Environment, Zone
from agent_urban_planning.core.agents import Agent, AgentPopulation, PreferenceWeights, persona_summary
from agent_urban_planning.core.market import (
    HousingMarket,
    AhlfeldtMarket,
    MarketResult,
)
from agent_urban_planning.core.metrics import WelfareMetrics, compute_metrics
from agent_urban_planning.core.results import SimulationResults, AgentResult
from agent_urban_planning.core.run_metadata import RunMetadata

# Decision-engine re-exports — Phase 3 consolidated API.
from agent_urban_planning.decisions.base import (
    DecisionEngine,
    LocationChoice,
    ZoneChoice,
)
from agent_urban_planning.decisions.utility import UtilityEngine
from agent_urban_planning.decisions.hybrid import HybridDecisionEngine
from agent_urban_planning.decisions.llm import LLMDecisionEngine

# Underlying paper-internal classes (advanced users only — the public
# API above is the recommended entry point for V1/V2/V3/V4/V5).
from agent_urban_planning.decisions.ahlfeldt_utility import AhlfeldtUtilityEngine
from agent_urban_planning.decisions.ahlfeldt_abm_engine import AhlfeldtABMEngine
from agent_urban_planning.decisions.ahlfeldt_argmax_hybrid_engine import (
    AhlfeldtArgmaxHybridEngine,
)
from agent_urban_planning.decisions.ahlfeldt_hierarchical_llm_engine import (
    AhlfeldtHierarchicalLLMEngine,
)
from agent_urban_planning.decisions._legacy_singapore_utility import (
    UtilityEngine as _LegacySingaporeUtilityEngine,
)

# Subpackages re-exported as namespaces.
from agent_urban_planning import core as core
from agent_urban_planning import decisions as decisions
from agent_urban_planning import data as data
from agent_urban_planning import analysis as analysis
from agent_urban_planning import research as research
from agent_urban_planning import llm as llm

__all__ = [
    "__version__",
    # Core
    "SimulationEngine",
    "Environment", "Zone",
    "Agent", "AgentPopulation", "PreferenceWeights", "persona_summary",
    "HousingMarket", "AhlfeldtMarket", "MarketResult",
    "WelfareMetrics", "compute_metrics",
    "SimulationResults", "AgentResult",
    "RunMetadata",
    # Decisions — public API
    "DecisionEngine", "LocationChoice", "ZoneChoice",
    "UtilityEngine", "HybridDecisionEngine", "LLMDecisionEngine",
    # Decisions — underlying paper-internal (advanced)
    "AhlfeldtUtilityEngine", "AhlfeldtABMEngine",
    "AhlfeldtArgmaxHybridEngine", "AhlfeldtHierarchicalLLMEngine",
    # Subpackages
    "core", "decisions", "data", "analysis", "research", "llm",
]
