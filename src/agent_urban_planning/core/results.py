"""Per-agent result tracking and simulation results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from agent_urban_planning.core.agents import Agent, PreferenceWeights
from agent_urban_planning.core.metrics import WelfareMetrics
from agent_urban_planning.core.run_metadata import RunMetadata


@dataclass
class AgentResult:
    agent_id: int
    weight: float
    demographics: dict[str, Any]  # flattened agent demographics
    preferences: dict[str, float]  # alpha, beta, gamma, delta
    zone_utilities: dict[str, float]
    zone_choice: str  # retained as residence zone (legacy alias)
    equilibrium_price: float
    commute_minutes: float
    realized_utility: float
    utility_vs_baseline: Optional[float] = None
    # Added in berlin-replication-abm: explicit residence / workplace fields.
    # For Singapore scenarios, workplace_zone == demographics['job_location'].
    # For Berlin scenarios, workplace_zone is the engine's joint choice output.
    residence_zone: str = ""
    workplace_zone: str = ""

    @classmethod
    def from_agent(
        cls,
        agent: Agent,
        zone_utilities: dict[str, float],
        zone_choice: str,
        equilibrium_price: float,
        commute_minutes: float,
        realized_utility: float,
        utility_vs_baseline: Optional[float] = None,
        workplace_zone: Optional[str] = None,
    ) -> "AgentResult":
        wp = workplace_zone if workplace_zone is not None else agent.job_location
        return cls(
            agent_id=agent.agent_id,
            weight=agent.weight,
            demographics={
                "household_size": agent.household_size,
                "age_head": agent.age_head,
                "has_children": agent.has_children,
                "has_elderly": agent.has_elderly,
                "income": agent.income,
                "savings": agent.savings,
                "job_location": agent.job_location,
                "car_owner": agent.car_owner,
            },
            preferences={
                "alpha": agent.preferences.alpha,
                "beta": agent.preferences.beta,
                "gamma": agent.preferences.gamma,
                "delta": agent.preferences.delta,
            },
            zone_utilities=zone_utilities,
            zone_choice=zone_choice,
            equilibrium_price=equilibrium_price,
            commute_minutes=commute_minutes,
            realized_utility=realized_utility,
            utility_vs_baseline=utility_vs_baseline,
            residence_zone=zone_choice,
            workplace_zone=wp,
        )


class SimulationResults:
    """Holds aggregate metrics and per-agent result records."""

    def __init__(
        self,
        metrics: WelfareMetrics,
        agent_results: list[AgentResult],
        policy_name: str = "",
        scenario_name: str = "",
        price_history: Optional[list[dict]] = None,
        metadata: Optional[RunMetadata] = None,
    ):
        self.metrics = metrics
        self.agent_results = agent_results
        self.policy_name = policy_name
        self.scenario_name = scenario_name
        self.price_history = price_history or []  # list of {iteration, prices, demand, excess_demand}
        self.metadata = metadata
        self._index = {r.agent_id: r for r in agent_results}

    def get_agent(self, agent_id: int) -> AgentResult:
        """Get result for a specific agent. Raises KeyError if not found."""
        if agent_id not in self._index:
            raise KeyError(f"Agent {agent_id} not found in results")
        return self._index[agent_id]

    def filter_agents(self, **criteria) -> list[AgentResult]:
        """Filter agents by demographic criteria.

        Supported criteria:
            has_children, has_elderly, car_owner (bool)
            income_min, income_max (float)
            age_min, age_max (int)
            zone_choice (str)
            job_location (str)
        """
        results = []
        for r in self.agent_results:
            match = True
            for key, value in criteria.items():
                if key == "zone_choice":
                    if r.zone_choice != value:
                        match = False
                elif key == "income_min":
                    if r.demographics["income"] < value:
                        match = False
                elif key == "income_max":
                    if r.demographics["income"] > value:
                        match = False
                elif key == "age_min":
                    if r.demographics["age_head"] < value:
                        match = False
                elif key == "age_max":
                    if r.demographics["age_head"] > value:
                        match = False
                elif key in r.demographics:
                    if r.demographics[key] != value:
                        match = False
                else:
                    raise ValueError(f"Unknown filter criterion: {key}")
            if match:
                results.append(r)
        return results

    def to_dict(self) -> dict:
        return {
            "policy_name": self.policy_name,
            "scenario_name": self.scenario_name,
            "metrics": self.metrics.to_dict(),
            "agent_results": [asdict(r) for r in self.agent_results],
            "price_history": self.price_history,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "SimulationResults":
        data = json.loads(json_str)
        metrics = WelfareMetrics.from_dict(data["metrics"])
        agent_results = [AgentResult(**r) for r in data["agent_results"]]
        meta_dict = data.get("metadata")
        metadata = RunMetadata.from_dict(meta_dict) if meta_dict else None
        return cls(
            metrics=metrics,
            agent_results=agent_results,
            policy_name=data.get("policy_name", ""),
            scenario_name=data.get("scenario_name", ""),
            price_history=data.get("price_history", []),
            metadata=metadata,
        )
