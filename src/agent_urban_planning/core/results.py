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
    """Per-agent record summarizing an agent's outcome in one run.

    Captures the agent's demographics, declared preferences, the
    zone-by-zone utilities seen at equilibrium, the realized choice
    and utility, the commute time, and (when a baseline is provided)
    the welfare difference relative to that baseline. Both the legacy
    ``zone_choice`` field and the explicit
    ``residence_zone``/``workplace_zone`` pair are populated; the
    legacy field equals ``residence_zone``.

    Attributes:
        agent_id: Stable agent identifier.
        weight: Population share assigned to this type.
        demographics: Flattened demographic snapshot (income,
            household size, etc.).
        preferences: Dict with keys ``alpha``, ``beta``, ``gamma``,
            ``delta`` reflecting the agent's recorded preference
            weights.
        zone_utilities: Per-zone utility evaluations.
        zone_choice: Residence zone (legacy alias for ``residence_zone``).
        equilibrium_price: Price paid at the chosen residence zone.
        commute_minutes: Realized commute time between residence and
            workplace.
        realized_utility: Utility at the chosen pair.
        utility_vs_baseline: Realized utility minus a baseline run's
            utility, when a baseline is supplied. ``None`` otherwise.
        residence_zone: Residence zone (matches ``zone_choice``).
        workplace_zone: Workplace zone. For Singapore scenarios equals
            ``demographics['job_location']``; for Berlin scenarios this
            is the engine's joint-choice output.

    Examples:
        >>> import agent_urban_planning as aup
        >>> # Returned in SimulationResults.agent_results; see SimulationEngine.run().
    """

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
        """Build an :class:`AgentResult` from an :class:`Agent` plus market output.

        Convenience constructor that flattens the agent's demographics
        and preferences into JSON-friendly dicts and fills in the
        residence/workplace fields from ``zone_choice`` and either an
        explicit ``workplace_zone`` or ``agent.job_location``.

        Args:
            agent: Source :class:`Agent` whose demographics are
                snapshotted.
            zone_utilities: Per-zone utility evaluations as seen by
                the agent at the equilibrium prices.
            zone_choice: The agent's chosen residence zone.
            equilibrium_price: Equilibrium price at the chosen zone.
            commute_minutes: Realized commute time.
            realized_utility: Utility at the chosen residence and
                workplace.
            utility_vs_baseline: Optional difference vs. a baseline run.
            workplace_zone: Optional workplace zone override; defaults
                to ``agent.job_location``.

        Returns:
            A new :class:`AgentResult` populated from the inputs.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # Used internally by SimulationEngine.run() — see that method.
        """
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
    """Top-level container for one simulation run's output.

    Aggregates the population-level :class:`WelfareMetrics`, the list of
    per-agent :class:`AgentResult` records, the price-history trajectory,
    and the run :class:`RunMetadata`. Returned by
    :meth:`SimulationEngine.run` and used directly by analysis utilities
    such as :func:`agent_urban_planning.analysis` plotting helpers.

    Args:
        metrics: Welfare metrics for the run.
        agent_results: Per-agent records.
        policy_name: Name of the policy that produced this run.
        scenario_name: Name of the scenario.
        price_history: Optional list of per-iteration market snapshots.
        metadata: Optional :class:`RunMetadata` with reproducibility
            info (LLM provider/model, seed, cluster config, etc.).

    Examples:
        >>> import agent_urban_planning as aup
        >>> # results = sim.run(policy)  # doctest: +SKIP
        >>> # results.metrics.avg_utility  # doctest: +SKIP
        >>> # results.get_agent(0).realized_utility  # doctest: +SKIP
    """

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
        """Look up the per-agent result record by id.

        Args:
            agent_id: Stable integer identifier of the agent.

        Returns:
            The :class:`AgentResult` for that agent.

        Raises:
            KeyError: If ``agent_id`` is not in the population.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # results.get_agent(0).realized_utility  # doctest: +SKIP
        """
        if agent_id not in self._index:
            raise KeyError(f"Agent {agent_id} not found in results")
        return self._index[agent_id]

    def filter_agents(self, **criteria) -> list[AgentResult]:
        """Filter agent results by demographic criteria.

        Returns the subset of :class:`AgentResult` records matching all
        of the supplied filter criteria. Useful for cohort-level
        analyses (e.g. "results for households with children").

        Args:
            **criteria: Keyword filters. Supported keys:

                * ``has_children`` (bool)
                * ``has_elderly`` (bool)
                * ``car_owner`` (bool)
                * ``income_min`` / ``income_max`` (float)
                * ``age_min`` / ``age_max`` (int)
                * ``zone_choice`` (str)
                * ``job_location`` (str)

        Returns:
            List of :class:`AgentResult` records satisfying every
            criterion. Empty when no agent matches.

        Raises:
            ValueError: If an unsupported criterion key is supplied.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # families = results.filter_agents(has_children=True)  # doctest: +SKIP
            >>> # high_income = results.filter_agents(income_min=10000)  # doctest: +SKIP
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
        """Return a JSON-serializable dict representation of this run.

        Returns:
            ``dict`` with keys ``policy_name``, ``scenario_name``,
            ``metrics``, ``agent_results``, ``price_history``, and
            ``metadata``.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # d = results.to_dict()  # doctest: +SKIP
            >>> # list(d)
            ['policy_name', 'scenario_name', 'metrics', 'agent_results',
             'price_history', 'metadata']
        """
        return {
            "policy_name": self.policy_name,
            "scenario_name": self.scenario_name,
            "metrics": self.metrics.to_dict(),
            "agent_results": [asdict(r) for r in self.agent_results],
            "price_history": self.price_history,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }

    def to_json(self) -> str:
        """Serialize this run to an indented JSON string.

        Returns:
            JSON string suitable for writing to disk.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # path.write_text(results.to_json())  # doctest: +SKIP
        """
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "SimulationResults":
        """Reconstruct a :class:`SimulationResults` from its JSON form.

        Args:
            json_str: JSON string previously produced by :meth:`to_json`.

        Returns:
            The deserialized :class:`SimulationResults`.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # restored = aup.SimulationResults.from_json(path.read_text())
        """
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
