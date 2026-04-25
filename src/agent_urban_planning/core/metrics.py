"""Welfare metrics computation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from agent_urban_planning.core.agents import Agent, AgentPopulation
from agent_urban_planning.decisions.base import LocationChoice, ZoneChoice
from agent_urban_planning.core.environment import Environment


@dataclass
class FacilityUtilization:
    zone: str
    facility_type: str
    capacity: int
    demand: float
    utilization: float  # demand / capacity


@dataclass
class WelfareMetrics:
    """Aggregate welfare metrics summarizing one simulation run.

    Captures both classical welfare measures (average utility, weighted
    Gini, Rawlsian min/max) and the planning-relevant indicators (commute
    times, housing affordability, facility utilization, zone-level
    population, employment, wages, prices). Populated by
    :func:`compute_metrics` from market output.

    Attributes:
        avg_utility: Population-weighted mean utility.
        gini_coefficient: Weighted Gini coefficient of utility values
            (0 = perfect equality, 1 = perfect inequality). Negative
            utilities are shifted to non-negative before computation.
        min_utility: Smallest realized utility (Rawlsian floor).
        max_utility: Largest realized utility.
        avg_commute_minutes: Weighted mean commute time.
        long_commute_share: Share of population with commute exceeding
            ``long_commute_threshold`` minutes.
        housing_unaffordable_share: Share of population whose
            price/income ratio exceeds ``affordability_threshold``.
        zone_populations: Mapping ``zone -> share`` of total population
            keyed by residence zone.
        zone_prices: Mapping ``zone -> equilibrium price``.
        facility_utilization: Per-facility utilization (demand /
            capacity) records.
        market_converged: Whether the market clearer reached
            convergence.
        market_convergence_metric: Final residual at termination.
        zone_employment: Mapping ``zone -> share`` keyed by workplace.
        zone_wages: Mapping ``zone -> wage`` (Ahlfeldt scenarios only).

    Examples:
        >>> import agent_urban_planning as aup
        >>> # Typically created by aup.compute_metrics(); see the quickstart tutorial.
        >>> # results.metrics.avg_utility
    """

    avg_utility: float
    gini_coefficient: float
    min_utility: float
    max_utility: float
    avg_commute_minutes: float
    long_commute_share: float  # share with commute > threshold
    housing_unaffordable_share: float  # share spending > 30% income
    zone_populations: dict[str, float]  # zone → population weight (by residence)
    zone_prices: dict[str, float]  # zone → equilibrium price
    facility_utilization: list[FacilityUtilization] = field(default_factory=list)
    market_converged: bool = True
    market_convergence_metric: float = 0.0
    # Added in berlin-replication-abm: zone → population weight keyed by workplace.
    # Populated from LocationChoice.workplace. For Singapore scenarios this
    # equals the aggregate of agent.job_location across the population;
    # for Berlin scenarios it reflects the engine's joint R×W choice.
    zone_employment: dict[str, float] = field(default_factory=dict)
    # Added in berlin-replication-abm: zone → wage (only populated in Berlin).
    zone_wages: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict copy of every field.

        Returns:
            Plain ``dict`` produced by :func:`dataclasses.asdict`.

        Examples:
            >>> from agent_urban_planning.core.metrics import WelfareMetrics
            >>> m = WelfareMetrics(
            ...     avg_utility=1.0, gini_coefficient=0.2, min_utility=0.0,
            ...     max_utility=2.0, avg_commute_minutes=20.0,
            ...     long_commute_share=0.1, housing_unaffordable_share=0.05,
            ...     zone_populations={}, zone_prices={},
            ... )
            >>> m.to_dict()["avg_utility"]
            1.0
        """
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to a pretty-printed JSON string.

        Returns:
            Indented JSON string suitable for writing to disk.

        Examples:
            >>> from agent_urban_planning.core.metrics import WelfareMetrics
            >>> m = WelfareMetrics(
            ...     avg_utility=1.0, gini_coefficient=0.2, min_utility=0.0,
            ...     max_utility=2.0, avg_commute_minutes=20.0,
            ...     long_commute_share=0.1, housing_unaffordable_share=0.05,
            ...     zone_populations={}, zone_prices={},
            ... )
            >>> "avg_utility" in m.to_json()
            True
        """
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "WelfareMetrics":
        """Reconstruct a :class:`WelfareMetrics` from its serialized dict.

        Args:
            data: Dict shaped like the output of :meth:`to_dict`.

        Returns:
            A :class:`WelfareMetrics` instance with
            ``facility_utilization`` records re-hydrated from their
            dict form.

        Examples:
            >>> from agent_urban_planning.core.metrics import WelfareMetrics
            >>> m = WelfareMetrics(
            ...     avg_utility=1.0, gini_coefficient=0.2, min_utility=0.0,
            ...     max_utility=2.0, avg_commute_minutes=20.0,
            ...     long_commute_share=0.1, housing_unaffordable_share=0.05,
            ...     zone_populations={}, zone_prices={},
            ... )
            >>> WelfareMetrics.from_dict(m.to_dict()).avg_utility
            1.0
        """
        fu_data = data.pop("facility_utilization", [])
        facility_utilization = [FacilityUtilization(**f) for f in fu_data]
        return cls(facility_utilization=facility_utilization, **data)


def compute_weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted Gini coefficient using the standard formula.

    If values contain negatives (common for utility), they are shifted
    so the minimum becomes zero before computing the Gini. This is the
    standard approach for welfare Gini with utility values that can be
    negative (see Atkinson 1970, Sen 1997).
    """
    if len(values) == 0 or weights.sum() == 0:
        return 0.0

    # Shift to non-negative if needed (Gini is undefined for negative values)
    shifted = values.copy()
    if shifted.min() < 0:
        shifted = shifted - shifted.min()

    # Sort by value
    sorted_indices = np.argsort(shifted)
    sorted_values = shifted[sorted_indices]
    sorted_weights = weights[sorted_indices]

    total_w = sorted_weights.sum()
    weighted_mean = np.average(sorted_values, weights=sorted_weights)

    if weighted_mean == 0:
        return 0.0

    # Weighted mean absolute difference
    # Gini = sum_i sum_j w_i w_j |x_i - x_j| / (2 * W^2 * mean)
    n = len(sorted_values)
    numerator = 0.0
    for i in range(n):
        for j in range(n):
            numerator += sorted_weights[i] * sorted_weights[j] * abs(sorted_values[i] - sorted_values[j])

    gini = numerator / (2.0 * total_w * total_w * weighted_mean)
    return max(0.0, min(1.0, gini))


def compute_metrics(
    population: AgentPopulation,
    environment: Environment,
    allocations: dict[int, ZoneChoice],
    prices: dict[str, float],
    long_commute_threshold: float = 60.0,
    affordability_threshold: float = 0.30,
    market_converged: bool = True,
    market_convergence_metric: float = 0.0,
    wages: Optional[dict[str, float]] = None,
) -> WelfareMetrics:
    """Compute aggregate welfare metrics from per-agent allocations.

    Walks the population, looks up each agent's chosen zone and route,
    and aggregates weighted means, the Gini coefficient, Rawlsian
    extremes, commute statistics, affordability share, facility
    utilization, and per-zone population/employment/wage tables.

    Args:
        population: The simulated :class:`AgentPopulation`.
        environment: The :class:`Environment` used by the simulation.
        allocations: Dict mapping ``agent_id`` to the
            :class:`ZoneChoice` produced by the market clearer.
        prices: Mapping ``zone -> equilibrium price``.
        long_commute_threshold: Minutes threshold for the
            ``long_commute_share`` indicator. Defaults to ``60.0``.
        affordability_threshold: Price/income ratio above which an
            agent's housing counts as unaffordable. Defaults to
            ``0.30``.
        market_converged: Whether the upstream market clearer reached
            convergence. Recorded in the result.
        market_convergence_metric: Final residual to record.
        wages: Optional ``zone -> wage`` mapping (Ahlfeldt scenarios).

    Returns:
        A :class:`WelfareMetrics` describing the run's welfare outcome.

    Examples:
        >>> import agent_urban_planning as aup
        >>> # metrics = aup.compute_metrics(population, env, allocations, prices)
        >>> # metrics.gini_coefficient
    """
    utilities = []
    weights = []
    commute_times = []
    housing_ratios = []
    zone_pops: dict[str, float] = {name: 0.0 for name in environment.zone_names}
    zone_emp: dict[str, float] = {name: 0.0 for name in environment.zone_names}

    for agent in population:
        choice = allocations[agent.agent_id]
        zone_name = choice.zone_name  # backward-compat alias for residence
        utility = choice.utility

        utilities.append(utility)
        weights.append(agent.weight)
        zone_pops[zone_name] += agent.weight

        # Workplace resolution priority:
        #   1. agent.job_location if present (Singapore-style — agent attribute is authoritative)
        #   2. LocationChoice.workplace if agent has no job_location (Berlin-style joint choice)
        #   3. fall back to residence zone
        workplace = agent.job_location or getattr(choice, "workplace", "") or zone_name
        if workplace in zone_emp:
            zone_emp[workplace] += agent.weight

        # Commute time based on resolved workplace
        route = environment.transport.get_best_route(zone_name, workplace)
        commute = route.time_minutes if route else 120.0
        commute_times.append(commute)

        # Housing ratio
        price = prices.get(zone_name, 0.0)
        ratio = price / agent.income if agent.income > 0 else 1.0
        housing_ratios.append(ratio)

    utilities_arr = np.array(utilities)
    weights_arr = np.array(weights)
    commute_arr = np.array(commute_times)
    ratio_arr = np.array(housing_ratios)

    # Weighted averages
    avg_utility = float(np.average(utilities_arr, weights=weights_arr))
    avg_commute = float(np.average(commute_arr, weights=weights_arr))
    long_commute_share = float(
        np.average(commute_arr > long_commute_threshold, weights=weights_arr)
    )
    unaffordable_share = float(
        np.average(ratio_arr > affordability_threshold, weights=weights_arr)
    )

    # Gini
    gini = compute_weighted_gini(utilities_arr, weights_arr)

    # Facility utilization
    facility_util = []
    for zone_name, zone in environment.zones.items():
        pop_in_zone = zone_pops[zone_name]
        for facility in zone.facilities:
            demand = pop_in_zone * sum(a.weight for a in population)  # approximate
            # Use population share as a proxy for demand
            util_rate = pop_in_zone / (facility.capacity / sum(
                z.housing_supply for z in environment.zones.values()
            )) if facility.capacity > 0 else 0.0
            facility_util.append(FacilityUtilization(
                zone=zone_name,
                facility_type=facility.type,
                capacity=facility.capacity,
                demand=pop_in_zone,
                utilization=util_rate,
            ))

    return WelfareMetrics(
        avg_utility=avg_utility,
        gini_coefficient=gini,
        min_utility=float(utilities_arr.min()),
        max_utility=float(utilities_arr.max()),
        avg_commute_minutes=avg_commute,
        long_commute_share=long_commute_share,
        housing_unaffordable_share=unaffordable_share,
        zone_populations=zone_pops,
        zone_prices=dict(prices),
        facility_utilization=facility_util,
        market_converged=market_converged,
        market_convergence_metric=market_convergence_metric,
        zone_employment=zone_emp,
        zone_wages=dict(wages) if wages else {},
    )
