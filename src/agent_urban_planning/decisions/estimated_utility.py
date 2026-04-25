"""Decision engine using empirically-estimated conditional logit coefficients.

Uses beta coefficients from ``estimation.py`` (either estimated from HDB
transaction data or from literature-fallback values) to compute indirect
utility for each zone:

    V_z = beta_1 * (price_z / income)
        + beta_2 * commute_z
        + beta_3 * facilities_density_z
        + beta_4 * amenity_z

The agent chooses the zone with the highest V_z from their affordable set.
"""

from __future__ import annotations

from agent_urban_planning.core.agents import Agent
from agent_urban_planning.core.constraints import affordable_zones
from agent_urban_planning.decisions.base import ZoneChoice
from agent_urban_planning.decisions.estimation import EstimationResult, literature_fallback
from agent_urban_planning.core.environment import Environment


class EstimatedUtilityEngine:
    """Decision engine using empirically-estimated beta coefficients.

    Loads an ``EstimationResult`` from a JSON file (or uses the literature
    fallback) and evaluates the conditional logit utility function for
    each zone in the agent's affordable choice set.
    """

    def __init__(
        self,
        coefficients_path: str = "config/estimated_coefficients.json",
        budget_constraint: bool = True,
    ):
        try:
            self._result = EstimationResult.from_file(coefficients_path)
        except (FileNotFoundError, OSError):
            self._result = literature_fallback()

        self._beta_price_income = self._result.beta_price_income_ratio
        self._beta_commute = self._result.beta_commute_minutes
        self._beta_facilities = self._result.beta_facilities_per_capita
        self._beta_amenity = self._result.beta_amenity
        self._eta = self._result.price_elasticity_eta
        self.budget_constraint = budget_constraint

    def decide(
        self,
        agent: Agent,
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> ZoneChoice:
        """Choose the zone with highest estimated utility for the agent."""
        # Budget constraint: filter to affordable zones
        if self.budget_constraint:
            filtered_zones = affordable_zones(
                agent.income,
                {z: prices.get(z, 0) for z in zone_options},
            )
        else:
            filtered_zones = list(zone_options)

        # Outside option: if nothing affordable, return home zone at zero utility
        if not filtered_zones:
            home = getattr(agent, "home_zone", "") or (
                zone_options[0] if zone_options else "unknown"
            )
            return ZoneChoice(
                zone_name=home,
                utility=0.0,
                zone_utilities={z: 0.0 for z in zone_options},
                workplace=agent.job_location,
            )

        zone_utilities: dict[str, float] = {}

        for zone_name in zone_options:
            if zone_name not in filtered_zones:
                zone_utilities[zone_name] = 0.0
                continue

            zone = environment.get_zone(zone_name)
            price = prices.get(zone_name, zone.housing_base_price)

            # Feature 1: price / income ratio
            if agent.income > 0:
                price_income_ratio = price / agent.income
            else:
                price_income_ratio = 10.0  # very high penalty

            # Feature 2: commute time (minutes)
            route = environment.transport.get_best_route(zone_name, agent.job_location)
            if route is not None:
                commute_minutes = route.time_minutes
            else:
                commute_minutes = 90.0  # penalty for unreachable

            # Feature 3: facilities density (average quality of all facilities)
            if zone.facilities:
                facilities_density = sum(f.quality for f in zone.facilities) / len(
                    zone.facilities
                )
            else:
                facilities_density = 0.0

            # Feature 4: amenity score
            amenity = zone.amenity_score

            # V_z = beta_1 * (price/income) + beta_2 * commute
            #      + beta_3 * facilities + beta_4 * amenity
            v_z = (
                self._beta_price_income * price_income_ratio
                + self._beta_commute * commute_minutes
                + self._beta_facilities * facilities_density
                + self._beta_amenity * amenity
            )

            zone_utilities[zone_name] = v_z

        # Choose best from affordable zones
        best_zone = max(filtered_zones, key=lambda z: zone_utilities.get(z, -1e9))
        return ZoneChoice(
            zone_name=best_zone,
            utility=zone_utilities[best_zone],
            zone_utilities=zone_utilities,
            workplace=agent.job_location,
        )

    def decide_batch(
        self,
        agents: list[Agent],
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> list[ZoneChoice]:
        """Choose zones for a batch of agents."""
        return [self.decide(a, environment, zone_options, prices) for a in agents]

    def set_cache(self, cache) -> None:
        """No-op: this engine does not use an LLM cache."""
        return None

    @property
    def price_elasticity(self) -> float:
        """Return the estimated price elasticity of housing demand."""
        return self._eta
