"""Utility-based decision engine using literature-calibrated coefficients.

The utility model is a linear-in-parameters discrete choice:

    V_z = β₁ × (price_z / income_i) + β₂ × commute_iz + β₃ × facilities_z + β₄ × amenity_z

Default coefficients are from published housing economics studies:
  - β₁ = -2.0: calibrated from Phang & Wong (1997) Singapore HDB demand elasticity η ≈ -0.5
  - β₂ = -0.015: from Lerman (1977) commute disutility
  - β₃ = 0.5: from Bayer, Ferreira & McMillan (2007) services/price ratio
  - β₄ = 0.8: from Bayer et al. (2007) amenity/price ratio

See ``estimation.py:literature_fallback()`` for full citations and
calibration methodology.
"""

from __future__ import annotations

from agent_urban_planning.core.agents import Agent
from agent_urban_planning.decisions.base import ZoneChoice
from agent_urban_planning.core.environment import Environment, Zone


# Default β coefficients from literature. These are loaded once at module
# level so every UtilityEngine instance uses the same values.
# Full citations and calibration methodology in estimation.py:literature_fallback().
_DEFAULT_BETA_PRICE_INCOME = -2.0       # Phang & Wong (1997)
_DEFAULT_BETA_COMMUTE = -0.015          # Lerman (1977)
_DEFAULT_BETA_FACILITIES = 0.5          # Bayer et al. (2007)
_DEFAULT_BETA_AMENITY = 0.8             # Bayer et al. (2007)
_DEFAULT_PRICE_ELASTICITY = -0.5        # Phang & Wong (1997)


def _compute_facilities_density(zone: Zone) -> float:
    """Average quality of all facilities in a zone.

    Quality scores are per-capita densities (computed at YAML generation
    time from real Census population + real facility counts). Returns 0
    if the zone has no facilities.
    """
    if not zone.facilities:
        return 0.0
    return sum(f.quality for f in zone.facilities) / len(zone.facilities)


class UtilityEngine:
    """Decision engine using literature-calibrated β coefficients.

    Computes zone utility as:

        V_z = β₁ × (price_z / income) + β₂ × commute_z + β₃ × facilities_z + β₄ × amenity_z

    Coefficients default to published housing economics studies (Phang &
    Wong 1997, Lerman 1977, Bayer et al. 2007). Custom coefficients can
    be provided via constructor arguments or by using
    ``EstimatedUtilityEngine`` with a JSON file.

    When ``budget_constraint=True`` (default), zones that the agent
    cannot afford (per MAS MSR / HDB income ceiling) are excluded from
    the choice set before utility computation. If no zone is affordable,
    the agent gets the "outside option" — staying in their home zone at
    zero utility.
    """

    def __init__(
        self,
        budget_constraint: bool = True,
        beta_price_income: float = _DEFAULT_BETA_PRICE_INCOME,
        beta_commute: float = _DEFAULT_BETA_COMMUTE,
        beta_facilities: float = _DEFAULT_BETA_FACILITIES,
        beta_amenity: float = _DEFAULT_BETA_AMENITY,
    ):
        self.budget_constraint = budget_constraint
        self.beta_price_income = beta_price_income
        self.beta_commute = beta_commute
        self.beta_facilities = beta_facilities
        self.beta_amenity = beta_amenity

    @property
    def price_elasticity(self) -> float:
        """Approximate price elasticity of housing demand.

        Used by ``market-clearing-v2`` for the tatonnement step size.
        """
        return _DEFAULT_PRICE_ELASTICITY

    def set_cache(self, cache) -> None:
        return None

    def decide_batch(
        self,
        agents,
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ):
        return [self.decide(a, environment, zone_options, prices) for a in agents]

    def _filter_affordable(
        self,
        agent: Agent,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> list[str]:
        """Filter zone_options to those the agent can afford."""
        if not self.budget_constraint:
            return list(zone_options)
        from agent_urban_planning.core.constraints import affordable_zones
        return affordable_zones(agent.income, {z: prices.get(z, 0) for z in zone_options})

    def decide(
        self,
        agent: Agent,
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> ZoneChoice:
        # Budget constraint: filter to affordable zones
        filtered_zones = self._filter_affordable(agent, zone_options, prices)

        # Outside option: if nothing affordable, return home zone at zero utility
        if not filtered_zones:
            home = getattr(agent, "home_zone", "") or (zone_options[0] if zone_options else "unknown")
            return ZoneChoice(
                zone_name=home,
                utility=0.0,
                zone_utilities={z: 0.0 for z in zone_options},
                workplace=agent.job_location,
            )

        zone_utilities = {}
        for zone_name in zone_options:
            zone = environment.get_zone(zone_name)
            price = prices.get(zone_name, zone.housing_base_price)

            if zone_name not in filtered_zones:
                zone_utilities[zone_name] = 0.0
                continue

            # Price / income ratio
            price_income = price / max(agent.income, 1.0)

            # Commute time to job location
            route = environment.transport.get_best_route(zone_name, agent.job_location)
            commute = route.time_minutes if route is not None else 120.0

            # Facility density (average quality, which is per-capita from builder)
            facilities = _compute_facilities_density(zone)

            # Amenity score (per-capita total facility density from builder)
            amenity = zone.amenity_score

            # Linear utility: V = β₁(price/income) + β₂(commute) + β₃(facilities) + β₄(amenity)
            v = (
                self.beta_price_income * price_income
                + self.beta_commute * commute
                + self.beta_facilities * facilities
                + self.beta_amenity * amenity
            )
            zone_utilities[zone_name] = v

        # Choose best from AFFORDABLE zones only
        best_zone = max(filtered_zones, key=lambda z: zone_utilities.get(z, -1e9))
        return ZoneChoice(
            zone_name=best_zone,
            utility=zone_utilities[best_zone],
            zone_utilities=zone_utilities,
            workplace=agent.job_location,
        )
