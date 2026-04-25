"""Housing affordability constraints and agglomeration wage premium.

Housing constraints from Singapore's regulatory framework:

MAS (Monetary Authority of Singapore) enforces the Mortgage Servicing
Ratio (MSR) and Total Debt Servicing Ratio (TDSR). HDB (Housing and
Development Board) enforces income ceilings for BTO flat purchases.

These are REAL institutional constraints, not model assumptions:
  - MSR source: MAS Notice 645
  - HDB income ceilings: HDB InfoWEB, updated periodically
  - All values as of 2024

The constraints are used by the decision engines to filter each agent's
choice set BEFORE utility computation — agents cannot "choose" zones
they literally cannot afford or are ineligible for.
"""

from __future__ import annotations

from typing import Optional

# MAS Mortgage Servicing Ratio: monthly mortgage repayment must not
# exceed this fraction of gross monthly income for HDB loans.
MSR = 0.30  # 30% — from MAS Notice 645

# HDB BTO income ceiling: households above this monthly income cannot
# purchase new BTO flats (they can buy resale HDB or private).
HDB_INCOME_CEILING = 14000  # S$14,000/month for 4-room and 5-room BTO

# Private housing: TDSR limit (55% of income). We use MSR for HDB
# and TDSR for private segment.
TDSR = 0.55  # 55% — from MAS Notice 645


def max_housing_spend(income: float, segment: str = "hdb") -> float:
    """Maximum monthly housing expenditure for an agent.

    Args:
        income: Agent's gross monthly household income.
        segment: "hdb" uses MSR (30%), "private" uses TDSR (55%).
    """
    if segment == "private":
        return income * TDSR
    return income * MSR


def can_afford_zone(income: float, zone_price: float, segment: str = "hdb") -> bool:
    """Can this agent afford housing in this zone?"""
    return max_housing_spend(income, segment) >= zone_price


def is_hdb_eligible(income: float) -> bool:
    """Can this agent buy BTO HDB flats (income at or below ceiling)?"""
    return income <= HDB_INCOME_CEILING


# ------------------------------------------------------------------
# Agglomeration wage premium
# ------------------------------------------------------------------

# Default agglomeration elasticity from urban economics literature:
# Rosenthal & Strange (2004): 3-8% per doubling of employment density
# Combes et al. (2008): 2-5% for European cities
# Cho et al. (2015): ~3% for Singapore CBD
DEFAULT_AGGLOMERATION_PHI = 0.03

import math


def compute_effective_income(
    base_income: float,
    job_zone_employment: int,
    median_employment: float,
    phi: float = None,
) -> float:
    """Compute income adjusted by agglomeration wage premium.

    ``effective_income = base_income × (1 + φ × ln(E_job / E_median))``

    where ``E_job`` is the employment count in the agent's workplace zone
    and ``E_median`` is the median employment across all zones.

    Args:
        base_income: Agent's base monthly household income.
        job_zone_employment: Total employed persons in the agent's job zone.
        median_employment: Median employment count across all zones.
        phi: Agglomeration elasticity (default 0.03 from literature).

    Returns:
        Effective income. Always >= base_income * 0.5 (floored to avoid
        negative income for very low-density zones).
    """
    if phi is None:
        phi = DEFAULT_AGGLOMERATION_PHI
    if phi == 0 or median_employment <= 0 or job_zone_employment <= 0:
        return base_income

    ratio = job_zone_employment / median_employment
    premium = 1.0 + phi * math.log(ratio)
    # Floor at 50% of base to prevent negative income for extreme low-density
    premium = max(premium, 0.5)
    return base_income * premium


# ------------------------------------------------------------------
# Affordable zones
# ------------------------------------------------------------------


def affordable_zones(
    income: float,
    zone_prices: dict[str, float],
    private_prices: Optional[dict[str, float]] = None,
) -> list[str]:
    """Return the list of zone names the agent can afford.

    For agents below HDB_INCOME_CEILING:
      - Check HDB prices using MSR
    For agents above HDB_INCOME_CEILING:
      - Check private prices using TDSR (if available)
      - Fall back to HDB prices if no private data

    Args:
        income: Agent's gross monthly household income.
        zone_prices: {zone_name: hdb_monthly_price} for all zones.
        private_prices: Optional {zone_name: private_monthly_price}.

    Returns:
        List of zone names the agent can afford. May be empty.
    """
    result = []

    if is_hdb_eligible(income):
        # HDB-eligible: check HDB prices at MSR
        budget = max_housing_spend(income, "hdb")
        for zone, price in zone_prices.items():
            if budget >= price:
                result.append(zone)
    else:
        # Above HDB ceiling: must use private segment
        if private_prices:
            budget = max_housing_spend(income, "private")
            for zone, price in private_prices.items():
                if budget >= price:
                    result.append(zone)
        else:
            # No private data available — fall back to HDB prices with TDSR
            budget = max_housing_spend(income, "private")
            for zone, price in zone_prices.items():
                if budget >= price:
                    result.append(zone)

    return result
