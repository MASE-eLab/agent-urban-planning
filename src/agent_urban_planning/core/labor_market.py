"""Labor-market clearing for Berlin / Ahlfeldt scenarios.

The firm at workplace zone ``j`` has Cobb-Douglas production
``Y_j = A_j * H_M_j^alpha * L_j^(1-alpha)`` where ``H_M`` is employment
and ``L`` is commercial floor space. Zero-profit + marginal-product-
equals-wage gives a closed-form labor demand curve

    H_M_d(w_j) = (alpha * A_j)^(1/(1-alpha)) * L_j * w_j^(-1/(1-alpha))

Ahlfeldt's 2015 calibration fixes ``alpha = 0.80``, giving a wage
elasticity of labor demand of ``-1/(1-alpha) = -5``. Combined with the
Fréchet-driven labor supply elasticity ``eps ≈ 6.69``, the total
excess-demand elasticity is ``|eta_wage| ≈ 11.69``, which the two-market
tatonnement uses for step sizing.

Commercial floor ``L_j`` is held fixed from pack-provided observed 2006
values (Decision D in the OpenSpec design: "fixed split θ_i"). The
zero-profit commercial-rent diagnostic is computed alongside but not
cleared as a market — it is reported so analysts can see where the
fixed-θ assumption shows strain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LaborMarketResult:
    """Snapshot of the labor market after a wage update step."""

    wages: dict[str, float]
    demand: dict[str, float]
    supply: dict[str, float]
    excess_demand_by_zone: dict[str, float]
    max_excess_ratio: float
    commercial_price_diagnostic: dict[str, float] = field(default_factory=dict)
    zero_supply_zones: list[str] = field(default_factory=list)


class LaborMarket:
    """Wage clearing for a population of workplace zones.

    Parameters
    ----------
    alpha: float
        Labor share in production (typically 0.80 from Ahlfeldt 2015).
    A: dict[str, float]
        Post-agglomeration productivity per workplace zone.
    L: dict[str, float]
        Commercial floor area per workplace zone (fixed).
    wage_observed: dict[str, float]
        Observed baseline wages per zone, used as warm-start.
    max_price_change_pct: float, default 0.5
        Per-iteration safety cap on ``|Δw / w|``.
    zero_supply_threshold: float, default 1e-6
        If aggregate labor supply at a zone falls below this, the wage
        update is suppressed for that zone in the current iteration.
    """

    def __init__(
        self,
        alpha: float,
        A: dict[str, float],
        L: dict[str, float],
        wage_observed: dict[str, float],
        *,
        max_price_change_pct: float = 0.5,
        zero_supply_threshold: float = 1e-6,
    ):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        # Zone ordering by workplace index; matched across A, L, wage vectors.
        self.zone_names = list(A.keys())
        self.alpha = float(alpha)
        self.A = np.array([float(A[z]) for z in self.zone_names], dtype=np.float64)
        self.L = np.array([float(L[z]) for z in self.zone_names], dtype=np.float64)
        self.wage_observed = np.array(
            [float(wage_observed.get(z, 1.0)) for z in self.zone_names],
            dtype=np.float64,
        )
        self.max_price_change_pct = float(max_price_change_pct)
        self.zero_supply_threshold = float(zero_supply_threshold)

        # Precompute production exponent
        self._one_minus_alpha = 1.0 - self.alpha
        self._inv_1ma = 1.0 / self._one_minus_alpha  # 1 / (1 - alpha)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_demand(self, wages: np.ndarray) -> np.ndarray:
        """Firm labor demand per zone given the wage vector.

        ``H_M_d(w_j) = (alpha * A_j)^(1/(1-alpha)) * L_j * w_j^(-1/(1-alpha))``
        """
        # Avoid 0^power explosions when wage is 0 (shouldn't happen, but guard).
        safe_w = np.maximum(wages, 1e-12)
        scale = (self.alpha * self.A) ** self._inv_1ma
        return scale * self.L * safe_w ** (-self._inv_1ma)

    def compute_commercial_price_diagnostic(self, wages: np.ndarray) -> np.ndarray:
        """Zero-profit commercial floor price per zone (diagnostic).

        ``q_j = (1 - alpha) * (alpha / w_j)^(alpha/(1-alpha)) * A_j^(1/(1-alpha))``

        Under `endogenous_land_use == False` this is a consistency check
        for the fixed-θ assumption. Under `endogenous_land_use == True`
        it becomes the reference value for the arbitrage gap
        ``|P_j - q_j_diag|`` reported by AhlfeldtMarket.
        """
        safe_w = np.maximum(wages, 1e-12)
        return (
            self._one_minus_alpha
            * (self.alpha / safe_w) ** (self.alpha * self._inv_1ma)
            * self.A ** self._inv_1ma
        )

    def compute_commercial_floor_demand(
        self,
        wages: np.ndarray,
        P: np.ndarray,
        A: np.ndarray,
        H_M: np.ndarray,
    ) -> np.ndarray:
        """Firm commercial floor demand per workplace zone.

        Under `endogenous_land_use == True` the unified floor price P_j
        clears combined demand. The firm's Cobb-Douglas FOC with
        arbitrage ``q_j = P_j`` gives:

            L_j = (1 - alpha)^{1/alpha} · A_j^{1/alpha} · H_M_j · P_j^{-1/alpha}

        This replaces the fixed ``L_j = commercial_floor_area`` input
        from the `endogenous_land_use == False` path. ``wages`` is
        accepted for API symmetry with :meth:`compute_demand` but is
        NOT used (under arbitrage, the labor FOC and the land FOC are
        both satisfied by the same P_j).

        Parameters
        ----------
        wages : np.ndarray
            Current wage vector (unused; kept for symmetry).
        P : np.ndarray
            Unified floor price per zone.
        A : np.ndarray
            Productivity per zone (live, from agglomeration injection).
        H_M : np.ndarray
            Workplace employment (worker supply shares × population).
        """
        safe_P = np.maximum(P, 1e-12)
        A_vec = np.maximum(A, 1e-12)
        return (
            (1.0 - self.alpha) ** self._inv_alpha  # uses 1/α exponent
            * A_vec ** self._inv_alpha
            * H_M
            * safe_P ** (-self._inv_alpha)
        )

    @property
    def _inv_alpha(self) -> float:
        """1/α exponent used in the commercial-floor-demand formula."""
        return 1.0 / self.alpha

    def update_wages(
        self,
        wages: np.ndarray,
        supply: np.ndarray,
        eta_wage: float,
        lambda_wage: float,
    ) -> LaborMarketResult:
        """Apply one tatonnement step to wages.

        Returns a :class:`LaborMarketResult` carrying the new wages,
        demand/supply by zone, excess-demand ratios, and the diagnostic
        commercial price per zone. Zero-supply zones (``supply <
        zero_supply_threshold``) are held at their current wage.
        """
        demand = self.compute_demand(wages)
        zero = supply < self.zero_supply_threshold
        zero_supply_zones = [
            self.zone_names[i] for i, z in enumerate(zero) if bool(z)
        ]

        # Excess demand ratio (signed). For zero-supply zones we record
        # the raw difference but zero out the update.
        safe_supply = np.where(zero, 1.0, supply)
        excess_ratio = (demand - supply) / safe_supply
        # Elasticity-based update
        delta_over_w = (lambda_wage / max(eta_wage, 1e-9)) * excess_ratio
        # Clip per-zone change to safety cap
        cap = self.max_price_change_pct
        delta_over_w = np.clip(delta_over_w, -cap, +cap)
        # Suppress update where supply = 0
        delta_over_w = np.where(zero, 0.0, delta_over_w)

        new_wages = wages * (1.0 + delta_over_w)
        # Keep strictly positive
        new_wages = np.maximum(new_wages, 1e-9)

        max_abs_ratio = float(
            np.max(np.abs(excess_ratio[~zero])) if np.any(~zero) else 0.0
        )

        q_diag = self.compute_commercial_price_diagnostic(new_wages)

        return LaborMarketResult(
            wages={z: float(new_wages[i]) for i, z in enumerate(self.zone_names)},
            demand={z: float(demand[i]) for i, z in enumerate(self.zone_names)},
            supply={z: float(supply[i]) for i, z in enumerate(self.zone_names)},
            excess_demand_by_zone={
                z: float((demand[i] - supply[i])) for i, z in enumerate(self.zone_names)
            },
            max_excess_ratio=max_abs_ratio,
            commercial_price_diagnostic={
                z: float(q_diag[i]) for i, z in enumerate(self.zone_names)
            },
            zero_supply_zones=zero_supply_zones,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_array(self, values: dict[str, float]) -> np.ndarray:
        """Convert a zone→value dict to an array in the market's zone order."""
        return np.array(
            [float(values.get(z, 0.0)) for z in self.zone_names], dtype=np.float64
        )

    def to_dict(self, arr: np.ndarray) -> dict[str, float]:
        """Convert a zone-ordered array back to a zone→value dict."""
        return {z: float(arr[i]) for i, z in enumerate(self.zone_names)}
