"""Open-city variant of AhlfeldtMarket for counterfactual evaluation.

Matches pack's "Run 2" solver path (public-transit-only counterfactual in
Ahlfeldt, Redding, Sturm & Wolf 2015) by rescaling total agent mass each
iteration so the current Fréchet log-sum utility matches a reservation
utility `Ubar_reservation` computed from the baseline (observed) state.

See Matlab reference:
  core/solver/solve_berlin_wall_2015_full_fidelity.m line 450
    HH = (Ubar ./ UU) .* HH

Run 1 (baseline) uses the parent class `AhlfeldtMarket` unchanged. This
subclass touches nothing in the parent except the empty `_post_iter_hook`
no-op, which it overrides here.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from agent_urban_planning.core.market import AhlfeldtMarket

if TYPE_CHECKING:  # pragma: no cover
    from agent_urban_planning.core.agents import Agent


class OpenCityAhlfeldtMarket(AhlfeldtMarket):
    """Open-city Ahlfeldt solver: population responds to utility pressure.

    Usage::

        # 1. Compute reservation utility from observed baseline state.
        Ubar_res = OpenCityAhlfeldtMarket.compute_reservation_utility(
            B_vec, Q_vec, w_vec, tau, params.kappa, params.epsilon,
            params.beta,
        )

        # 2. Instantiate subclass pointing at Ubar_res.
        market = OpenCityAhlfeldtMarket(
            scenario.market, scenario.ahlfeldt_params,
            Ubar_reservation=Ubar_res,
        )

        # 3. Call .clear() as usual. Inside, each iteration adjusts agent
        #    weights by (Ubar_res / current_Ubar).
    """

    def __init__(
        self, *args,
        Ubar_reservation: float,
        mass_damping: float = 0.25,
        ratio_cap: tuple[float, float] = (0.9, 1.1),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if Ubar_reservation <= 0:
            raise ValueError(
                f"Ubar_reservation must be positive; got {Ubar_reservation}"
            )
        self.Ubar_reservation = float(Ubar_reservation)
        # Pack damps all its iterative updates at 0.25 (line 446-449). We
        # use matching damping on the mass update plus a per-iter cap on
        # the ratio — large ratios compound and cause mass to run away.
        self.mass_damping = float(mass_damping)
        self.ratio_cap = ratio_cap
        self._open_city_cache: dict = {}
        self.utility_history: list[float] = []
        self.mass_history: list[float] = []

    # ---------------------------------------------------------------
    # Reservation-utility helper (static so callers can compute it
    # from observed data without instantiating the market).
    # ---------------------------------------------------------------
    @staticmethod
    def compute_reservation_utility(
        B_vec: np.ndarray,
        Q_vec: np.ndarray,
        w_vec: np.ndarray,
        tau: np.ndarray,
        kappa: float,
        epsilon: float,
        beta: float,
    ) -> float:
        """Compute Fréchet log-sum utility from a given state.

        From Ahlfeldt's (2015) closed-form:

            U = Γ(1 - 1/ε) · Φ^(1/ε)

        where Φ = Σ_{i, j} exp(-ε·κ·τ_{ij}) · B_i^ε · Q_i^{-ε(1-β)} · w_j^ε
        is the ex-ante utility denominator over all residence×workplace pairs.
        """
        EPS = 1e-12
        log_B = np.log(np.maximum(B_vec, EPS))
        log_Q = np.log(np.maximum(Q_vec, EPS))
        log_w = np.log(np.maximum(w_vec, EPS))
        # U_ij = log_B[i] + log_w[j] - (1-β)·log_Q[i] - κ·τ_ij
        U = (
            log_B[:, None]
            + log_w[None, :]
            - (1.0 - beta) * log_Q[:, None]
            - kappa * tau
        )
        # log-sum-exp over all (i, j)
        max_logit = float(np.max(epsilon * U))
        log_Phi = max_logit + math.log(float(np.sum(np.exp(epsilon * U - max_logit))))
        # Γ(1 - 1/ε) factor
        gamma_f = math.gamma(1.0 - 1.0 / epsilon)
        Ubar = gamma_f * math.exp(log_Phi / epsilon)
        return float(Ubar)

    # ---------------------------------------------------------------
    # Override hook: rescale agent weights to keep current U ≈ Ubar_res
    # ---------------------------------------------------------------
    def _post_iter_hook(
        self,
        *,
        iteration: int,
        Q: dict,
        wages: dict,
        zone_names: list,
        environment,
        agents_list: list,
        **extra,
    ) -> None:
        # Build / cache static inputs: tau, B, ordering.
        if not self._open_city_cache:
            tau = self._build_tau(environment, zone_names)
            B_vec = np.array(
                [float(environment.get_zone(z).amenity_B) for z in zone_names],
                dtype=np.float64,
            )
            self._open_city_cache = {"tau": tau, "B": B_vec}
        tau = self._open_city_cache["tau"]
        B_vec = self._open_city_cache["B"]

        Q_vec = np.array([Q[z] for z in zone_names], dtype=np.float64)
        w_vec = np.array([wages[z] for z in zone_names], dtype=np.float64)

        # Current Fréchet utility at this iteration's Q, w.
        UU = self.compute_reservation_utility(
            B_vec, Q_vec, w_vec, tau,
            kappa=self.params.kappa,
            epsilon=self.params.epsilon,
            beta=self.params.beta,
        )
        self.utility_history.append(UU)

        if UU <= 0 or not math.isfinite(UU):
            return  # numerical guard — skip rescale

        # Pack's update (solver line 450): HH = (Ubar_current / Ubar_reservation) · HH
        # If current utility < reservation, ratio < 1 → mass shrinks (workers leave).
        # If current utility > reservation, ratio > 1 → mass grows (workers move in).
        raw_ratio = UU / self.Ubar_reservation
        # Tight per-iter cap prevents compounding mass runaway.
        raw_ratio = max(self.ratio_cap[0], min(raw_ratio, self.ratio_cap[1]))
        # Geometric damping: blend log-space toward identity, then exponentiate.
        #   effective_ratio = raw_ratio^mass_damping
        # At mass_damping=0.25 with raw_ratio=1.1, effective ≈ 1.024 (2.4% step).
        effective_ratio = raw_ratio ** self.mass_damping
        for a in agents_list:
            a.weight *= effective_ratio
        total_mass = sum(a.weight for a in agents_list)
        self.mass_history.append(total_mass)

    # ---------------------------------------------------------------
    @staticmethod
    def _build_tau(environment, zone_names: list) -> np.ndarray:
        """Build dense NxN travel-time matrix aligned to ``zone_names``.

        Prefers `environment.transport_matrix` (dense NPZ loaded at scenario
        build time) for block-level scenarios. Falls back to `transport.travel_time`
        only for scenarios without a dense matrix.
        """
        N = len(zone_names)
        if (
            getattr(environment, "transport_matrix", None) is not None
            and getattr(environment, "transport_matrix_index", None)
        ):
            tt_idx = [environment._matrix_index_map[z] for z in zone_names]
            return environment.transport_matrix[np.ix_(tt_idx, tt_idx)].astype(
                np.float64, copy=False
            )
        tau = np.zeros((N, N), dtype=np.float64)
        for i, zi in enumerate(zone_names):
            for j, zj in enumerate(zone_names):
                tau[i, j] = environment.travel_time(zi, zj)
        return tau
