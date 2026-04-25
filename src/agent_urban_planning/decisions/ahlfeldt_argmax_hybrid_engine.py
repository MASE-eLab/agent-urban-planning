"""V4 argmax-hybrid engine: per-type heterogeneous preferences, no shocks.

Combines:
  * Run 1-H's per-type β, κ scaling (AhlfeldtHybridEngine math)
  * AhlfeldtABMEngine's aggregate-from-counts pattern
  * Pure argmax: NO Fréchet / Normal / any shock is added to V_ij

For each demographic type k:
    V_ij(k) = log B_i + log w_j − (1 − β_k) log Q_i − κ_k · τ_ij
    (i*_k, j*_k) = argmax V_ij(k)

The aggregate P_agg is built from empirical counts: one type → one cell.
Passed to AhlfeldtMarket.clear() unchanged.

Tier-1 numerical safety: log(max(x, 1e-12)) inside V_ij.
Tier-2 diagnostics per decide_batch: zero-marginal counts, cell coverage,
LLM clip rates, renorm-means.

Cache namespace: .cache/llm_preferences_berlin_v4/
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from agent_urban_planning.core.agents import Agent
from agent_urban_planning.data.loaders import AhlfeldtParams
from agent_urban_planning.decisions.ahlfeldt_abm_engine import AhlfeldtABMEngine
from agent_urban_planning.decisions.base import LocationChoice


# Pack-calibrated defaults (same as Run 1-H).
BETA_BAR_DEFAULT = 0.75
KAPPA_EPS_DEFAULT = 0.0987
EPSILON_DEFAULT = 6.6941


class AhlfeldtArgmaxHybridEngine(AhlfeldtABMEngine):
    """Pure argmax with LLM-elicited per-type ``beta_k`` / ``kappa_k`` preferences.

    The V4-B engine. Combines per-type heterogeneous preferences from
    Run 1-H's elicitation pipeline with the aggregate-from-counts
    pattern of :class:`AhlfeldtABMEngine`, but with no Fréchet / Normal
    / shock added to ``V_ij``. For each demographic type ``k`` the
    indirect utility is

    ::

        V_ij(k) = log B_i + log w_j - (1 - beta_k) log Q_i - kappa_k * tau_ij

    and the choice ``(i*_k, j*_k)`` is the deterministic argmax. The
    aggregate empirical choice matrix passed to
    :class:`AhlfeldtMarket.clear` is the count of types in each cell.
    Most users should configure this via :class:`HybridDecisionEngine`.

    Args:
        params: Structural Ahlfeldt parameters.
        elicitor: Object exposing ``elicit_batch(agents, cache_dir,
            verbose)`` that returns per-agent preference weights
            (e.g. ``simulator.decisions.elicitation.LLMPreferenceElicitor``).
        preference_cache_dir: Directory for caching elicited
            preferences across runs.
        clip_warn_threshold: Fraction of clipped scaling factors above
            which a warning is logged.
        seed: Optional integer seed.
        **kwargs: Forwarded to :class:`AhlfeldtABMEngine`.

    Examples:
        >>> import agent_urban_planning as aup
        >>> # Prefer the public wrapper:
        >>> # engine = aup.HybridDecisionEngine(params, elicitor=elicitor)
    """

    def __init__(
        self,
        params: AhlfeldtParams,
        elicitor=None,
        *,
        preference_cache_dir: str | Path = ".cache/llm_preferences_berlin_v4",
        clip_warn_threshold: float = 0.05,
        seed: Optional[int] = None,
        **kwargs,
    ):
        # Force shock_distribution to something the parent accepts; we'll
        # bypass the shock draw entirely in our overridden decide_batch.
        kwargs.setdefault("shock_distribution", "frechet")  # parent requires supported
        kwargs.setdefault("num_agents", 1)  # placeholder — not used in V4
        super().__init__(params, seed=seed, **kwargs)
        self.elicitor = elicitor
        self.preference_cache_dir = Path(preference_cache_dir)
        self.clip_warn_threshold = float(clip_warn_threshold)
        self._type_params: dict[int, dict] = {}
        self._elicited_agent_ids: set[int] = set()
        self.last_abm_diagnostics: dict = {}

    # ------------------------------------------------------------------
    # LLM elicitation — per-dimension renormalized β/κ scaling (same math
    # as AhlfeldtHybridEngine).
    # ------------------------------------------------------------------
    def ensure_elicitation(self, agents: list[Agent], verbose: bool = True) -> None:
        """Elicit LLM preferences and compute per-type ``beta_k``, ``kappa_k``.

        Idempotent: re-calling with the same agent set is a no-op.
        On first call the configured elicitor is invoked once per
        agent type, results are cached on disk under
        ``self.preference_cache_dir``, and weight-weighted
        renormalization is applied across the four preference axes
        (housing, commute, services, amenities). Per-type ``beta_k``
        and ``kappa_k`` are stored in ``self._type_params``.

        Args:
            agents: List of :class:`Agent` instances.
            verbose: When ``True`` (default), enables the elicitor's
                progress bar and incremental cache writes.

        Returns:
            None. Mutates ``self._type_params`` and
            ``self._elicited_agent_ids``.

        Raises:
            ValueError: If ``self.elicitor`` is ``None``.
            RuntimeError: If a per-axis renormalization mean is
                non-positive (indicates degenerate elicitation).

        Examples:
            >>> import agent_urban_planning as aup
            >>> # engine = aup.HybridDecisionEngine(params, elicitor=el)
            >>> # engine.ensure_elicitation(list(population))
        """
        ids = {int(a.agent_id) for a in agents}
        if self._elicited_agent_ids == ids and self._type_params:
            return
        if self.elicitor is None:
            raise ValueError(
                "AhlfeldtArgmaxHybridEngine requires an LLMPreferenceElicitor; "
                "pass elicitor=... at construction."
            )

        raw = self.elicitor.elicit_batch(
            agents, cache_dir=self.preference_cache_dir, verbose=verbose,
        )
        assert len(raw) == len(agents)
        for agent, pw in zip(agents, raw):
            agent.preferences = pw

        # Per-dimension weight-weighted renormalization.
        weights = np.array([float(a.weight) for a in agents], dtype=np.float64)
        total_w = float(weights.sum())
        if total_w <= 0:
            raise ValueError("Agent weights must sum to a positive value.")

        def _dim(attr: str) -> np.ndarray:
            return np.array([getattr(pw, attr) for pw in raw], dtype=np.float64)

        dim_raw = {
            "Q":   _dim("alpha"),   # housing
            "tau": _dim("beta"),    # commute
            "B":   _dim("gamma"),   # services (metadata-only)
            "w":   _dim("delta"),   # amenities (metadata-only)
        }
        dim_scaled: dict[str, np.ndarray] = {}
        clip_rates: dict[str, float] = {}
        renorm_mean_pre: dict[str, float] = {}
        for name, raw_arr in dim_raw.items():
            s_raw = raw_arr / 0.25
            clipped_lo = s_raw < 0.2
            clipped_hi = s_raw > 4.0
            clip_rates[name] = float(np.mean(clipped_lo | clipped_hi))
            s = np.clip(s_raw, 0.2, 4.0)
            s_bar = float((weights * s).sum() / total_w)
            renorm_mean_pre[name] = s_bar
            if s_bar <= 0:
                raise RuntimeError(f"Pre-renorm mean for dim {name} non-positive.")
            dim_scaled[name] = s / s_bar

        beta_bar = BETA_BAR_DEFAULT
        eps = float(self.params.epsilon) if self.params.epsilon else EPSILON_DEFAULT
        kappa_bar = float(self.params.kappa) if eps > 0 else (KAPPA_EPS_DEFAULT / EPSILON_DEFAULT)

        for i, agent in enumerate(agents):
            self._type_params[int(agent.agent_id)] = {
                "beta":       float(dim_scaled["Q"][i]   * beta_bar),
                "kappa":      float(dim_scaled["tau"][i] * kappa_bar),
                "alpha_Q":    float(dim_raw["Q"][i]),
                "alpha_tau":  float(dim_raw["tau"][i]),
                "alpha_B":    float(dim_raw["B"][i]),
                "alpha_w":    float(dim_raw["w"][i]),
                "scaling_Q":  float(dim_scaled["Q"][i]),
                "scaling_tau": float(dim_scaled["tau"][i]),
            }

        self._elicited_agent_ids = ids
        self._last_clip_rates = clip_rates
        self._last_renorm_mean_pre = renorm_mean_pre

        # Warn on high clip rates.
        logger = logging.getLogger(__name__)
        for k, rate in clip_rates.items():
            if rate > self.clip_warn_threshold:
                logger.warning(
                    "AhlfeldtArgmaxHybridEngine: dim %s clips %.1f%% of types; "
                    "pre-renorm mean %.3f.",
                    k, 100.0 * rate, renorm_mean_pre[k],
                )

    # ------------------------------------------------------------------
    # Per-type argmax (no shock). Overrides ABM engine's sampling.
    # ------------------------------------------------------------------
    def decide_batch(
        self,
        agents: list[Agent],
        environment,
        zone_options: list[str],
        prices: dict,
    ) -> list[LocationChoice]:
        """Pure argmax over per-type ``V_ij(k)`` with LLM-elicited preferences.

        Builds per-type ``V_ij`` matrices from log B, log w, log Q, and
        ``tau``, takes the deterministic argmax for each agent type,
        and accumulates the empirical choice matrix in
        ``self.last_choice_probabilities`` for consumption by
        :class:`AhlfeldtMarket`. Triggers
        :meth:`ensure_elicitation` on first call.

        Args:
            agents: List of :class:`Agent` instances to decide for.
            environment: The :class:`Environment` carrying zones.
            zone_options: Allowed zone names.
            prices: Mapping ``zone -> Q_i``.

        Returns:
            List of :class:`LocationChoice`, one per input agent and
            in the same order.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # engine = aup.HybridDecisionEngine(params, elicitor=el)
            >>> # choices = engine.decide_batch(agents, env, zones, prices)
        """
        if not agents:
            return []

        self.ensure_elicitation(agents)

        N = len(zone_options)
        zones = list(zone_options)
        EPS = 1e-12

        # Assemble zone vectors.
        amen_source = self._current_amenity
        def _resolve_B(z):
            if amen_source and z in amen_source:
                v = amen_source[z]
                if v is not None:
                    return float(v)
            v = environment.get_zone(z).amenity_B
            return float(v) if v else EPS
        B = np.array([_resolve_B(z) for z in zones], dtype=np.float64)

        wages_source = self._current_wages
        w_vec = np.array(
            [
                float(
                    (wages_source.get(z) if wages_source else None)
                    or environment.get_zone(z).wage_observed
                    or 1.0
                )
                for z in zones
            ],
            dtype=np.float64,
        )
        Q = np.array(
            [float(prices.get(z, environment.get_zone(z).floor_price_observed))
             for z in zones],
            dtype=np.float64,
        )

        if environment.transport_matrix is not None and environment.transport_matrix_index:
            tt_idx = [environment._matrix_index_map[z] for z in zones]
            tau = environment.transport_matrix[np.ix_(tt_idx, tt_idx)].astype(
                np.float64, copy=False,
            )
        else:
            tau = np.zeros((N, N), dtype=np.float64)
            for i, zi in enumerate(zones):
                for j, zj in enumerate(zones):
                    tau[i, j] = environment.travel_time(zi, zj)

        # Tier-1: log floor.
        log_B = np.log(np.maximum(B, EPS))
        log_w = np.log(np.maximum(w_vec, EPS))
        log_Q = np.log(np.maximum(Q, EPS))
        U_common = log_B[:, None] + log_w[None, :]  # (N, N)

        # Aggregates.
        total_weight = sum(float(a.weight) for a in agents)
        if total_weight <= 0:
            raise ValueError("agent weights sum to zero")
        P_agg = np.zeros((N, N), dtype=np.float64)
        HR_counts = np.zeros(N, dtype=np.float64)
        HM_counts = np.zeros(N, dtype=np.float64)
        results: list[LocationChoice] = []

        for agent in agents:
            tp = self._type_params[int(agent.agent_id)]
            beta_k = tp["beta"]
            kappa_k = tp["kappa"]
            V_k = (
                U_common
                - (1.0 - beta_k) * log_Q[:, None]
                - kappa_k * tau
            )
            # Argmax over flat (i, j).
            flat_idx = int(np.argmax(V_k))
            i_star = flat_idx // N
            j_star = flat_idx - i_star * N
            share = float(agent.weight) / total_weight
            P_agg[i_star, j_star] += share
            HR_counts[i_star] += share
            HM_counts[j_star] += share
            results.append(LocationChoice(
                residence=zones[i_star],
                workplace=zones[j_star],
                utility=float(V_k[i_star, j_star]),
                zone_utilities={},
            ))

        # Expose empirical P_agg to the market.
        self.last_choice_probabilities = P_agg.astype(self._np_dtype, copy=False)

        # Tier-2 diagnostics.
        n_cells_nonzero = int(np.sum(P_agg > 0))
        n_res_zero = int(np.sum(HR_counts == 0))
        n_wp_zero = int(np.sum(HM_counts == 0))
        self.last_abm_diagnostics = {
            "n_types_total":               len(agents),
            "n_cells_nonzero":             n_cells_nonzero,
            "total_cells":                 N * N,
            "n_residence_marginal_zero":   n_res_zero,
            "n_workplace_marginal_zero":   n_wp_zero,
            "clip_rates":                  dict(self._last_clip_rates),
            "renorm_means_pre":            dict(self._last_renorm_mean_pre),
            "HR_min":                      float(HR_counts.min()),
            "HR_max":                      float(HR_counts.max()),
            "HM_min":                      float(HM_counts.min()),
            "HM_max":                      float(HM_counts.max()),
            "shock_distribution":          "none",
            "num_agents":                  len(agents),
        }

        logger = logging.getLogger(__name__)
        if n_res_zero > 0:
            empty_res = [zones[i] for i in range(N) if HR_counts[i] == 0][:5]
            logger.warning(
                "V4: %d Ortsteile have zero residence marginal; first few: %s",
                n_res_zero, empty_res,
            )
        if n_wp_zero > 0:
            empty_wp = [zones[j] for j in range(N) if HM_counts[j] == 0][:5]
            logger.warning(
                "V4: %d Ortsteile have zero workplace marginal; first few: %s",
                n_wp_zero, empty_wp,
            )

        # Per-zone utility summaries (last agent's V_k for reporting compat).
        self.last_diagnostics = {
            "expected_utility": float(results[-1].utility) if results else 0.0,
            "n_agents": len(results),
            "n_zones": N,
            "sampling_path": "v4-argmax-hybrid-no-shock",
            "B_mean": float(np.mean(B)),
            "B_std":  float(np.std(B)),
        }

        return results

    def type_parameters(self) -> dict[int, dict]:
        """Return a snapshot of the per-type scaling parameters.

        Each type's record contains the scaled ``beta`` and
        ``kappa``, the raw LLM scores per axis (``alpha_Q``,
        ``alpha_tau``, ``alpha_B``, ``alpha_w``), and the per-axis
        renormalized scaling factors (``scaling_Q``, ``scaling_tau``).
        Used for diagnostic output and for paper tables.

        Returns:
            Dict mapping ``agent_id -> per-type parameter dict``.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # engine.ensure_elicitation(agents)
            >>> # engine.type_parameters()[0]["beta"]
        """
        return dict(self._type_params)
