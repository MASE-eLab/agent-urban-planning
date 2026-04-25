"""Berlin Run 1-H: Ahlfeldt engine with LLM-derived per-type β and κ.

Heterogeneous extension of :class:`AhlfeldtUtilityEngine`. The pack's
indirect utility ``ln u_ij = ln B_i + ln w_j − (1−β) ln Q_i − κ τ_ij``
has only two agent-level levers that keep the FOC closure valid:

  * ``β``  — residential floor-expenditure share (scales the −(1−β)·ln Q_i
             sensitivity)
  * ``κ``  — per-minute commute disutility

For each demographic type ``k`` we compute a scaled ``β_k`` and ``κ_k``
from the LLM-elicited preference weights of that type::

    s_k  = clamp(α_{Q,k} / 0.25, 0.2, 4.0)
    ŝ_k  = s_k / s̄         # weight-weighted mean of s across types = 1
    β_k  = ŝ_{Q,k} × β̄      # β̄ = pack residential floor share (default 0.75)
    κ_k  = ŝ_{τ,k} × κ̄      # κ̄ = params.kappa = κ_ε / ε (per-minute decay)

Note on κ̄ scale: The pack's paper documents the commute disutility as
``κ_ε × τ`` applied inside the logit (where ``κ_ε ≈ 0.0987``). Our engine
stores ``κ = κ_ε / ε`` and applies ``ε·U`` as the logit, so the effective
coefficient on τ is recovered as ``ε·κ = κ_ε``. Consequently the
per-agent scaling target is ``κ̄ = κ_ε / ε ≈ 0.01474``, not ``ε·κ_ε``.

The other two LLM dimensions (α_B amenity, α_w wage) are stored on each
type as reporting-only metadata. They do NOT enter the structural
equations — Table B's distributional report reads them for context
but the FOC closure never consumes them.

Key invariants (verified by tests):

  * When every type returns identical LLM weights (0.25, 0.25, 0.25, 0.25),
    β_k = β̄ and κ_k = κ̄ for all k → reduces to Run 1 exactly.
  * The weight-weighted mean of ŝ_{Q,k} across types equals 1.0 to
    within 1e-9 (per-dimension renormalization).
  * The aggregate choice probability matrix ``P_agg = Σ_k ω_k · P_k`` is
    exposed as ``self.last_choice_probabilities`` so :class:`AhlfeldtMarket`
    can compute demand aggregates unchanged from Run 1.

Call sequence::

    eng = AhlfeldtHybridEngine(params, elicitor, cache_dir=..., ...)
    eng.ensure_elicitation(agents)   # idempotent; called automatically
                                     # on first decide_batch if skipped
    choices = eng.decide_batch(agents, env, zone_names, Q)
    P_agg = eng.last_choice_probabilities
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from agent_urban_planning.core.agents import Agent, PreferenceWeights
from agent_urban_planning.data.loaders import AhlfeldtParams
from agent_urban_planning.decisions.ahlfeldt_utility import AhlfeldtUtilityEngine
from agent_urban_planning.decisions.base import LocationChoice

# Pack-calibrated structural means (Ahlfeldt et al. 2015).
BETA_BAR_DEFAULT = 0.75        # residential floor-expenditure share
KAPPA_EPS_DEFAULT = 0.0987     # per-minute commute decay in utility
EPSILON_DEFAULT = 6.6941       # Fréchet shape parameter


class AhlfeldtHybridEngine(AhlfeldtUtilityEngine):
    """Per-type β and κ derived from LLM-elicited preference weights."""

    # LLM preferences map to four dimension names on PreferenceWeights:
    #   alpha → housing (α_Q)         → β scaling     [structural]
    #   beta  → commute (α_τ)         → κ scaling     [structural]
    #   gamma → services (α_B)        → reporting only
    #   delta → amenities (α_w)       → reporting only

    def __init__(
        self,
        params: AhlfeldtParams,
        elicitor=None,
        *,
        preference_cache_dir: str | Path | None = None,
        clip_warn_threshold: float = 0.05,
        elicitation_concurrency: int = 0,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self.elicitor = elicitor
        self.preference_cache_dir = (
            Path(preference_cache_dir) if preference_cache_dir else None
        )
        self.clip_warn_threshold = float(clip_warn_threshold)
        # Elicitation-time concurrency cap. 0 = auto (client.total_concurrency
        # for multi-provider, or 10 for single-provider). When >0, overrides
        # the auto-resolution inside LLMPreferenceElicitor.elicit_batch.
        self.elicitation_concurrency = int(elicitation_concurrency)

        # Per-type cached scaling factors populated by ensure_elicitation().
        # Index by agent_id → (beta_k, kappa_k, raw_preferences, scalings ŝ).
        self._type_params: dict[int, dict] = {}
        self._elicited = False
        # Metadata captured during the latest ensure_elicitation call.
        self.last_elicitation_diag: dict = {}
        # Per-type residence/workplace marginal shares from the most recent
        # decide_batch call. Shapes: {agent_id: ndarray(N,)}. Consumed by
        # the Table B distributional writer.
        self.last_per_type_residence_shares: dict[int, np.ndarray] = {}
        self.last_per_type_workplace_shares: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Preference elicitation
    # ------------------------------------------------------------------
    def ensure_elicitation(self, agents: list[Agent]) -> None:
        """Elicit LLM preferences and compute per-type β_k, κ_k.

        Safe to call multiple times — subsequent calls short-circuit as
        long as the same agent set is passed. Assumes all agents in
        ``agents`` are the canonical demographic types (weight = type share).
        """
        if self._elicited and self._type_params and set(self._type_params.keys()) == {
            a.agent_id for a in agents
        }:
            return

        if self.elicitor is None:
            raise ValueError(
                "AhlfeldtHybridEngine requires an LLMPreferenceElicitor. "
                "Pass elicitor=... to the constructor."
            )

        # Phase 1 — elicit raw 4-tuple weights per type.
        raw_weights = self.elicitor.elicit_batch(
            agents,
            cache_dir=self.preference_cache_dir,
            concurrency=self.elicitation_concurrency,
        )
        assert len(raw_weights) == len(agents)
        for agent, pw in zip(agents, raw_weights):
            agent.preferences = pw

        # Phase 2 — clamp + per-dimension weight-weighted renormalization.
        weights = np.array([float(a.weight) for a in agents], dtype=np.float64)
        total_w = float(weights.sum())
        if total_w <= 0:
            raise ValueError("Agent weights must sum to a positive value.")

        # alpha→Q, beta→τ, gamma→B, delta→w. Store all 4 so we can report
        # α_B and α_w in Table B.
        def _dim(attr: str) -> np.ndarray:
            return np.array([getattr(pw, attr) for pw in raw_weights], dtype=np.float64)

        dim_raw = {
            "Q": _dim("alpha"),
            "tau": _dim("beta"),
            "B": _dim("gamma"),
            "w": _dim("delta"),
        }
        dim_scaled: dict[str, np.ndarray] = {}
        clip_rates: dict[str, float] = {}
        renorm_mean_pre: dict[str, float] = {}
        for name, raw in dim_raw.items():
            s_raw = raw / 0.25  # divisor = uniform neutral weight
            # Clamp to [0.2, 4.0]; record which entries clip for diagnostics.
            clipped_lo = s_raw < 0.2
            clipped_hi = s_raw > 4.0
            clip_count = int(np.sum(clipped_lo | clipped_hi))
            clip_rates[name] = clip_count / len(s_raw)
            s = np.clip(s_raw, 0.2, 4.0)
            # Weight-weighted mean.
            s_bar = float((weights * s).sum() / total_w)
            renorm_mean_pre[name] = s_bar
            if s_bar <= 0:
                raise RuntimeError(f"Pre-renorm mean for dimension {name} is non-positive: {s_bar}")
            dim_scaled[name] = s / s_bar  # ŝ_k,i — weight-weighted mean = 1 by construction

        # Phase 3 — compute β_k, κ_k per type.
        # κ̄ is the per-minute decay the engine internally applies — i.e.
        # ``params.kappa = κ_ε / ε``. See module docstring for the logit-
        # space derivation.
        beta_bar = BETA_BAR_DEFAULT
        eps = float(self.params.epsilon) if self.params.epsilon else EPSILON_DEFAULT
        kappa_bar = float(self.params.kappa) if eps > 0 else (KAPPA_EPS_DEFAULT / EPSILON_DEFAULT)
        for i, agent in enumerate(agents):
            self._type_params[int(agent.agent_id)] = {
                "beta": float(dim_scaled["Q"][i] * beta_bar),
                "kappa": float(dim_scaled["tau"][i] * kappa_bar),
                "alpha_B": float(dim_raw["B"][i]),
                "alpha_w": float(dim_raw["w"][i]),
                "alpha_Q": float(dim_raw["Q"][i]),
                "alpha_tau": float(dim_raw["tau"][i]),
                "scaling_Q": float(dim_scaled["Q"][i]),
                "scaling_tau": float(dim_scaled["tau"][i]),
            }

        # Phase 4 — clip-rate warning.
        over_threshold = {
            k: v for k, v in clip_rates.items() if v > self.clip_warn_threshold
        }
        if over_threshold:
            logger = logging.getLogger(__name__)
            for k, rate in over_threshold.items():
                logger.warning(
                    "AhlfeldtHybridEngine: dimension %s clips %.1f%% of types at "
                    "[0.2, 4.0]. Pre-renorm mean was %.3f. Check LLM output distribution.",
                    k, 100.0 * rate, renorm_mean_pre[k],
                )

        self._elicited = True
        self.last_elicitation_diag = {
            "num_types": len(agents),
            "clip_rates": clip_rates,
            "renorm_mean_pre": renorm_mean_pre,
            "beta_bar": beta_bar,
            "kappa_bar": kappa_bar,
        }

    # ------------------------------------------------------------------
    # decide_batch override — per-type choice probabilities
    # ------------------------------------------------------------------
    def decide_batch(
        self,
        agents: list[Agent],
        environment,
        zone_options: list[str],
        prices: dict,
    ) -> list[LocationChoice]:
        """Compute per-type P_k, aggregate P_agg, sample one agent per type.

        Assumes ``len(agents) == num_types`` — one representative agent
        per demographic type, with ``agent.weight`` equal to the type's
        population share (weights summing to 1.0). The caller scales total
        mass separately (same contract as Run 1).
        """
        if not agents:
            return []

        self.ensure_elicitation(agents)

        N = len(zone_options)
        zones = list(zone_options)

        # Assemble shared zone vectors (same as base class).
        amen_source = self._current_amenity
        def _resolve_B(z):
            if amen_source and z in amen_source:
                v = amen_source[z]
                if v is not None:
                    return float(v)
            v = environment.get_zone(z).amenity_B
            return float(v) if v else 1e-12
        B = np.array([_resolve_B(z) for z in zones], dtype=self._np_dtype)

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
            dtype=self._np_dtype,
        )
        Q = np.array(
            [float(prices.get(z, environment.get_zone(z).floor_price_observed))
             for z in zones],
            dtype=self._np_dtype,
        )

        # Travel-time matrix.
        if environment.transport_matrix is not None and environment.transport_matrix_index:
            tt_idx = [environment._matrix_index_map[z] for z in zones]
            tau = environment.transport_matrix[np.ix_(tt_idx, tt_idx)].astype(
                self._np_dtype, copy=False
            )
        else:
            tau = np.zeros((N, N), dtype=self._np_dtype)
            for i, zi in enumerate(zones):
                for j, zj in enumerate(zones):
                    tau[i, j] = environment.travel_time(zi, zj)

        EPS = 1e-12
        log_B = np.log(np.maximum(B, EPS))
        log_w = np.log(np.maximum(w_vec, EPS))
        log_Q = np.log(np.maximum(Q, EPS))
        # Common, agent-invariant parts of U_ij. Per-type contributions
        # (β_k · log_Q[i] and κ_k · tau[i,j]) are added per-agent below.
        U_common = log_B[:, None] + log_w[None, :]

        # Affordability mask (structural parameter β here is β̄, used only
        # for the budget gate, not the utility). Keeps run 1-H's budget
        # semantics identical to Run 1.
        if self.budget_constraint:
            afford_mask = ((1.0 - self.beta) * w_vec[None, :]) >= (Q[:, None] * 1.0)
            if not np.any(afford_mask):
                afford_mask = np.ones_like(afford_mask, dtype=bool)
        else:
            afford_mask = np.ones((N, N), dtype=bool)

        # Aggregate choice-probability matrix and diagnostic vectors.
        # P_agg must be a PROBABILITY distribution summing to 1.0 (same
        # convention as the base class's last_choice_probabilities). The
        # market multiplies it by total agent mass downstream. We therefore
        # aggregate by POPULATION SHARE (agent.weight / total_weight), not
        # by raw weight — agents may be scaled to an arbitrary population
        # mass by the driver script.
        total_weight = sum(float(a.weight) for a in agents)
        if total_weight <= 0:
            raise ValueError("Sum of agent weights must be positive")
        P_agg = np.zeros((N, N), dtype=self._np_dtype)
        results: list[LocationChoice] = []
        utilities = np.zeros(len(agents), dtype=np.float64)
        zone_utils_template: dict[str, float] = {}
        self.last_per_type_residence_shares = {}
        self.last_per_type_workplace_shares = {}

        # Per-type seed for deterministic sampling (matches Run 1 semantics).
        det_seed_base = abs(hash((self.seed, "hybrid-deterministic-sampler"))) % (2**32 - 1)

        # Iterate one type at a time. At N=12k each P_k is ~600MB; processing
        # sequentially keeps peak memory bounded regardless of num_types.
        for k, agent in enumerate(agents):
            tp = self._type_params[int(agent.agent_id)]
            beta_k = tp["beta"]
            kappa_k = tp["kappa"]

            U_k = (
                U_common
                - (1.0 - beta_k) * log_Q[:, None]
                - kappa_k * tau
            )
            P_k = self._compute_choice_probs(U_k, afford_mask)

            share_k = float(agent.weight) / total_weight
            P_agg += (share_k * P_k).astype(self._np_dtype, copy=False)
            # Per-type marginal shares: these are the residence / workplace
            # choice distributions of type k (not multiplied by its weight).
            self.last_per_type_residence_shares[int(agent.agent_id)] = np.asarray(
                P_k.sum(axis=1), dtype=np.float64,
            )
            self.last_per_type_workplace_shares[int(agent.agent_id)] = np.asarray(
                P_k.sum(axis=0), dtype=np.float64,
            )

            # Sample one (i, j) from P_k — deterministic, per-type sub-seed.
            sub_seed = (det_seed_base + int(agent.agent_id)) % (2**32 - 1)
            rng = np.random.default_rng(sub_seed)
            P_flat = np.asarray(P_k.reshape(-1), dtype=np.float64)
            s = P_flat.sum()
            if s > 0:
                P_flat = P_flat / s
            cdf = np.cumsum(P_flat)
            cdf[-1] = 1.0
            u = float(rng.random())
            idx = int(np.searchsorted(cdf, u, side="right"))
            idx = max(0, min(idx, P_flat.size - 1))
            i = idx // N
            j = idx - i * N
            u_ij = float(U_k[i, j])
            utilities[k] = u_ij
            # zone_utilities — per-type row/col means for this agent's own U_k.
            # Small cost at type scale (50-200) and provides useful per-type context.
            zone_utils = {
                f"R:{z}": float(U_k[ii].mean()) for ii, z in enumerate(zones)
            }
            zone_utils.update({
                f"W:{z}": float(U_k[:, jj].mean()) for jj, z in enumerate(zones)
            })
            if not zone_utils_template:
                zone_utils_template = dict(zone_utils)
            results.append(
                LocationChoice(
                    residence=zones[i],
                    workplace=zones[j],
                    utility=u_ij,
                    zone_utilities=zone_utils,
                )
            )

        self.last_choice_probabilities = P_agg

        # Diagnostics — same schema as the base class.
        residences = [r.residence for r in results]
        if residences:
            counts = np.zeros(N, dtype=np.float64)
            zone_to_idx = {z: i for i, z in enumerate(zones)}
            for a, r in zip(agents, residences):
                counts[zone_to_idx[r]] += float(a.weight)
            probs = counts / max(counts.sum(), 1e-12)
            nonzero = probs[probs > 0]
            entropy_nats = float(-np.sum(nonzero * np.log(nonzero)))
        else:
            entropy_nats = 0.0

        self.last_diagnostics = {
            "expected_utility": float(np.sum(utilities * np.array([a.weight for a in agents]))),
            "entropy_residence_nats": entropy_nats,
            "utility_min": float(np.min(utilities)),
            "utility_max": float(np.max(utilities)),
            "utility_mean": float(np.mean(utilities)),
            "n_agents": len(results),
            "n_zones": N,
            "n_types": len(agents),
            "sampling_path": "hybrid-deterministic",
            "B_mean": float(np.mean(B)),
            "B_std": float(np.std(B)),
        }
        return results

    # ------------------------------------------------------------------
    # Accessors for Table B (distributional report)
    # ------------------------------------------------------------------
    def type_parameters(self) -> dict[int, dict]:
        """Return per-type scaling parameters for the distributional report.

        Each entry: ``{"beta", "kappa", "alpha_Q", "alpha_tau",
        "alpha_B", "alpha_w", "scaling_Q", "scaling_tau"}``.
        α_B and α_w are captured as metadata — they do not affect V_ij.
        """
        return dict(self._type_params)
