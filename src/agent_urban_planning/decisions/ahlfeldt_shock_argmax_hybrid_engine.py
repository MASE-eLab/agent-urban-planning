"""V4 shock-argmax-hybrid engine: mixed-logit / random-coefficients logit.

V4 = V2 (AhlfeldtABMEngine, argmax + Gumbel shock) with per-type
(beta_k, kappa_k) scaling factors from LLM-elicited preferences replacing
V2's homogeneous (beta, kappa).

For each demographic type k:
    V_k[i, j] = log B_i + log w_j - (1 - beta_k) log Q_i - kappa_k * tau_ij
Per agent, draw a (N, N) Gumbel shock; argmax over V_k[i, j] + shock / epsilon.
Aggregate empirical counts feed AhlfeldtMarket unchanged.

Setting all (beta_k, kappa_k) == (beta, kappa) collapses V4 to V2 bit-for-bit
(same RNG, same shock sequence). This is the uniform-collapse invariant.

Cache namespace: reuses V4-A's `.cache/llm_preferences_berlin_v4/` by default.
"""
from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from agent_urban_planning.core.agents import Agent
from agent_urban_planning.data.loaders import AhlfeldtParams
from agent_urban_planning.decisions.ahlfeldt_abm_engine import AhlfeldtABMEngine
from agent_urban_planning.decisions.base import LocationChoice


def _demographic_cache_key(agent: Agent) -> str:
    """Same formula as LLMPreferenceElicitor._cache_key; inlined so this
    engine works with any elicitor (including test stubs)."""
    data = (
        f"{agent.age_head}_{agent.household_size}_{agent.has_children}_"
        f"{agent.has_elderly}_{agent.income:.0f}_{agent.job_location}_{agent.car_owner}"
    )
    return hashlib.md5(data.encode()).hexdigest()


BETA_BAR_DEFAULT = 0.75
KAPPA_EPS_DEFAULT = 0.0987
EPSILON_DEFAULT = 6.6941


class AhlfeldtShockArgmaxHybridEngine(AhlfeldtABMEngine):
    """V2's argmax + Gumbel shock, with per-type (beta_k, kappa_k) from LLM."""

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
        # Parent (AhlfeldtABMEngine) handles shock_distribution, num_agents,
        # batch_size, seed. All defaults stay compatible with V2.
        super().__init__(params, seed=seed, **kwargs)
        self.elicitor = elicitor
        self.preference_cache_dir = Path(preference_cache_dir)
        self.clip_warn_threshold = float(clip_warn_threshold)
        self._type_params: dict[int, dict] = {}
        self._elicited_agent_ids: set[int] = set()
        self._agent_type_idx: Optional[np.ndarray] = None  # (len(agents),) int32
        self._unique_beta: Optional[np.ndarray] = None     # (K,) float64
        self._unique_kappa: Optional[np.ndarray] = None    # (K,) float64
        self._unique_keys: list[str] = []                  # length K, cache keys
        self._last_clip_rates: dict[str, float] = {}
        self._last_renorm_mean_pre: dict[str, float] = {}

    # ------------------------------------------------------------------
    # LLM elicitation — identical clamp + per-dim weight-weighted renorm to
    # V4-A. Builds unique (beta_k, kappa_k) tables indexed by cache key.
    # ------------------------------------------------------------------
    def ensure_elicitation(self, agents: list[Agent], verbose: bool = True) -> None:
        """Elicit LLM preferences and assemble per-type (beta_k, kappa_k).

        Idempotent: re-calling with the same agent set is a no-op.
        """
        ids = {int(a.agent_id) for a in agents}
        if self._elicited_agent_ids == ids and self._type_params:
            return
        if self.elicitor is None:
            raise ValueError(
                "AhlfeldtShockArgmaxHybridEngine requires an LLMPreferenceElicitor; "
                "pass elicitor=... at construction."
            )

        raw = self.elicitor.elicit_batch(
            agents, cache_dir=self.preference_cache_dir, verbose=verbose,
        )
        assert len(raw) == len(agents)
        for agent, pw in zip(agents, raw):
            agent.preferences = pw

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

        # Per-agent (beta_k, kappa_k) — same math as V4-A.
        per_agent_beta = dim_scaled["Q"] * beta_bar
        per_agent_kappa = dim_scaled["tau"] * kappa_bar

        self._type_params = {}
        for i, agent in enumerate(agents):
            self._type_params[int(agent.agent_id)] = {
                "beta":        float(per_agent_beta[i]),
                "kappa":       float(per_agent_kappa[i]),
                "alpha_Q":     float(dim_raw["Q"][i]),
                "alpha_tau":   float(dim_raw["tau"][i]),
                "alpha_B":     float(dim_raw["B"][i]),
                "alpha_w":     float(dim_raw["w"][i]),
                "scaling_Q":   float(dim_scaled["Q"][i]),
                "scaling_tau": float(dim_scaled["tau"][i]),
            }

        # Build unique-type table for fast V_k precompute. Two agents with the
        # same cache key (same demographic tuple) share (beta_k, kappa_k) — we
        # collapse them to a single K-row table and a length-M index vector.
        cache_keys = [_demographic_cache_key(a) for a in agents]
        key_to_unique_idx: dict[str, int] = {}
        unique_beta: list[float] = []
        unique_kappa: list[float] = []
        agent_type_idx = np.empty(len(agents), dtype=np.int32)
        for i, k in enumerate(cache_keys):
            if k not in key_to_unique_idx:
                key_to_unique_idx[k] = len(unique_beta)
                unique_beta.append(float(per_agent_beta[i]))
                unique_kappa.append(float(per_agent_kappa[i]))
            agent_type_idx[i] = key_to_unique_idx[k]

        self._agent_type_idx = agent_type_idx
        self._unique_beta = np.asarray(unique_beta, dtype=np.float64)
        self._unique_kappa = np.asarray(unique_kappa, dtype=np.float64)
        self._unique_keys = list(key_to_unique_idx.keys())
        self._elicited_agent_ids = ids
        self._last_clip_rates = clip_rates
        self._last_renorm_mean_pre = renorm_mean_pre

        logger = logging.getLogger(__name__)
        for name, rate in clip_rates.items():
            if rate > self.clip_warn_threshold:
                logger.warning(
                    "AhlfeldtShockArgmaxHybridEngine: dim %s clips %.1f%% of types; "
                    "pre-renorm mean %.3f.",
                    name, 100.0 * rate, renorm_mean_pre[name],
                )

    # ------------------------------------------------------------------
    # Per-type V_k + Gumbel shock + argmax. Mirrors AhlfeldtABMEngine.decide_batch
    # but swaps the homogeneous V for a (K, N, N) V_per_type tensor indexed per
    # agent.
    # ------------------------------------------------------------------
    def decide_batch(
        self,
        agents: list[Agent],
        environment,
        zone_options: list[str],
        prices: dict,
    ) -> list[LocationChoice]:
        if not agents:
            return []

        self.ensure_elicitation(agents)

        N = len(zone_options)
        zones = list(zone_options)
        EPS = 1e-12

        # ---- Zone vectors (same as parent) ---------------------------
        amen_source = self._current_amenity
        def _resolve_B(z):
            if amen_source and z in amen_source:
                v = amen_source[z]
                if v is not None:
                    return float(v)
            v = environment.get_zone(z).amenity_B
            return float(v) if v else EPS
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

        if environment.transport_matrix is not None and environment.transport_matrix_index:
            tt_idx = [environment._matrix_index_map[z] for z in zones]
            tau = environment.transport_matrix[np.ix_(tt_idx, tt_idx)].astype(
                self._np_dtype, copy=False,
            )
        else:
            tau = np.zeros((N, N), dtype=self._np_dtype)
            for i, zi in enumerate(zones):
                for j, zj in enumerate(zones):
                    tau[i, j] = environment.travel_time(zi, zj)

        log_B = np.log(np.maximum(B, EPS))
        log_w = np.log(np.maximum(w_vec, EPS))
        log_Q = np.log(np.maximum(Q, EPS))

        U_common = (log_B[:, None] + log_w[None, :]).astype(np.float32, copy=False)
        log_Q_col = log_Q.astype(np.float32, copy=False)[:, None]  # (N, 1)
        tau_f32 = tau.astype(np.float32, copy=False)

        # ---- Precompute V_per_type: shape (K, N, N) in float32 --------
        K = int(self._unique_beta.shape[0])
        # V_k[i, j] = U_common[i, j] - (1 - beta_k) * log_Q[i] - kappa_k * tau[i, j]
        beta_f32 = self._unique_beta.astype(np.float32)
        kappa_f32 = self._unique_kappa.astype(np.float32)
        # Broadcast: (K, 1, 1) * (1, N, 1) -> (K, N, 1); (K, 1, 1) * (1, N, N) -> (K, N, N)
        V_per_type = (
            U_common[None, :, :]
            - (1.0 - beta_f32[:, None, None]) * log_Q_col[None, :, :]
            - kappa_f32[:, None, None] * tau_f32[None, :, :]
        ).astype(np.float32, copy=False)

        # ---- Affordability mask (same structural beta_bar as parent) --
        if self.budget_constraint:
            afford_mask = ((1.0 - self.beta) * w_vec[None, :]) >= (Q[:, None] * 1.0)
            if not np.any(afford_mask):
                afford_mask = np.ones_like(afford_mask, dtype=bool)
        else:
            afford_mask = np.ones((N, N), dtype=bool)
        # Apply mask to V_per_type (broadcast over K).
        V_per_type = np.where(
            afford_mask[None, :, :], V_per_type, np.float32(-1e30),
        ).astype(np.float32, copy=False)

        eps_scale = 1.0 / max(float(self.epsilon), EPS)

        # ---- MC loop: M = self.num_agents replicates -----------------
        M = self.num_agents
        if self._agent_type_idx is None:
            raise RuntimeError("agent_type_idx not built; ensure_elicitation missing")
        # For V4 we expand the K-type catalog to M MC replicates. When the
        # agent list already has length M (e.g., 1M explicit agents), use the
        # agent_type_idx directly. When it has length < M (e.g., K-type
        # catalog + MC expansion), we tile weighted by agent.weight.
        if len(agents) == M:
            # Explicit-1M path: the agent list IS the MC replicate set.
            type_idx_M = self._agent_type_idx  # shape (M,), int32
        else:
            # Weighted MC expansion from K agents to M replicates. Uses the
            # master RNG (same seed → same allocation across iterations).
            weights_arr = np.array(
                [float(a.weight) for a in agents], dtype=np.float64,
            )
            weights_arr = weights_arr / max(float(weights_arr.sum()), EPS)
            alloc_rng = np.random.default_rng(
                (self.seed * 2_000_003 + 7) & 0xFFFFFFFF,
            )
            # For each of M replicates, pick a catalog index weighted by
            # agent.weight, then look up its unique-type index.
            catalog_idx = alloc_rng.choice(
                len(agents), size=M, p=weights_arr,
            ).astype(np.int64)
            type_idx_M = self._agent_type_idx[catalog_idx].astype(np.int32)

        HR_count = np.zeros(N, dtype=np.int64)
        HM_count = np.zeros(N, dtype=np.int64)
        P_agg_count = np.zeros((N, N), dtype=np.int64)
        vv_sum = np.zeros(N, dtype=np.float64)
        wage_f64 = w_vec.astype(np.float64)

        samples: list[LocationChoice] = []
        sample_stride = None
        if self.store_agent_samples > 0:
            sample_stride = max(1, M // self.store_agent_samples)

        rng = self._fresh_rng()
        n_batches = (M + self.batch_size - 1) // self.batch_size
        for b in range(n_batches):
            start = b * self.batch_size
            stop = min(start + self.batch_size, M)
            this_batch = stop - start
            shocks = self._draw_shocks_batch(rng, this_batch, N)  # (B, N, N) f32
            # Fancy-index the per-batch V slices. Shape: (B, N, N) f32.
            V_batch = V_per_type[type_idx_M[start:stop]]
            # eff = V_k + shock / eps_structural
            eff = V_batch + shocks * np.float32(eps_scale)
            # Argmax across flattened (i, j).
            flat = eff.reshape(eff.shape[0], -1).argmax(axis=1)
            i_star = flat // N
            j_star = flat - i_star * N
            np.add.at(HR_count, i_star, 1)
            np.add.at(HM_count, j_star, 1)
            np.add.at(P_agg_count, (i_star, j_star), 1)
            np.add.at(vv_sum, i_star, wage_f64[j_star])

            if sample_stride is not None:
                for offset in range(0, eff.shape[0], sample_stride):
                    k = offset
                    samples.append(
                        LocationChoice(
                            residence=zones[int(i_star[k])],
                            workplace=zones[int(j_star[k])],
                            utility=float(V_batch[k, int(i_star[k]), int(j_star[k])]),
                            zone_utilities={},
                        )
                    )

        # Empirical P_agg (shares).
        P_agg = (P_agg_count.astype(np.float64) / float(M)).astype(
            self._np_dtype, copy=False
        )
        self.last_choice_probabilities = P_agg

        # Diagnostics. Weighted expected utility uses the POPULATION-MEAN
        # V (across types) — a reasonable mean-field summary; per-agent
        # utility is available via samples if requested.
        V_mean = V_per_type.mean(axis=0).astype(np.float64)
        utilities_mean = float(np.sum(P_agg.astype(np.float64) * V_mean))

        self.last_abm_diagnostics = {
            "num_agents":                  M,
            "n_unique_types":              K,
            "batch_size":                  self.batch_size,
            "n_batches":                   n_batches,
            "shock_distribution":          self.shock_distribution,
            "epsilon":                     float(self.epsilon),
            "n_cells_nonzero":             int(np.sum(P_agg_count > 0)),
            "total_cells":                 int(P_agg_count.size),
            "n_residence_marginal_zero":   int(np.sum(HR_count == 0)),
            "n_workplace_marginal_zero":   int(np.sum(HM_count == 0)),
            "HR_min":                      int(HR_count.min()),
            "HR_max":                      int(HR_count.max()),
            "HM_min":                      int(HM_count.min()),
            "HM_max":                      int(HM_count.max()),
            "expected_utility":            utilities_mean,
            "clip_rates":                  dict(self._last_clip_rates),
            "renorm_means_pre":            dict(self._last_renorm_mean_pre),
        }

        logger = logging.getLogger(__name__)
        if self.last_abm_diagnostics["n_residence_marginal_zero"] > 0:
            empty_res = [zones[i] for i in range(N) if HR_count[i] == 0][:5]
            logger.warning(
                "V4: %d Ortsteile have zero residence marginal; first few: %s",
                self.last_abm_diagnostics["n_residence_marginal_zero"], empty_res,
            )
        if self.last_abm_diagnostics["n_workplace_marginal_zero"] > 0:
            empty_wp = [zones[j] for j in range(N) if HM_count[j] == 0][:5]
            logger.warning(
                "V4: %d Ortsteile have zero workplace marginal; first few: %s",
                self.last_abm_diagnostics["n_workplace_marginal_zero"], empty_wp,
            )

        # Per-input-agent LocationChoice records (population-modal cell).
        results: list[LocationChoice] = []
        flat_mode = int(P_agg_count.argmax())
        i_mode = flat_mode // N
        j_mode = flat_mode - i_mode * N
        for a in agents:
            results.append(
                LocationChoice(
                    residence=zones[i_mode],
                    workplace=zones[j_mode],
                    utility=float(V_mean[i_mode, j_mode]),
                    zone_utilities={},
                )
            )

        self.last_agent_samples = samples
        self.last_diagnostics = {
            "expected_utility": utilities_mean,
            "n_agents": M,
            "n_zones": N,
            "sampling_path": "v4-shock-argmax-hybrid",
            "B_mean": float(np.mean(B)),
            "B_std": float(np.std(B)),
        }
        return results

    def type_parameters(self) -> dict[int, dict]:
        """Per-agent (beta_k, kappa_k) table for downstream persistence."""
        return dict(self._type_params)
