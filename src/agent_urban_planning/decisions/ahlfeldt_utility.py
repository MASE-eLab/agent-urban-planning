"""Ahlfeldt-Cobb-Douglas decision engine for Berlin replication.

Implements the Ahlfeldt et al. (2015) indirect utility

    ln u_ij = ln B_i + ln w_j - (1 - beta) * ln Q_i - kappa * tau_ij

where ``i`` is the residence zone, ``j`` is the workplace zone, ``B_i``
is the (endogenous or exogenous) amenity at ``i``, ``w_j`` is the wage
at ``j``, ``Q_i`` is the residential floor price per m², ``tau_ij`` is
the bilateral travel time, and ``kappa = kappa_eps / epsilon`` is the
per-minute commute-cost decay.

Each agent has a per-(R, W) Fréchet(ε) idiosyncratic shock, drawn once
at the start of the run (per-agent sub-RNG seeded deterministically
from the global seed and agent id) and held fixed across all tatonnement
iterations within the run. Implementation uses the Gumbel trick:

    argmax_{i,j} (ln u_ij + g_ij / epsilon),  g_ij ~ Gumbel(0, 1)

For scenarios with up to ``large_N_threshold`` zones (default 200 —
covers Bezirke and Ortsteile), we enumerate the full ``N × N`` matrix
per agent. For larger scenarios (block-level), we use the factorized
marginal-then-conditional sampling strategy (``sample_factorized``).

Current wages are injected by ``AhlfeldtMarket`` before each batch
call via :meth:`set_current_wages`. If unset, the engine falls back
to ``zone.wage_observed`` from the scenario YAML.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from agent_urban_planning.core.agents import Agent
from agent_urban_planning.data.loaders import AhlfeldtParams
from agent_urban_planning.decisions.base import LocationChoice
from agent_urban_planning.core.environment import Environment


class AhlfeldtUtilityEngine:
    """Cobb-Douglas + Fréchet joint residence-workplace engine.

    Three code paths depending on scenario scale and the `sampling_method`
    kwarg:

    - ``gumbel`` (small-N default) — per-agent ``N × N`` Gumbel shock
      matrix stored in ``_shock_cache``; argmax over ``log_Phi + g/ε``.
      Used for Bezirke (N=23) and Ortsteile (N=97) where memory is trivial.
    - ``multinomial`` (factorized, large-N default) — shared ``log_Phi``
      computed once per ``decide_batch`` call; each agent samples from
      ``softmax(ε · log_Phi)`` via ``rng.choice(N², p=P_flat)``. Memory
      footprint is independent of agent count; used at block scale (N=12k).
    - ``deterministic`` — continuum-limit interpretation: each agent
      contributes fractional weight ``P_ij · weight`` to every (i, j).
      Matches Ahlfeldt's closed-form aggregate equilibrium exactly, no
      Monte Carlo noise. Used as the PRIMARY block-level path for
      pack-reproducing results.
    """

    def __init__(
        self,
        params: AhlfeldtParams,
        seed: Optional[int] = None,
        large_N_threshold: int = 200,
        budget_constraint: bool = True,
        sampling_method: str = "auto",
        deterministic: bool = False,
        dtype: str = "float64",
    ):
        self.params = params
        self.seed = int(seed) if seed is not None else 0
        self.large_N_threshold = int(large_N_threshold)
        self.budget_constraint = bool(budget_constraint)

        if sampling_method not in ("auto", "gumbel", "multinomial"):
            raise ValueError(
                f"sampling_method must be one of auto|gumbel|multinomial; got {sampling_method}"
            )
        self.sampling_method = sampling_method
        self.deterministic = bool(deterministic)

        if dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be 'float32' or 'float64'; got {dtype}")
        self.dtype = dtype
        self._np_dtype = np.float32 if dtype == "float32" else np.float64

        # Derived constants
        self.alpha = params.alpha
        self.beta = params.beta
        self.epsilon = params.epsilon
        self.kappa = params.kappa  # = kappa_eps / epsilon

        # Current wages injected by the market. Keyed by zone name. When
        # unset the engine falls back to zone.wage_observed.
        self._current_wages: Optional[dict[str, float]] = None

        # Current productivity / amenity injected by the market when
        # endogenous agglomeration is active. Both dicts are keyed by
        # zone name. When unset the engine falls back to Zone.productivity_A
        # / Zone.amenity_B read from the environment.
        self._current_productivity: Optional[dict[str, float]] = None
        self._current_amenity: Optional[dict[str, float]] = None

        # Diagnostics populated per decide_batch call
        self.last_diagnostics: dict[str, float] = {}

        # Per-agent Fréchet shock cache (only used in `gumbel` path).
        self._shock_cache: dict[int, np.ndarray] = {}
        self._shock_cache_n_zones: Optional[int] = None

        # Full choice-probability matrix populated by deterministic mode.
        # Shape (N, N); None otherwise. Consumed by AhlfeldtMarket when
        # aggregating demand.
        self.last_choice_probabilities: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Protocol: set_cache (no-op — this engine doesn't use LLM cache)
    # ------------------------------------------------------------------
    def set_cache(self, cache) -> None:
        return None

    # ------------------------------------------------------------------
    # Public state injection
    # ------------------------------------------------------------------
    def set_current_wages(self, wages: dict[str, float]) -> None:
        """Inject current wage vector from the market loop."""
        self._current_wages = dict(wages)

    def set_current_productivity(self, A: dict[str, float]) -> None:
        """Inject per-zone productivity A_i from the market loop.

        Called each iteration by AhlfeldtMarket when endogenous
        agglomeration is active. Keys are zone names; values are the
        damped A_i after the per-iteration update. Missing keys fall
        back to Zone.productivity_A at utility-computation time.
        """
        self._current_productivity = dict(A)

    def set_current_amenity(self, B: dict[str, float]) -> None:
        """Inject per-zone amenity B_i from the market loop.

        Mirrors set_current_productivity. The indirect-utility formula
        reads ln B_i via this injection when present; falls back to
        Zone.amenity_B per zone for missing keys.
        """
        self._current_amenity = dict(B)

    @property
    def price_elasticity(self) -> float:
        """Market clearing code queries this for tatonnement step sizing.

        For Ahlfeldt scenarios the floor-price elasticity is derived
        structurally: ``(1 - beta) * epsilon``.
        """
        if self.params.eta_floor_override is not None:
            return float(self.params.eta_floor_override)
        return (1.0 - self.beta) * self.epsilon

    # ------------------------------------------------------------------
    # Decide entry points
    # ------------------------------------------------------------------
    def decide(
        self,
        agent: Agent,
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> LocationChoice:
        """Single-agent decide — delegates to batch for consistency."""
        return self.decide_batch([agent], environment, zone_options, prices)[0]

    def decide_batch(
        self,
        agents: list[Agent],
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> list[LocationChoice]:
        """Joint (R, W) choice for a batch of agents.

        ``prices`` holds residential floor prices Q_i keyed by zone name.
        Current wages are read from :attr:`_current_wages` (set via
        :meth:`set_current_wages`) or fall back to ``zone.wage_observed``.
        """
        if not agents:
            return []

        N = len(zone_options)
        zones = list(zone_options)
        zone_to_idx = {z: i for i, z in enumerate(zones)}

        # --- Build zone-ordered parameter vectors -----------------------
        # Directly in the target dtype to avoid a post-hoc astype() pass.
        Q = np.array(
            [float(prices.get(z, environment.get_zone(z).floor_price_observed))
             for z in zones],
            dtype=self._np_dtype,
        )
        # Amenity B_i: priority is injected value (endogenous agglomeration)
        # → Zone.amenity_B fallback. Peripheral blocks legitimately have
        # zero amenity; we keep them near-zero rather than defaulting to
        # 1.0 which would spuriously attract agents to uninhabitable zones.
        amen_source = self._current_amenity
        def _resolve_B(z):
            if amen_source and z in amen_source:
                v = amen_source[z]
                if v is not None:
                    return float(v)
            v = environment.get_zone(z).amenity_B
            return float(v) if v else 1e-12
        B = np.array([_resolve_B(z) for z in zones], dtype=self._np_dtype)

        # Productivity A_i
        prod_source = self._current_productivity
        A_vec: Optional[np.ndarray] = None
        if prod_source is not None:
            def _resolve_A(z):
                if z in prod_source:
                    v = prod_source[z]
                    if v is not None:
                        return float(v)
                v = environment.get_zone(z).productivity_A
                return float(v) if v else 1e-12
            A_vec = np.array([_resolve_A(z) for z in zones], dtype=self._np_dtype)

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

        # --- Build travel-time matrix (in target dtype) ----------------
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

        # --- Base log-utility matrix U[i, j] ---------------------------
        # ln u_ij = ln B_i + ln w_j - (1-beta)*ln Q_i - kappa * tau_ij
        # Guard against non-positive values to avoid NaN — clamp at a tiny
        # positive floor.
        EPS = 1e-12
        log_B = np.log(np.maximum(B, EPS))
        log_w = np.log(np.maximum(w_vec, EPS))
        log_Q = np.log(np.maximum(Q, EPS))

        # Broadcasting: U[i, j] = log_B[i] + log_w[j] - (1-beta)*log_Q[i] - kappa*tau[i, j]
        U = (
            log_B[:, None]
            + log_w[None, :]
            - (1.0 - self.beta) * log_Q[:, None]
            - self.kappa * tau
        )

        # --- Budget constraint mask ------------------------------------
        # For Cobb-Douglas preferences the agent spends beta share of
        # income on goods and (1 - beta) on floor. Income at workplace j
        # equals w_j (unit labor supply). Floor "affordability" is
        # loosely: the agent must be able to afford at least one unit of
        # residential floor at Q_i while spending (1 - beta) of w_j.
        #   (1 - beta) * w_j >= Q_i * min_unit
        # We set min_unit = 1.0 (one m²), which is ultra-loose for
        # typical wages/rents and in practice never binds for Ahlfeldt's
        # calibration. This matches the closed-form model which admits
        # all (i, j) pairs with positive probability.
        if self.budget_constraint:
            afford_mask = ((1.0 - self.beta) * w_vec[None, :]) >= (Q[:, None] * 1.0)
            if not np.any(afford_mask):
                # Guard against pathological scenarios where nothing is
                # affordable: fall back to "everyone can live/work anywhere".
                afford_mask = np.ones_like(afford_mask, dtype=bool)
        else:
            afford_mask = np.ones((N, N), dtype=bool)

        # U is already in self._np_dtype if the input vectors were; no cast needed.
        if U.dtype != self._np_dtype:
            U = U.astype(self._np_dtype, copy=False)

        # Precompute shared zone_utilities (same for all agents in this call).
        # Vectorized per-row/per-col mean — replaces a 2N-length Python loop.
        row_means = U.mean(axis=1)
        col_means = U.mean(axis=0)
        zone_utils_shared: dict[str, float] = {}
        for idx, z in enumerate(zones):
            zone_utils_shared[f"R:{z}"] = float(row_means[idx])
            zone_utils_shared[f"W:{z}"] = float(col_means[idx])

        # --- Dispatch to sampling path ---------------------------------
        path = self._resolve_sampling_path(N, agents)

        if path == "deterministic":
            results, utilities = self._decide_deterministic(
                agents, zones, U, afford_mask, zone_utils_shared
            )
        elif path == "multinomial":
            results, utilities = self._decide_multinomial(
                agents, zones, U, afford_mask, zone_utils_shared
            )
        else:  # gumbel
            results, utilities = self._decide_gumbel(
                agents, zones, U, afford_mask, zone_utils_shared
            )

        # --- Diagnostics summary --------------------------------------
        # Entropy of residence-marginal choice distribution (nats).
        residences = [r.residence for r in results]
        if residences:
            counts = np.zeros(N, dtype=np.float64)
            for res in residences:
                counts[zone_to_idx[res]] += 1.0
            probs = counts / max(counts.sum(), 1.0)
            nonzero = probs[probs > 0]
            entropy_nats = float(-np.sum(nonzero * np.log(nonzero)))
        else:
            entropy_nats = 0.0

        self.last_diagnostics = {
            "expected_utility": float(np.mean(utilities)) if utilities.size else 0.0,
            "entropy_residence_nats": entropy_nats,
            "utility_min": float(np.min(utilities)) if utilities.size else 0.0,
            "utility_max": float(np.max(utilities)) if utilities.size else 0.0,
            "utility_mean": float(np.mean(utilities)) if utilities.size else 0.0,
            "n_agents": len(results),
            "n_zones": N,
            "sampling_path": path,
            # Always report B distribution (injected or from YAML fallback)
            "B_mean": float(np.mean(B)),
            "B_std": float(np.std(B)),
        }
        # Add injected-productivity diagnostics when endogenous agglomeration
        # is active (i.e. market has called set_current_productivity).
        if A_vec is not None:
            self.last_diagnostics.update(
                {
                    "A_mean": float(np.mean(A_vec)),
                    "A_std": float(np.std(A_vec)),
                    "A_injected_mean": float(np.mean(A_vec)),
                }
            )

        return results

    # ------------------------------------------------------------------
    # Sampling path dispatch
    # ------------------------------------------------------------------
    def _resolve_sampling_path(self, N: int, agents: list[Agent]) -> str:
        """Pick one of {"deterministic", "multinomial", "gumbel"} for this call.

        Priority:
          1. deterministic=True on engine → "deterministic" always
          2. sampling_method forced → honor request (warn on memory-heavy combos)
          3. Heterogeneous agents → always use "gumbel" (factorized path assumes
             homogeneous utility, so we fall back)
          4. Auto: N > large_N_threshold → "multinomial"; else "gumbel"
        """
        if self.deterministic:
            return "deterministic"

        # Detect heterogeneity (varying income or preferences)
        if len(agents) > 1:
            first = agents[0]
            for a in agents[1:]:
                if (
                    a.income != first.income
                    or a.preferences.alpha != first.preferences.alpha
                ):
                    # Heterogeneous — the shared log_Phi assumption breaks.
                    # Force gumbel path unless user explicitly forced multinomial.
                    if self.sampling_method == "multinomial":
                        raise NotImplementedError(
                            "multinomial path requires homogeneous agents; got heterogeneous"
                        )
                    return "gumbel"

        if self.sampling_method == "gumbel":
            if N > self.large_N_threshold:
                import warnings
                warnings.warn(
                    f"Forced sampling_method='gumbel' at N={N} > threshold "
                    f"{self.large_N_threshold}; memory cost is O(K·N²). "
                    f"Consider 'multinomial' or 'deterministic'.",
                    UserWarning,
                )
            return "gumbel"
        if self.sampling_method == "multinomial":
            return "multinomial"
        # auto
        return "multinomial" if N > self.large_N_threshold else "gumbel"

    def _decide_gumbel(
        self,
        agents: list[Agent],
        zones: list[str],
        U: np.ndarray,
        afford_mask: np.ndarray,
        zone_utils: dict[str, float],
    ) -> tuple[list[LocationChoice], np.ndarray]:
        """Per-agent Gumbel-max on a stored N×N shock matrix (small-N path)."""
        N = len(zones)
        EPS = 1e-12
        results: list[LocationChoice] = []
        for agent in agents:
            shocks = self._get_shocks(agent.agent_id, N)
            # Cast shocks to match U dtype
            if shocks.dtype != U.dtype:
                shocks = shocks.astype(U.dtype, copy=False)
            eff = U + shocks / max(self.epsilon, EPS)
            eff = np.where(afford_mask, eff, -np.inf)
            flat_idx = int(np.argmax(eff))
            i = flat_idx // N
            j = flat_idx % N
            results.append(
                LocationChoice(
                    residence=zones[i],
                    workplace=zones[j],
                    utility=float(eff[i, j]),
                    zone_utilities=dict(zone_utils),
                )
            )
        utilities = np.array([r.utility for r in results], dtype=np.float64)
        self.last_choice_probabilities = None
        return results, utilities

    def _compute_choice_probs(
        self, U: np.ndarray, afford_mask: np.ndarray
    ) -> np.ndarray:
        """Shared softmax of ε·U with the affordability mask applied.

        Returns an ``(N, N)`` probability matrix summing to 1.0.
        """
        logits = self.epsilon * U
        logits = np.where(afford_mask, logits, -np.inf)
        # logsumexp trick for numerical stability in float32
        max_logit = float(np.max(logits[np.isfinite(logits)]))
        shifted = logits - max_logit
        exp_shifted = np.exp(shifted)
        exp_shifted[~np.isfinite(logits)] = 0.0
        total = float(exp_shifted.sum())
        if total <= 0.0:
            # Degenerate: uniform over zones
            P = np.ones_like(U) / (U.size)
        else:
            P = exp_shifted / total
        return P.astype(self._np_dtype, copy=False)

    def _decide_multinomial(
        self,
        agents: list[Agent],
        zones: list[str],
        U: np.ndarray,
        afford_mask: np.ndarray,
        zone_utils: dict[str, float],
    ) -> tuple[list[LocationChoice], np.ndarray]:
        """Factorized sampling: shared softmax + per-agent multinomial draw.

        CDF + searchsorted pattern avoids the O(N²) cumsum inside
        ``rng.choice(p=...)``; each agent's per-hash sub-RNG produces a
        single uniform draw used as the inverse-CDF input.
        """
        N = len(zones)
        P = self._compute_choice_probs(U, afford_mask)
        self.last_choice_probabilities = P
        P_flat = P.reshape(-1).astype(np.float64, copy=False)
        s = P_flat.sum()
        if s > 0:
            P_flat = P_flat / s
        cdf = np.cumsum(P_flat)
        cdf[-1] = 1.0
        log_u_flat = U.reshape(-1)

        # Per-agent deterministic uniform via hash(seed, agent_id)
        uniforms = np.empty(len(agents), dtype=np.float64)
        for k, agent in enumerate(agents):
            sub_seed = abs(
                hash((self.seed, int(agent.agent_id), "frechet_shocks"))
            ) % (2**32 - 1)
            uniforms[k] = np.random.default_rng(sub_seed).random()
        flat_indices = np.searchsorted(cdf, uniforms, side="right")
        np.clip(flat_indices, 0, P_flat.size - 1, out=flat_indices)

        shared_utils = zone_utils
        results: list[LocationChoice] = []
        for flat_idx in flat_indices:
            idx = int(flat_idx)
            i = idx // N
            j = idx - i * N
            results.append(
                LocationChoice(
                    residence=zones[i],
                    workplace=zones[j],
                    utility=float(log_u_flat[idx]),
                    zone_utilities=shared_utils,
                )
            )
        utilities = np.asarray(log_u_flat[flat_indices], dtype=np.float64)
        return results, utilities

    def _decide_deterministic(
        self,
        agents: list[Agent],
        zones: list[str],
        U: np.ndarray,
        afford_mask: np.ndarray,
        zone_utils: dict[str, float],
    ) -> tuple[list[LocationChoice], np.ndarray]:
        """Continuum mode: shared P_ij drives both the market-level aggregate
        demand (exposed via ``self.last_choice_probabilities``) AND the
        per-agent allocations (via a deterministic multinomial sampler seeded
        from ``self.seed``).

        This has two semantic properties:

        1. **Zero run-to-run variance**: with a fixed engine seed, the exact
           same per-agent (i, j) sequence is produced on every replicate.
        2. **Per-agent distribution matches P**: agents are allocated
           proportionally to `P_ij`, so standard aggregations over
           ``agent_results`` (commute, zone_populations, zone_employment)
           converge to their continuum values at the law-of-large-numbers
           rate.

        The market layer additionally consumes ``last_choice_probabilities``
        to compute demand shares exactly from the continuum rather than the
        finite sample — that path gives zero aggregation error regardless
        of agent count.
        """
        N = len(zones)
        P = self._compute_choice_probs(U, afford_mask)
        self.last_choice_probabilities = P
        P_flat = P.reshape(-1).astype(np.float64, copy=False)
        s = P_flat.sum()
        if s > 0:
            P_flat = P_flat / s

        # Seeded deterministic sampler via single-pass cumulative distribution.
        # rng.choice with p builds a cumsum internally per call — at N²=151M
        # and 1000+ agents that's prohibitive. We build the CDF ONCE and
        # binary-search per agent via searchsorted.
        det_seed = abs(hash((self.seed, "deterministic-sampler"))) % (2**32 - 1)
        rng = np.random.default_rng(det_seed)
        cdf = np.cumsum(P_flat)
        cdf[-1] = 1.0  # guard against rounding
        u_samples = rng.random(len(agents))
        flat_indices = np.searchsorted(cdf, u_samples, side="right")
        # Clamp out-of-bounds from float rounding
        np.clip(flat_indices, 0, P_flat.size - 1, out=flat_indices)

        # Population-level expected utility for diagnostics
        log_u_flat = U.reshape(-1)

        # Batch-construct LocationChoice objects. All agents share the
        # same zone_utils dict (homogeneous assumption) — no per-agent copy.
        shared_utils = zone_utils  # same ref across agents
        results: list[LocationChoice] = []
        for flat_idx in flat_indices:
            idx = int(flat_idx)
            i = idx // N
            j = idx - i * N
            results.append(
                LocationChoice(
                    residence=zones[i],
                    workplace=zones[j],
                    utility=float(log_u_flat[idx]),
                    zone_utilities=shared_utils,
                )
            )
        utilities = np.asarray(log_u_flat[flat_indices], dtype=np.float64)
        return results, utilities

    # ------------------------------------------------------------------
    # Fréchet / Gumbel shock cache
    # ------------------------------------------------------------------
    def _get_shocks(self, agent_id: int, N: int) -> np.ndarray:
        """Return cached Gumbel(0, 1) shocks for (R, W) pairs of this agent.

        Shocks are drawn from a per-agent sub-RNG seeded as
        ``hash((self.seed, agent_id, "frechet_shocks"))``. They are drawn
        lazily on first request and reused unchanged across iterations.

        If the zone-count changes between calls (rare — only during
        scenario reconfiguration), the cache is invalidated.
        """
        if self._shock_cache_n_zones is not None and self._shock_cache_n_zones != N:
            self._shock_cache.clear()
        self._shock_cache_n_zones = N

        if agent_id in self._shock_cache:
            return self._shock_cache[agent_id]

        sub_seed = abs(
            hash((self.seed, int(agent_id), "frechet_shocks"))
        ) % (2**32 - 1)
        rng = np.random.default_rng(sub_seed)
        shocks = rng.gumbel(loc=0.0, scale=1.0, size=(N, N)).astype(self._np_dtype)
        self._shock_cache[agent_id] = shocks
        return shocks

    # ------------------------------------------------------------------
    # Optional: reset shock cache (for Monte Carlo replicates)
    # ------------------------------------------------------------------
    def clear_shock_cache(self) -> None:
        self._shock_cache.clear()
        self._shock_cache_n_zones = None
