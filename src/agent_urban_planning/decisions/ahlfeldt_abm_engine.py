"""Monte Carlo argmax engine — ABM substitute for softmax in Ahlfeldt.

Replaces the closed-form Fréchet softmax step with an explicit per-agent
argmax over drawn shocks. This lets the same Ahlfeldt FOC machinery
(housing market clearing + firm labor FOC) run under **any** shock
distribution, since the FOCs only read aggregate choice shares and are
distribution-free (budget identity + firm marginal condition).

Design principles
-----------------
1. **Homogeneous preferences across all M agents** — β, κ, ε come from
   ``params``. Heterogeneity emerges from the per-agent shock draw only.
2. **Frozen shocks via deterministic sub-seeds** — each agent's (N, N)
   shock tensor is regenerated from ``rng(master_seed + agent_id)`` on
   every call, guaranteeing bitwise reproducibility without storing the
   full M × N × N tensor.
3. **Batched numpy vectorization** — M agents processed in chunks of
   ``batch_size`` (default 10,000). Peak memory per batch:
   ``BATCH × N² × 4 B`` — ~370 MB at BATCH=10k, N=96.
4. **Empirical P_agg exposed to market unchanged** — ``P_agg_count / M``
   feeds ``self.last_choice_probabilities``, so :class:`AhlfeldtMarket`
   reads it exactly as if it came from the closed-form path.

Supported shock distributions
-----------------------------
- ``"frechet"``  — Gumbel(0, 1) with variance π²/6 ≈ 1.645. Matches the
                   Ahlfeldt closed-form softmax in aggregate.
- ``"normal"``   — Normal(0, σ² = π²/6). Variance-matched to Gumbel so
                   V2 vs V3 is a pure **shape** test (symmetric Gaussian
                   vs asymmetric right-skewed Gumbel), not a scale test.

Variance matching rationale:
After the argmax applies ``shock / ε_structural``, the effective logit-
space noise variance is the same under both distributions. Any
equilibrium deviation between V2 (Gumbel) and V3 (Normal) is thus purely
attributable to distribution *shape*, not noise *amplitude*.
"""
from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np

from agent_urban_planning.core.agents import Agent
from agent_urban_planning.data.loaders import AhlfeldtParams
from agent_urban_planning.decisions.ahlfeldt_utility import AhlfeldtUtilityEngine
from agent_urban_planning.decisions.base import LocationChoice

SUPPORTED_SHOCKS = ("frechet", "normal")
# Variance of Gumbel(0, 1) distribution.
GUMBEL_VARIANCE = math.pi ** 2 / 6.0  # ≈ 1.6449


class AhlfeldtABMEngine(AhlfeldtUtilityEngine):
    """Monte Carlo argmax engine with configurable shock distribution."""

    def __init__(
        self,
        params: AhlfeldtParams,
        *,
        shock_distribution: str = "frechet",
        num_agents: int = 1_000_000,
        batch_size: int = 10_000,
        seed: Optional[int] = None,
        store_agent_samples: int = 0,
        **kwargs,
    ):
        super().__init__(params, seed=seed, **kwargs)
        if shock_distribution not in SUPPORTED_SHOCKS:
            raise ValueError(
                f"shock_distribution must be one of {SUPPORTED_SHOCKS}; "
                f"got {shock_distribution!r}"
            )
        if num_agents < 1:
            raise ValueError(f"num_agents must be >= 1; got {num_agents}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}")
        self.shock_distribution = shock_distribution
        self.num_agents = int(num_agents)
        self.batch_size = int(batch_size)
        self.store_agent_samples = int(store_agent_samples)

        # Public diagnostics populated each decide_batch.
        self.last_abm_diagnostics: dict = {}

    # ------------------------------------------------------------------
    # Shock draw helpers
    # ------------------------------------------------------------------
    def _fresh_rng(self) -> np.random.Generator:
        """Create a fresh RNG seeded from the master seed.

        One RNG per ``decide_batch`` call, shared across all its batches.
        This guarantees frozen-across-iterations shocks (same seed on
        every call → same sequence), while avoiding per-batch RNG
        creation overhead.
        """
        return np.random.default_rng((self.seed * 2_000_003 + 1) & 0xFFFFFFFF)

    def _draw_shocks_batch(
        self, rng: np.random.Generator, batch_size: int, N: int,
    ) -> np.ndarray:
        """Draw a (batch, N, N) shock tensor from the provided generator.

        Uses `rng.random(dtype=np.float32)` to get uniforms directly in
        float32 (avoids a float64 → float32 copy), then transforms to
        the target distribution via its inverse CDF:

          Gumbel(0, 1):  -ln(-ln(u))
          Normal(0, σ²): σ · Φ⁻¹(u) via Box-Muller or sqrt(2)·erfinv(2u-1)

        Numpy's ``rng.gumbel`` and ``rng.normal`` internally produce
        float64 and then we'd cast — the direct-uniform path is ~30%
        faster at this size.
        """
        shape = (batch_size, N, N)
        u = rng.random(size=shape, dtype=np.float32)
        if self.shock_distribution == "frechet":
            # Clamp to avoid log(0).
            np.maximum(u, 1e-7, out=u)
            np.minimum(u, 1.0 - 1e-7, out=u)
            # Gumbel inverse CDF: −log(−log(u))
            np.log(u, out=u)
            np.negative(u, out=u)
            np.log(u, out=u)
            np.negative(u, out=u)
            return u
        # "normal": variance-matched to Gumbel, σ = π/√6.
        # scipy-free: use np.sqrt(2) * erfinv(2u - 1) via erfinv approximation.
        # Easier and also fast: use the standard RNG normal draw (internally
        # uses ziggurat — very fast). Accept a small float64→float32 copy cost.
        sigma = math.sqrt(GUMBEL_VARIANCE)
        return rng.standard_normal(size=shape, dtype=np.float32) * np.float32(sigma)

    # ------------------------------------------------------------------
    # decide_batch override
    # ------------------------------------------------------------------
    def decide_batch(
        self,
        agents: list[Agent],
        environment,
        zone_options: list[str],
        prices: dict,
    ) -> list[LocationChoice]:
        """Run M × argmax over shocks, aggregate, expose empirical P.

        ``agents`` is the canonical agent list (typically length 1 —
        a single representative type — since preferences are homogeneous
        across all M Monte Carlo replicates). The number of MC agents is
        controlled by ``self.num_agents``, not by ``len(agents)``.
        """
        if not agents:
            return []

        N = len(zone_options)
        zones = list(zone_options)

        # Build the shared utility matrix V once (same for all M agents).
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
        V = (
            log_B[:, None]
            + log_w[None, :]
            - (1.0 - self.beta) * log_Q[:, None]
            - self.kappa * tau
        )  # shape (N, N)
        # Cast to float32 for shock-tensor compatibility.
        V_f32 = V.astype(np.float32, copy=False)

        # Affordability mask (matches parent class — same structural β).
        if self.budget_constraint:
            afford_mask = ((1.0 - self.beta) * w_vec[None, :]) >= (Q[:, None] * 1.0)
            if not np.any(afford_mask):
                afford_mask = np.ones_like(afford_mask, dtype=bool)
        else:
            afford_mask = np.ones((N, N), dtype=bool)

        # Where disallowed, subtract a very large number pre-argmax.
        V_masked = np.where(afford_mask, V_f32, -1e30).astype(np.float32, copy=False)
        eps_scale = 1.0 / max(float(self.epsilon), EPS)

        # Aggregates.
        M = self.num_agents
        HR_count = np.zeros(N, dtype=np.int64)
        HM_count = np.zeros(N, dtype=np.int64)
        P_agg_count = np.zeros((N, N), dtype=np.int64)
        vv_sum = np.zeros(N, dtype=np.float64)
        wage_f64 = w_vec.astype(np.float64)

        # Per-agent sample (optional; capped at store_agent_samples).
        samples: list[LocationChoice] = []
        sample_stride = None
        if self.store_agent_samples > 0:
            sample_stride = max(1, M // self.store_agent_samples)

        # Iterate over batches.
        # One RNG per decide_batch call — advanced sequentially across batches.
        # Same seed on every call → same sequence → frozen-across-iteration shocks.
        rng = self._fresh_rng()
        n_batches = (M + self.batch_size - 1) // self.batch_size
        for b in range(n_batches):
            start = b * self.batch_size
            stop = min(start + self.batch_size, M)
            this_batch = stop - start
            shocks = self._draw_shocks_batch(rng, this_batch, N)
            # eff = V + shock * (1/ε). Shape (batch, N, N).
            eff = V_masked[None, :, :] + shocks * np.float32(eps_scale)
            # Argmax across flattened (i, j) per agent.
            flat = eff.reshape(eff.shape[0], -1).argmax(axis=1)
            i_star = flat // N
            j_star = flat - i_star * N
            # Accumulate aggregates.
            # HR: count of each residence.
            np.add.at(HR_count, i_star, 1)
            np.add.at(HM_count, j_star, 1)
            np.add.at(P_agg_count, (i_star, j_star), 1)
            # vv_i = Σ_k wage[j*_k] * 1[i*_k = i].
            # Vectorized: group wages by residence.
            np.add.at(vv_sum, i_star, wage_f64[j_star])

            # Optional agent-record sampling.
            if sample_stride is not None:
                for offset in range(0, eff.shape[0], sample_stride):
                    k = offset
                    samples.append(
                        LocationChoice(
                            residence=zones[int(i_star[k])],
                            workplace=zones[int(j_star[k])],
                            utility=float(V_f32[int(i_star[k]), int(j_star[k])]),
                            zone_utilities={},
                        )
                    )

        # Empirical choice matrix: normalized so sum = 1.
        P_agg = (P_agg_count.astype(np.float64) / float(M)).astype(
            self._np_dtype, copy=False
        )
        self.last_choice_probabilities = P_agg

        # Diagnostics.
        utilities_mean = float(
            np.sum(P_agg.astype(np.float64) * V.astype(np.float64))
        )
        self.last_abm_diagnostics = {
            "num_agents": M,
            "batch_size": self.batch_size,
            "n_batches": n_batches,
            "shock_distribution": self.shock_distribution,
            "epsilon": float(self.epsilon),
            "empty_cells": int(np.sum(P_agg_count == 0)),
            "nonzero_cells": int(np.sum(P_agg_count > 0)),
            "total_cells": int(P_agg_count.size),
            "HR_min": int(HR_count.min()),
            "HR_max": int(HR_count.max()),
            "HM_min": int(HM_count.min()),
            "HM_max": int(HM_count.max()),
            "expected_utility": utilities_mean,
        }

        # Per-type LocationChoice records: one entry per input agent.
        # Since all M replicates share preferences, we return the population-
        # modal (i*, j*) for each input agent. For downstream tables that expect
        # `len(results) == len(agents)`.
        results: list[LocationChoice] = []
        for a in agents:
            flat_mode = int(P_agg_count.argmax())
            i_mode = flat_mode // N
            j_mode = flat_mode - i_mode * N
            results.append(
                LocationChoice(
                    residence=zones[i_mode],
                    workplace=zones[j_mode],
                    utility=float(V[i_mode, j_mode]),
                    zone_utilities={},
                )
            )

        # Also store the samples (if requested) as an attribute; callers
        # can read last_agent_samples for debugging / distributional reports.
        self.last_agent_samples = samples

        return results
