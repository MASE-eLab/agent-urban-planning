"""Monte Carlo runner for K-seed replicated simulations.

For Berlin replication (and any ABM with Fréchet-like idiosyncratic
shocks), a single seed gives a point estimate with finite-sample noise.
Running K independent seeds and reporting ``μ ± σ`` on every headline
metric is standard practice (e.g., Ahlfeldt et al. 2015 report seed-
aware confidence intervals in their replication supplement).

This module provides :class:`MonteCarloRunner`, which orchestrates K
independent :class:`SimulationEngine` runs with distinct seeds and
aggregates their results. Runs are parallelized across processes by
default (one process per seed, bounded by CPU count); a ``--sequential``
mode is available for debugging.

Usage from code:

    runner = MonteCarloRunner(scenario, agents)
    mc_results = runner.run(policy=None, k=5, base_seed=42)
    print(mc_results.mean_metrics.avg_utility, mc_results.std_metrics['avg_utility'])

Usage from CLI (via ``run_simulation.py --monte-carlo 5``): see
task 12.6 for wiring.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional

from agent_urban_planning.data.loaders import (
    AgentDistributionalConfig,
    PolicyConfig,
    ScenarioConfig,
)
from agent_urban_planning.core.engine import SimulationEngine
from agent_urban_planning.core.metrics import WelfareMetrics
from agent_urban_planning.core.results import SimulationResults


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloResults:
    """Aggregated results across K Monte Carlo replicates."""

    results: list[SimulationResults]
    mean_metrics: WelfareMetrics
    std_metrics: dict[str, float]
    converged_count: int
    k: int
    base_seed: int
    seeds_used: list[int] = field(default_factory=list)

    @property
    def converged_rate(self) -> float:
        return self.converged_count / max(self.k, 1)


# ---------------------------------------------------------------------------
# Worker (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------


def _run_one_replicate(args: tuple) -> SimulationResults:
    """Run one simulation replicate with the given seed.

    Must be a module-level function (not a lambda / closure) so
    ``ProcessPoolExecutor`` can pickle it.
    """
    (
        scenario,
        agent_config,
        policy,
        seed,
        engine_kwargs,
    ) = args
    engine = SimulationEngine(
        scenario=scenario,
        agent_config=agent_config,
        seed=seed,
        **(engine_kwargs or {}),
    )
    return engine.run(policy=policy)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _scalar_metric_keys(metrics: WelfareMetrics) -> list[str]:
    """Subset of WelfareMetrics fields that are simple floats/ints."""
    return [
        "avg_utility",
        "gini_coefficient",
        "min_utility",
        "max_utility",
        "avg_commute_minutes",
        "long_commute_share",
        "housing_unaffordable_share",
        "market_convergence_metric",
    ]


def _aggregate_scalar_metrics(
    results: list[SimulationResults],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute element-wise mean and stdev of scalar metrics across results."""
    keys = _scalar_metric_keys(results[0].metrics)
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in keys:
        values = [float(getattr(r.metrics, key)) for r in results]
        means[key] = sum(values) / len(values) if values else 0.0
        stds[key] = (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        )
    return means, stds


def _aggregate_zone_metrics(
    results: list[SimulationResults], field_name: str
) -> dict[str, float]:
    """Element-wise mean of a zone→float dict across results."""
    all_zones: set[str] = set()
    for r in results:
        all_zones.update(getattr(r.metrics, field_name, {}).keys())
    out: dict[str, float] = {}
    for zone in all_zones:
        values = [
            float(getattr(r.metrics, field_name, {}).get(zone, 0.0)) for r in results
        ]
        out[zone] = sum(values) / len(values) if values else 0.0
    return out


def build_mean_metrics(results: list[SimulationResults]) -> WelfareMetrics:
    """Construct a WelfareMetrics with element-wise means across results."""
    if not results:
        raise ValueError("Cannot build mean metrics from empty results")
    means, _ = _aggregate_scalar_metrics(results)
    return WelfareMetrics(
        avg_utility=means["avg_utility"],
        gini_coefficient=means["gini_coefficient"],
        min_utility=means["min_utility"],
        max_utility=means["max_utility"],
        avg_commute_minutes=means["avg_commute_minutes"],
        long_commute_share=means["long_commute_share"],
        housing_unaffordable_share=means["housing_unaffordable_share"],
        zone_populations=_aggregate_zone_metrics(results, "zone_populations"),
        zone_prices=_aggregate_zone_metrics(results, "zone_prices"),
        facility_utilization=list(results[0].metrics.facility_utilization),
        market_converged=all(r.metrics.market_converged for r in results),
        market_convergence_metric=means["market_convergence_metric"],
        zone_employment=_aggregate_zone_metrics(results, "zone_employment"),
        zone_wages=_aggregate_zone_metrics(results, "zone_wages"),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class MonteCarloRunner:
    """Orchestrates K independent simulation runs with distinct seeds."""

    def __init__(
        self,
        scenario: ScenarioConfig,
        agent_config: AgentDistributionalConfig,
        engine_kwargs: Optional[dict[str, Any]] = None,
        max_workers: Optional[int] = None,
    ):
        self.scenario = scenario
        self.agent_config = agent_config
        self.engine_kwargs = dict(engine_kwargs or {})
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)

    def run(
        self,
        policy: Optional[PolicyConfig] = None,
        k: int = 5,
        base_seed: int = 42,
        sequential: bool = False,
    ) -> MonteCarloResults:
        """Run K replicates and aggregate.

        Args:
            policy: Policy to apply (or None for policy-less runs —
                standard for Ahlfeldt replication).
            k: Number of replicates.
            base_seed: Seeds used will be ``base_seed, base_seed+1, …, base_seed+k-1``.
            sequential: If True, run replicates in-process (for debugging).
                Default False uses a process pool.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        seeds = [base_seed + i for i in range(k)]
        work = [
            (
                self.scenario,
                self.agent_config,
                policy,
                seed,
                self.engine_kwargs,
            )
            for seed in seeds
        ]

        results: list[SimulationResults]
        if sequential or k == 1:
            results = [_run_one_replicate(w) for w in work]
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(self.max_workers, k)
            ) as executor:
                results = list(executor.map(_run_one_replicate, work))

        mean_metrics = build_mean_metrics(results)
        _, std_metrics = _aggregate_scalar_metrics(results)
        converged = sum(1 for r in results if r.metrics.market_converged)

        return MonteCarloResults(
            results=results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            converged_count=converged,
            k=k,
            base_seed=base_seed,
            seeds_used=seeds,
        )
