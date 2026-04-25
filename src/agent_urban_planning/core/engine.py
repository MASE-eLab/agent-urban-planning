"""Simulation engine orchestrating the full simulation loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from agent_urban_planning.core.agents import AgentPopulation
from agent_urban_planning.data.loaders import (
    AgentDistributionalConfig,
    PolicyConfig,
    ScenarioConfig,
    load_agents,
    load_policy,
    load_scenario,
)
from agent_urban_planning.decisions.base import DecisionEngine
from agent_urban_planning.decisions.clustering import (
    ClusteringConfig,
    ClusterizedDecisionEngine,
)
from agent_urban_planning.decisions._legacy_singapore_utility import UtilityEngine
from agent_urban_planning.core.environment import Environment
from agent_urban_planning.core.market import AhlfeldtMarket, HousingMarket, MarketResult
from agent_urban_planning.core.metrics import WelfareMetrics, compute_metrics
from agent_urban_planning.core.results import AgentResult, SimulationResults
from agent_urban_planning.core.run_metadata import RunMetadata, WallClock


class SimulationEngine:
    """Orchestrates the full simulation: config → environment → decisions → market → metrics."""

    def __init__(
        self,
        scenario: ScenarioConfig,
        agent_config: AgentDistributionalConfig,
        engine: Optional[DecisionEngine] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
        clustering: Optional[ClusteringConfig] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_concurrency: Optional[int] = None,
        price_elasticity: Optional[float] = None,
        initial_damping: Optional[float] = None,
        market_convergence_threshold: Optional[float] = None,
        max_market_iterations: Optional[int] = None,
    ):
        self.scenario = scenario
        self.agent_config = agent_config
        self.seed = seed or scenario.simulation.random_seed
        self.verbose = verbose

        # Provider/model info for metadata
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_concurrency = llm_concurrency

        # Market clearing parameters
        self._price_elasticity = price_elasticity
        self._initial_damping = initial_damping
        self._market_convergence_threshold = market_convergence_threshold
        self._max_market_iterations = max_market_iterations

        # Build the inner engine, then optionally wrap with clustering.
        # For Ahlfeldt scenarios, default to AhlfeldtUtilityEngine when no
        # explicit engine is provided.
        if engine is None:
            if scenario.ahlfeldt_params is not None:
                from agent_urban_planning.decisions.ahlfeldt_utility import AhlfeldtUtilityEngine
                inner_engine = AhlfeldtUtilityEngine(
                    params=scenario.ahlfeldt_params,
                    seed=seed,
                    dtype=getattr(scenario.ahlfeldt_params, "dtype", "float64"),
                    deterministic=getattr(
                        scenario.ahlfeldt_params, "deterministic", False
                    ),
                )
            else:
                inner_engine = UtilityEngine()
        else:
            inner_engine = engine
        self.inner_engine = inner_engine
        self.clustering_config = clustering

        if self.verbose:
            print(f"Loading scenario: {scenario.name} ({len(scenario.zones)} zones)")
            print(f"Generating {agent_config.num_types} agent types (seed={self.seed})...")

        rng = np.random.RandomState(self.seed) if self.seed is not None else None
        self.rng = rng
        # strict=True for real Singapore scenarios (>5 zones, no ahlfeldt_params).
        # Ahlfeldt replication scenarios use homogeneous agents by design
        # (matching the closed-form continuum-of-identical-workers assumption)
        # and therefore do not require per-zone Census distributions.
        _strict = (
            len(scenario.zones) > 5
            and scenario.ahlfeldt_params is None
        )
        self.population = AgentPopulation.from_config(agent_config, rng=rng, strict=_strict)
        self.base_env = Environment.from_config(scenario)

        # Wrap with clustering if configured (and not "none")
        if clustering is not None and clustering.algo != "none":
            self.engine = ClusterizedDecisionEngine(
                inner=inner_engine,
                config=clustering,
                environment=self.base_env,
                rng=rng,
            )
        else:
            self.engine = inner_engine

        if self.verbose:
            print(f"  Zones: {', '.join(self.base_env.zone_names)}")
            print(f"  Agents: {len(self.population)} types")
            if clustering is not None and clustering.algo != "none":
                print(
                    f"  Clustering: {clustering.algo} k={clustering.k} "
                    f"samples={clustering.samples_per_archetype} "
                    f"({clustering.within_cluster_assignment})"
                )
            print()

    @classmethod
    def from_paths(
        cls,
        scenario_path: str,
        agents_path: str,
        engine: Optional[DecisionEngine] = None,
        seed: Optional[int] = None,
    ) -> "SimulationEngine":
        return cls(
            scenario=load_scenario(scenario_path),
            agent_config=load_agents(agents_path),
            engine=engine,
            seed=seed,
        )

    def run(
        self,
        policy: Optional[PolicyConfig] = None,
        baseline: Optional[SimulationResults] = None,
        market_resume: Optional[dict] = None,
        market_checkpoint_callback=None,
        llm_cache_path: Optional[str] = None,
    ) -> SimulationResults:
        """Run a simulation for a given policy and return full results.

        When ``policy`` is ``None``, the scenario's built-in environment is
        used unchanged — no transit or facility investments are applied.
        This supports (a) Ahlfeldt replication runs, which have no notion
        of government investment, and (b) baseline observational runs that
        report the pre-intervention equilibrium.
        """
        if self.verbose:
            if policy is None:
                print(f"Running scenario: {self.scenario.name} (no policy applied)")
            else:
                n_transit = len(policy.transit_investments)
                n_facility = len(policy.facility_investments)
                print(f"Running policy: {policy.name}")
                print(f"  Investments: {n_transit} transit, {n_facility} facilities")
            print(f"  Market clearing...")

        wall_clock = WallClock()
        wall_clock.__enter__()

        # 1. Apply policy to environment (skip when policy is None)
        env = self.base_env.apply_policy(policy) if policy is not None else self.base_env

        # 2. Run market clearing. Dispatch on scenario.ahlfeldt_params:
        #    - Berlin / Ahlfeldt scenarios → AhlfeldtMarket (two-market)
        #    - Everything else → HousingMarket (single-segment or HDB/private)
        if self.scenario.ahlfeldt_params is not None:
            # External override (e.g. OpenCityAhlfeldtMarket for Run 2 CF).
            # Scripts can attach an already-constructed market instance to
            # `engine._market_override`; when present, we use it unchanged.
            if getattr(self, "_market_override", None) is not None:
                market = self._market_override
            else:
                market = AhlfeldtMarket(
                    ahlfeldt_params=self.scenario.ahlfeldt_params,
                    initial_damping=self._initial_damping or 0.3,
                    convergence_threshold=(
                        self._market_convergence_threshold
                        if self._market_convergence_threshold is not None
                        else self.scenario.simulation.market_convergence_threshold
                    ),
                    max_iterations=(
                        self._max_market_iterations
                        or self.scenario.simulation.market_max_iterations
                    ),
                    verbose=self.verbose,
                )
        else:
            # Resolve price elasticity: CLI override > engine property > default
            eta = self._price_elasticity
            if eta is None and hasattr(self.inner_engine, 'price_elasticity'):
                eta = abs(self.inner_engine.price_elasticity)
            if eta is None:
                eta = 0.5

            market = HousingMarket(
                max_iterations=self._max_market_iterations or self.scenario.simulation.market_max_iterations,
                convergence_threshold=(
                    self._market_convergence_threshold
                    if self._market_convergence_threshold is not None
                    else self.scenario.simulation.market_convergence_threshold
                ),
                price_elasticity=eta,
                initial_damping=self._initial_damping or 0.3,
                verbose=self.verbose,
            )
        market_result = market.clear(
            self.population,
            env,
            self.engine,
            resume_state=market_resume,
            checkpoint_callback=market_checkpoint_callback,
            cache_path=llm_cache_path,
        )

        # 3. Build per-agent results
        agent_results = []
        for agent in self.population:
            choice = market_result.allocations[agent.agent_id]
            route = env.transport.get_best_route(choice.zone_name, agent.job_location)
            commute = route.time_minutes if route else 120.0
            price = market_result.prices.get(choice.zone_name, 0.0)

            # Compute utility change vs baseline
            utility_vs_baseline = None
            if baseline:
                try:
                    base_result = baseline.get_agent(agent.agent_id)
                    utility_vs_baseline = choice.utility - base_result.realized_utility
                except KeyError:
                    pass

            agent_results.append(AgentResult.from_agent(
                agent=agent,
                zone_utilities=choice.zone_utilities,
                zone_choice=choice.zone_name,
                equilibrium_price=price,
                commute_minutes=commute,
                realized_utility=choice.utility,
                utility_vs_baseline=utility_vs_baseline,
                workplace_zone=getattr(choice, "workplace", "") or agent.job_location,
            ))

        # 4. Compute aggregate metrics
        if self.verbose:
            print(f"  Computing welfare metrics...")

        metrics = compute_metrics(
            self.population,
            env,
            market_result.allocations,
            market_result.prices,
            market_converged=market_result.converged,
            market_convergence_metric=market_result.convergence_metric,
        )

        if self.verbose:
            pops = ", ".join(f"{z}: {p:.1%}" for z, p in sorted(metrics.zone_populations.items()))
            print(f"  Results:")
            print(f"    Avg utility:    {metrics.avg_utility:.4f}")
            print(f"    Gini:           {metrics.gini_coefficient:.4f}")
            print(f"    Min utility:    {metrics.min_utility:.4f}")
            print(f"    Avg commute:    {metrics.avg_commute_minutes:.1f} min")
            print(f"    Unaffordable:   {metrics.housing_unaffordable_share:.1%}")
            print(f"    Zone pops:      {pops}")
            print()

        # Convert market history to serializable dicts. A/B / wages only
        # populated when the scenario is Ahlfeldt + endogenous flag.
        price_history = []
        for snap in market_result.history:
            entry = {
                "iteration": snap.iteration,
                "prices": snap.prices,
                "demand": snap.demand,
                "excess_demand": snap.excess_demand,
                "excess_demand_by_zone": snap.excess_demand_by_zone,
            }
            if getattr(snap, "productivity_A", None):
                entry["productivity_A"] = snap.productivity_A
                entry["amenity_B"] = snap.amenity_B
                entry["max_delta_A_rel"] = snap.max_delta_A_rel
                entry["max_delta_B_rel"] = snap.max_delta_B_rel
            if getattr(snap, "wages", None):
                entry["wages"] = snap.wages
            price_history.append(entry)

        # Stop the clock and assemble run metadata
        wall_clock.__exit__(None, None, None)
        metadata = self._build_metadata(policy, market_result, wall_clock.elapsed)

        return SimulationResults(
            metrics=metrics,
            agent_results=agent_results,
            policy_name=policy.name if policy is not None else "",
            scenario_name=self.scenario.name,
            price_history=price_history,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Metadata construction
    # ------------------------------------------------------------------

    def _build_metadata(
        self,
        policy: Optional[PolicyConfig],
        market_result: MarketResult,
        wall_clock_seconds: float,
    ) -> RunMetadata:
        """Assemble RunMetadata from market result and engine state."""
        clustering_algo = "none"
        num_archetypes = None
        samples_per_archetype = 1
        within_cluster_assignment = "deterministic"
        cluster_features: list[str] = []
        cluster_assignments: Optional[dict[int, int]] = None

        if self.clustering_config is not None:
            clustering_algo = self.clustering_config.algo
            if clustering_algo != "none":
                num_archetypes = self.clustering_config.k
                samples_per_archetype = self.clustering_config.samples_per_archetype
                within_cluster_assignment = self.clustering_config.within_cluster_assignment
                cluster_features = list(self.clustering_config.features or [])
                if isinstance(self.engine, ClusterizedDecisionEngine):
                    cluster_assignments = (
                        dict(self.engine.cluster_assignments)
                        if self.engine.cluster_assignments is not None
                        else None
                    )

        engine_name = self.inner_engine.__class__.__name__

        # Market clearing info from the result
        price_elasticity_used = getattr(market_result, 'price_elasticity_used', None)
        damping_final = getattr(market_result, 'damping_final', None)

        meta = RunMetadata(
            scenario_name=self.scenario.name,
            policy_name=policy.name if policy is not None else "",
            seed=self.seed,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            llm_temperature=self.llm_temperature,
            llm_concurrency=self.llm_concurrency,
            total_llm_calls=int(market_result.cache_misses),
            cached_llm_calls=int(market_result.cache_hits),
            total_input_tokens=int(market_result.total_input_tokens),
            total_output_tokens=int(market_result.total_output_tokens),
            wall_clock_seconds=wall_clock_seconds,
            clustering_algo=clustering_algo,
            num_archetypes=num_archetypes,
            samples_per_archetype=samples_per_archetype,
            within_cluster_assignment=within_cluster_assignment,
            cluster_features=cluster_features,
            cluster_assignments=cluster_assignments,
            decision_engine_name=engine_name,
            price_elasticity_used=price_elasticity_used,
            damping_final=damping_final,
            market_iterations_actual=market_result.iterations,
            convergence_achieved=market_result.converged,
        )
        meta.update_cache_hit_rate()
        meta.update_cost()
        return meta

    def compare_policies(
        self,
        policies: list[PolicyConfig],
    ) -> dict[str, SimulationResults]:
        """Run simulation for multiple policies and return results keyed by policy name.

        The first policy is used as the baseline for utility_vs_baseline computation.
        """
        results = {}
        baseline = None
        for policy in policies:
            result = self.run(policy, baseline=baseline)
            results[policy.name] = result
            if baseline is None:
                baseline = result
        return results

    def budget_sweep(
        self,
        base_policy: PolicyConfig,
        transit_shares: list[float],
    ) -> list[tuple[float, SimulationResults]]:
        """Sweep over transit budget share and return results for each.

        At each share level:
        - Transit investments: included if share > 0, with travel time improvement
          scaled proportionally (e.g., 50% budget = halfway between old and new time).
        - Facility investments: included if share < 1, with capacity and quality
          scaled proportionally.
        """
        from copy import deepcopy

        sweep_results = []
        budget = base_policy.total_budget
        base_transit_cost = sum(t.cost for t in base_policy.transit_investments) or 1.0
        base_facility_cost = sum(f.cost for f in base_policy.facility_investments) or 1.0

        # Get baseline travel times from current environment for interpolation
        base_times = {}
        for t in base_policy.transit_investments:
            route = self.base_env.transport.get_best_route(t.route[0], t.route[1])
            old_time = route.time_minutes if route else 120.0
            base_times[tuple(t.route)] = old_time

        for share in transit_shares:
            policy = deepcopy(base_policy)
            policy.name = f"sweep_{share:.2f}"

            transit_budget = budget * share
            facility_budget = budget * (1.0 - share)

            # Scale transit: interpolate travel time based on budget fraction
            if share > 0 and base_policy.transit_investments:
                transit_scale = min(transit_budget / base_transit_cost, 1.0)
                for t in policy.transit_investments:
                    old_time = base_times.get(tuple(t.route), 120.0)
                    full_new_time = t.new_time_minutes
                    # Interpolate: more budget = closer to full improvement
                    t.new_time_minutes = old_time - transit_scale * (old_time - full_new_time)
                    t.cost = t.cost * (transit_budget / base_transit_cost)
            else:
                policy.transit_investments = []

            # Scale facilities: scale capacity and quality with budget
            if (1.0 - share) > 0 and base_policy.facility_investments:
                facility_scale = min(facility_budget / base_facility_cost, 1.0)
                for f in policy.facility_investments:
                    f.capacity = max(1, int(f.capacity * facility_scale))
                    f.quality = f.quality * (0.5 + 0.5 * facility_scale)  # floor at 50%
                    f.cost = f.cost * (facility_budget / base_facility_cost)
            else:
                policy.facility_investments = []

            result = self.run(policy)
            sweep_results.append((share, result))

        return sweep_results
