"""Housing market clearing mechanism.

Implements an elasticity-based Walrasian tatonnement with adaptive damping,
stall detection, two-segment (HDB + private) market clearing, and dynamic
budget constraint enforcement.

The price update rule:

    Δp_z = (λ / |η|) × ((D_z - S_z) / S_z) × p_z

where η is the price elasticity of housing demand (Phang & Wong 1997),
λ is an adaptive damping factor (Scarf 1973, Shoven & Whalley 1992),
and the step is proportional to the current price level.

References:
  - Phang & Wong (1997): Singapore HDB demand elasticity η ≈ -0.5
  - Scarf (1973): Computation of economic equilibria
  - Shoven & Whalley (1992): Applying general equilibrium
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from agent_urban_planning.core.agents import Agent, AgentPopulation
from agent_urban_planning.core.constraints import is_hdb_eligible
from agent_urban_planning.decisions.base import DecisionEngine, LocationChoice, ZoneChoice
from agent_urban_planning.llm.cache import DiskBackedLLMCallCache, LLMCallCache
from agent_urban_planning.core.environment import Environment
from agent_urban_planning.core.labor_market import LaborMarket


@dataclass
class MarketSnapshot:
    """State of the market at one iteration."""
    iteration: int
    prices: dict[str, float]
    demand: dict[str, float]  # population weight per zone
    excess_demand: float      # max absolute excess demand
    excess_demand_by_zone: dict[str, float] = field(default_factory=dict)
    # Berlin / Ahlfeldt extensions. Empty for Singapore snapshots.
    wages: dict[str, float] = field(default_factory=dict)
    labor_demand: dict[str, float] = field(default_factory=dict)
    labor_supply: dict[str, float] = field(default_factory=dict)
    labor_excess_by_zone: dict[str, float] = field(default_factory=dict)
    max_labor_excess: float = 0.0
    # Endogenous-agglomeration extensions. Empty dicts / 0.0 when the flag is off.
    productivity_A: dict[str, float] = field(default_factory=dict)
    amenity_B: dict[str, float] = field(default_factory=dict)
    max_delta_A_rel: float = 0.0
    max_delta_B_rel: float = 0.0
    # Endogenous-land-use extensions. Empty dicts when flag is off.
    theta_diagnostic: dict[str, float] = field(default_factory=dict)
    arbitrage_gap_by_zone: dict[str, float] = field(default_factory=dict)


@dataclass
class MarketResult:
    """Result of housing market clearing."""
    prices: dict[str, float]            # equilibrium price per zone (Q for Ahlfeldt)
    allocations: dict[int, ZoneChoice]  # agent_id → zone choice at equilibrium
    convergence_metric: float           # max excess demand at termination
    iterations: int                     # number of iterations run
    converged: bool                     # whether convergence threshold was met
    history: list[MarketSnapshot] = field(default_factory=list)  # per-iteration snapshots
    cache_hits: int = 0                 # LLM cache hits during this clearing
    cache_misses: int = 0               # LLM cache misses during this clearing
    total_input_tokens: int = 0         # cumulative input tokens (LLM only)
    total_output_tokens: int = 0        # cumulative output tokens (LLM only)
    damping_final: float = 0.0          # final damping factor λ
    price_elasticity_used: float = 0.5  # η used for step sizing
    # Berlin / Ahlfeldt extensions. Empty / zero for Singapore runs.
    wages: dict[str, float] = field(default_factory=dict)
    commercial_price_diagnostic: dict[str, float] = field(default_factory=dict)
    damping_final_wage: float = 0.0
    converged_floor: bool = True        # aligned with ``converged`` for Singapore
    converged_labor: bool = True        # always True for Singapore (no labor market)
    eta_wage_used: float = 0.0
    # Endogenous-agglomeration extensions. Empty dicts / True for Singapore.
    productivity_A: dict[str, float] = field(default_factory=dict)
    amenity_B: dict[str, float] = field(default_factory=dict)
    converged_agglomeration: bool = True
    agglomeration_trajectory: list = field(default_factory=list)
    # Endogenous-land-use extensions. Empty / 0.0 when flag is off.
    theta_diagnostic: dict[str, float] = field(default_factory=dict)
    arbitrage_gap_by_zone: dict[str, float] = field(default_factory=dict)
    max_arbitrage_gap: float = 0.0
    theta_trajectory: list = field(default_factory=list)


def _serialize_choice(choice: ZoneChoice) -> dict:
    return {
        "zone_name": choice.zone_name,
        "utility": choice.utility,
        "zone_utilities": choice.zone_utilities,
    }


def _deserialize_choice(payload: dict) -> ZoneChoice:
    return ZoneChoice(
        zone_name=payload["zone_name"],
        utility=float(payload["utility"]),
        zone_utilities={
            str(zone): float(value)
            for zone, value in payload.get("zone_utilities", {}).items()
        },
    )


def _serialize_allocations(allocations: dict[int, ZoneChoice]) -> dict[str, dict]:
    return {str(agent_id): _serialize_choice(choice) for agent_id, choice in allocations.items()}


def _deserialize_allocations(payload: Optional[dict]) -> dict[int, ZoneChoice]:
    if not payload:
        return {}
    return {
        int(agent_id): _deserialize_choice(choice_payload)
        for agent_id, choice_payload in payload.items()
    }


def _serialize_history(history: list[MarketSnapshot]) -> list[dict]:
    return [
        {
            "iteration": snap.iteration,
            "prices": snap.prices,
            "demand": snap.demand,
            "excess_demand": snap.excess_demand,
            "excess_demand_by_zone": snap.excess_demand_by_zone,
        }
        for snap in history
    ]


def _deserialize_history(payload: Optional[list[dict]]) -> list[MarketSnapshot]:
    if not payload:
        return []
    return [
        MarketSnapshot(
            iteration=int(snap["iteration"]),
            prices={str(zone): float(price) for zone, price in snap.get("prices", {}).items()},
            demand={str(zone): float(demand) for zone, demand in snap.get("demand", {}).items()},
            excess_demand=float(snap.get("excess_demand", 0.0)),
            excess_demand_by_zone={
                str(zone): float(value)
                for zone, value in snap.get("excess_demand_by_zone", {}).items()
            },
        )
        for snap in payload
    ]


def _counter_delta(current: int, start: int) -> int:
    return max(0, int(current) - int(start))


class HousingMarket:
    """Elasticity-based tatonnement with adaptive damping and two-segment clearing.

    Each zone has two housing segments:
      - HDB (public): supply from ``zone.housing_supply``, price adjusted by tatonnement
      - Private: supply from ``zone.private_supply``, price held exogenous

    Agents with income > HDB_INCOME_CEILING are routed to the private segment;
    all others compete for HDB units. Convergence is evaluated on the HDB
    segment only.

    Budget constraints are enforced dynamically: at each iteration, each agent's
    choice set is filtered by affordability at current prices.
    """

    def __init__(
        self,
        price_elasticity: float = 0.5,
        initial_damping: float = 0.3,
        convergence_threshold: float = 0.01,
        stall_threshold: float = 1e-6,
        stall_window: int = 10,
        max_iterations: int = 1000,
        max_price_change_pct: float = 0.5,
        verbose: bool = False,
    ):
        self.price_elasticity = abs(price_elasticity) or 0.5
        self.initial_damping = initial_damping
        self.convergence_threshold = convergence_threshold
        self.stall_threshold = stall_threshold
        self.stall_window = stall_window
        self.max_iterations = max_iterations
        self.max_price_change_pct = max_price_change_pct
        self.verbose = verbose

    def clear(
        self,
        population: AgentPopulation,
        environment: Environment,
        engine: DecisionEngine,
        resume_state: Optional[dict] = None,
        checkpoint_callback: Optional[Callable[[dict], None]] = None,
        cache_path: Optional[str] = None,
    ) -> MarketResult:
        """Run tatonnement to find equilibrium prices and allocations."""
        zone_names = environment.zone_names
        agents_list = list(population)
        start_input_tokens = int(getattr(engine, "total_input_tokens", 0) or 0)
        start_output_tokens = int(getattr(engine, "total_output_tokens", 0) or 0)
        resumed_input_tokens = int((resume_state or {}).get("total_input_tokens", 0) or 0)
        resumed_output_tokens = int((resume_state or {}).get("total_output_tokens", 0) or 0)
        resumed_cache_hits = int((resume_state or {}).get("cache_hits", 0) or 0)
        resumed_cache_misses = int((resume_state or {}).get("cache_misses", 0) or 0)

        # --- Supply fractions ---
        # HDB segment
        hdb_supplies = {
            name: environment.get_zone(name).housing_supply
            for name in zone_names
        }
        total_hdb_supply = sum(hdb_supplies.values()) or 1

        # Supply fractions (for excess demand computation)
        hdb_supply_frac = {
            z: hdb_supplies[z] / total_hdb_supply for z in zone_names
        }

        # --- Initialize prices ---
        hdb_prices = {
            name: environment.get_zone(name).housing_base_price
            for name in zone_names
        }
        private_prices = {
            name: environment.get_zone(name).private_base_price
            for name in zone_names
        }

        # Per-clearing LLM cache (optionally restored from disk for crash recovery).
        if cache_path:
            cache = DiskBackedLLMCallCache(base_prices=hdb_prices, path=cache_path)
        else:
            cache = LLMCallCache(base_prices=hdb_prices)
        if hasattr(engine, "set_cache"):
            try:
                engine.set_cache(cache)
            except Exception:
                pass

        # --- Route agents to segments ---
        hdb_agents = []
        private_agents = []
        for agent in agents_list:
            if is_hdb_eligible(agent.income):
                hdb_agents.append(agent)
            else:
                private_agents.append(agent)

        show_decision_progress = (
            self.verbose and engine.__class__.__name__.lower().startswith("llm")
        )
        use_batch = hasattr(engine, "decide_batch") and callable(engine.decide_batch)

        def current_input_tokens() -> int:
            return resumed_input_tokens + _counter_delta(
                getattr(engine, "total_input_tokens", 0) or 0,
                start_input_tokens,
            )

        def current_output_tokens() -> int:
            return resumed_output_tokens + _counter_delta(
                getattr(engine, "total_output_tokens", 0) or 0,
                start_output_tokens,
            )

        def current_cache_hits() -> int:
            return resumed_cache_hits + cache.hits

        def current_cache_misses() -> int:
            return resumed_cache_misses + cache.misses

        def emit_checkpoint(stage: str, completed_hdb_iterations: int) -> None:
            if checkpoint_callback is None:
                return
            cache.flush()
            checkpoint_callback(
                {
                    "version": 1,
                    "stage": stage,
                    "completed_hdb_iterations": completed_hdb_iterations,
                    "hdb_prices": dict(hdb_prices),
                    "private_allocations": _serialize_allocations(private_allocations),
                    "lambda": lambda_,
                    "excess_history": list(excess_history),
                    "stall_count": stall_count,
                    "stall_boosted": stall_boosted,
                    "best_max_excess": (
                        None if best_max_excess == float("inf") else best_max_excess
                    ),
                    "best_prices": dict(best_prices),
                    "best_allocations": _serialize_allocations(best_allocations),
                    "history": _serialize_history(history),
                    "cache_hits": current_cache_hits(),
                    "cache_misses": current_cache_misses(),
                    "total_input_tokens": current_input_tokens(),
                    "total_output_tokens": current_output_tokens(),
                }
            )

        resume_stage = (resume_state or {}).get("stage")
        if resume_stage in ("private_segment_complete", "hdb_iterations"):
            hdb_prices = {
                name: float((resume_state or {}).get("hdb_prices", {}).get(name, hdb_prices[name]))
                for name in zone_names
            }
            private_allocations = _deserialize_allocations(
                (resume_state or {}).get("private_allocations")
            )
            lambda_ = float((resume_state or {}).get("lambda", self.initial_damping))
            excess_history = [
                float(value) for value in (resume_state or {}).get("excess_history", [])
            ]
            stall_count = int((resume_state or {}).get("stall_count", 0) or 0)
            stall_boosted = bool((resume_state or {}).get("stall_boosted", False))
            best_value = (resume_state or {}).get("best_max_excess")
            best_max_excess = float(best_value) if best_value is not None else float("inf")
            best_prices = {
                name: float(((resume_state or {}).get("best_prices") or hdb_prices).get(name, hdb_prices[name]))
                for name in zone_names
            }
            best_allocations = _deserialize_allocations(
                (resume_state or {}).get("best_allocations")
            )
            history = _deserialize_history((resume_state or {}).get("history"))
            start_iteration = int((resume_state or {}).get("completed_hdb_iterations", 0) or 0)
            if private_agents and not private_allocations:
                raise ValueError(
                    "Resume state is missing private-segment allocations for "
                    f"{len(private_agents)} private agents"
                )
        else:
            # Private agents: fixed allocation (no tatonnement on private prices)
            # They choose once based on private prices (exogenous)
            private_allocations = self._allocate_private_segment(
                private_agents, environment, zone_names, private_prices, engine,
            )
            lambda_ = self.initial_damping
            excess_history = []
            stall_count = 0
            stall_boosted = False
            best_max_excess = float("inf")
            best_prices = dict(hdb_prices)
            best_allocations = {}
            history = []
            start_iteration = 0
            emit_checkpoint("private_segment_complete", 0)

        # Private demand (fixed throughout clearing)
        private_demand = {name: 0.0 for name in zone_names}
        for agent in private_agents:
            if agent.agent_id in private_allocations:
                choice = private_allocations[agent.agent_id]
                private_demand[choice.zone_name] += agent.weight

        # Total HDB-eligible weight (for demand fraction normalization)
        total_hdb_weight = sum(a.weight for a in hdb_agents) or 1.0

        for iteration in range(start_iteration, self.max_iterations):
            # --- Phase 1: Budget constraint filtering + agent decisions ---
            # Build the price dict agents will see (HDB prices for HDB agents)
            current_prices = dict(hdb_prices)
            allocations: dict[int, ZoneChoice] = dict(private_allocations)
            hdb_demand: dict[str, float] = {name: 0.0 for name in zone_names}

            if show_decision_progress:
                print(
                    f"    Iteration {iteration + 1}/{self.max_iterations}: "
                    f"evaluating {len(hdb_agents)} HDB agent decisions (λ={lambda_:.3f})...",
                    flush=True,
                )

            if use_batch:
                choices = engine.decide_batch(
                    hdb_agents, environment, zone_names, current_prices,
                )
                for agent, choice in zip(hdb_agents, choices):
                    allocations[agent.agent_id] = choice
                    hdb_demand[choice.zone_name] += agent.weight
            else:
                for agent in hdb_agents:
                    choice = engine.decide(agent, environment, zone_names, current_prices)
                    allocations[agent.agent_id] = choice
                    hdb_demand[choice.zone_name] += agent.weight

            # --- Phase 2: Compute excess demand (HDB segment only) ---
            excess_by_zone: dict[str, float] = {}
            max_excess = 0.0

            for name in zone_names:
                supply_frac = hdb_supply_frac[name]
                if supply_frac <= 0:
                    excess_by_zone[name] = 0.0
                    continue
                # Demand fraction among HDB agents
                demand_frac = hdb_demand[name] / total_hdb_weight if total_hdb_weight > 0 else 0.0
                excess = demand_frac - supply_frac
                excess_by_zone[name] = excess
                if abs(excess) > max_excess:
                    max_excess = abs(excess)

            excess_history.append(max_excess)

            # --- Phase 3: Record snapshot ---
            # Combine HDB + private demand for the snapshot
            total_demand = {
                z: hdb_demand[z] + private_demand.get(z, 0.0) for z in zone_names
            }
            history.append(MarketSnapshot(
                iteration=iteration + 1,
                prices=dict(hdb_prices),
                demand=total_demand,
                excess_demand=max_excess,
                excess_demand_by_zone=dict(excess_by_zone),
            ))

            # Track best solution
            if max_excess < best_max_excess:
                best_max_excess = max_excess
                best_prices = dict(hdb_prices)
                best_allocations = dict(allocations)

            # Progress output
            if self.verbose and (
                iteration % 50 == 0
                or iteration == self.max_iterations - 1
                or max_excess < self.convergence_threshold
            ):
                top_zones = sorted(
                    excess_by_zone.items(), key=lambda x: abs(x[1]), reverse=True
                )[:3]
                excess_str = ", ".join(f"{z}: {e:+.4f}" for z, e in top_zones)
                print(
                    f"    Iter {iteration + 1}/{self.max_iterations}  "
                    f"max |excess|: {max_excess:.4f}  λ={lambda_:.3f}  "
                    f"top excess: [{excess_str}]",
                    flush=True,
                )

            # --- Phase 4: Check convergence ---
            if max_excess < self.convergence_threshold:
                if self.verbose:
                    print(
                        f"    Converged at iteration {iteration + 1} "
                        f"(max |excess|: {max_excess:.4f})",
                        flush=True,
                    )
                return self._build_result(
                    prices=hdb_prices,
                    allocations=allocations,
                    convergence_metric=max_excess,
                    iterations=iteration + 1,
                    converged=True,
                    history=history,
                    cache_hits=current_cache_hits(),
                    cache_misses=current_cache_misses(),
                    total_input_tokens=current_input_tokens(),
                    total_output_tokens=current_output_tokens(),
                    damping_final=lambda_,
                )

            # --- Phase 5: Adaptive damping ---
            if len(excess_history) >= 3:
                recent = excess_history[-3:]
                if recent[2] < recent[1] < recent[0]:
                    # Monotonic decrease → accelerate
                    lambda_ = min(lambda_ * 1.1, 0.8)
                elif (recent[2] - recent[1]) * (recent[1] - recent[0]) < 0:
                    # Oscillation (sign change in deltas) → brake
                    lambda_ = max(lambda_ * 0.7, 0.05)

            # --- Phase 6: Stall detection ---
            if len(excess_history) >= 2:
                delta = abs(excess_history[-1] - excess_history[-2])
                if delta < self.stall_threshold:
                    stall_count += 1
                else:
                    stall_count = 0
                    stall_boosted = False

                if stall_count >= self.stall_window:
                    if not stall_boosted:
                        # First stall: boost λ by 50%
                        lambda_ = min(lambda_ * 1.5, 0.8)
                        stall_boosted = True
                        stall_count = 0
                        if self.verbose:
                            print(
                                f"    Stall detected at iter {iteration + 1}, "
                                f"boosting λ to {lambda_:.3f}",
                                flush=True,
                            )
                    else:
                        # Second stall after boost: terminate
                        if self.verbose:
                            print(
                                f"    Persistent stall at iter {iteration + 1}, "
                                f"terminating (best excess: {best_max_excess:.4f})",
                                flush=True,
                            )
                        return self._build_result(
                            prices=best_prices,
                            allocations=best_allocations,
                            convergence_metric=best_max_excess,
                            iterations=iteration + 1,
                            converged=False,
                            history=history,
                            cache_hits=current_cache_hits(),
                            cache_misses=current_cache_misses(),
                            total_input_tokens=current_input_tokens(),
                            total_output_tokens=current_output_tokens(),
                            damping_final=lambda_,
                        )

            # --- Phase 7: Elasticity-based price update (HDB only) ---
            # Cap Δp at ±max_price_change_pct of current price to prevent
            # single-iteration spikes in small-supply zones where discrete
            # agent switching creates step-function demand jumps.
            for name in zone_names:
                supply_frac = hdb_supply_frac[name]
                if supply_frac <= 0:
                    continue
                excess = excess_by_zone[name]
                # Δp = (λ / |η|) × (excess / supply_frac) × p
                delta_p = (
                    (lambda_ / self.price_elasticity)
                    * (excess / supply_frac)
                    * hdb_prices[name]
                )
                # Clamp to ±50% of current price per iteration
                max_delta = self.max_price_change_pct * hdb_prices[name]
                delta_p = max(-max_delta, min(delta_p, max_delta))
                hdb_prices[name] = max(hdb_prices[name] + delta_p, 1.0)

            emit_checkpoint("hdb_iterations", iteration + 1)

        # Did not converge — return best snapshot
        if self.verbose:
            print(
                f"    Did not converge after {self.max_iterations} iterations "
                f"(best excess demand: {best_max_excess:.4f})",
                flush=True,
            )
        return self._build_result(
            prices=best_prices,
            allocations=best_allocations,
            convergence_metric=best_max_excess,
            iterations=self.max_iterations,
            converged=False,
            history=history,
            cache_hits=current_cache_hits(),
            cache_misses=current_cache_misses(),
            total_input_tokens=current_input_tokens(),
            total_output_tokens=current_output_tokens(),
            damping_final=lambda_,
        )

    def _allocate_private_segment(
        self,
        agents: list[Agent],
        environment: Environment,
        zone_names: list[str],
        private_prices: dict[str, float],
        engine: DecisionEngine,
    ) -> dict[int, ZoneChoice]:
        """Allocate private-segment agents once (prices are exogenous).

        Uses the same engine but with private prices. Budget constraint
        is handled by the engine internally.
        """
        if not agents:
            return {}

        allocations: dict[int, ZoneChoice] = {}
        use_batch = hasattr(engine, "decide_batch") and callable(engine.decide_batch)

        if use_batch:
            choices = engine.decide_batch(agents, environment, zone_names, private_prices)
            for agent, choice in zip(agents, choices):
                allocations[agent.agent_id] = choice
        else:
            for agent in agents:
                choice = engine.decide(agent, environment, zone_names, private_prices)
                allocations[agent.agent_id] = choice

        return allocations

    def _build_result(
        self,
        prices: dict[str, float],
        allocations: dict[int, ZoneChoice],
        convergence_metric: float,
        iterations: int,
        converged: bool,
        history: list[MarketSnapshot],
        cache_hits: int,
        cache_misses: int,
        total_input_tokens: int,
        total_output_tokens: int,
        damping_final: float,
    ) -> MarketResult:
        # Report HDB prices as the policy-relevant equilibrium prices.
        merged_prices = dict(prices)
        return MarketResult(
            prices=merged_prices,
            allocations=allocations,
            convergence_metric=convergence_metric,
            iterations=iterations,
            converged=converged,
            history=history,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            damping_final=damping_final,
            price_elasticity_used=self.price_elasticity,
        )


class AhlfeldtMarket(HousingMarket):
    """Two-market tatonnement clearing Q (residential floor) AND w (wages).

    For Berlin / Ahlfeldt scenarios only. Selected by ``SimulationEngine``
    when ``scenario.ahlfeldt_params is not None``. Supersedes the
    HDB/private segmentation in ``HousingMarket.clear``; Berlin has one
    unified floor market per zone with the commercial / residential split
    fixed from observed 2006 values (Decision: fixed θ_i).

    Per-market elasticities are computed from the Ahlfeldt parameters:
        eta_floor = (1 - beta) * epsilon
        eta_wage  = 1 / (1 - alpha) + epsilon
    and can be overridden via scenario-level ``eta_floor_override`` /
    ``eta_wage_override`` fields.
    """

    def __init__(
        self,
        ahlfeldt_params,
        initial_damping: float = 0.3,
        convergence_threshold: float = 0.01,
        stall_threshold: float = 1e-6,
        stall_window: int = 10,
        max_iterations: int = 1000,
        max_price_change_pct: float = 0.5,
        max_price_change_pct_wage: float | None = None,
        verbose: bool = False,
    ):
        # Derive per-market elasticities from Ahlfeldt structural parameters.
        alpha = ahlfeldt_params.alpha
        beta = ahlfeldt_params.beta
        epsilon = ahlfeldt_params.epsilon
        default_eta_floor = (1.0 - beta) * epsilon
        default_eta_wage = 1.0 / (1.0 - alpha) + epsilon

        eta_floor = (
            float(ahlfeldt_params.eta_floor_override)
            if ahlfeldt_params.eta_floor_override is not None
            else default_eta_floor
        )
        eta_wage = (
            float(ahlfeldt_params.eta_wage_override)
            if ahlfeldt_params.eta_wage_override is not None
            else default_eta_wage
        )

        # Parent init carries shared knobs (stall detection, iteration cap,
        # price cap) — its ``price_elasticity`` is repurposed here as the
        # floor-market elasticity eta_floor.
        super().__init__(
            price_elasticity=eta_floor,
            initial_damping=initial_damping,
            convergence_threshold=convergence_threshold,
            stall_threshold=stall_threshold,
            stall_window=stall_window,
            max_iterations=max_iterations,
            max_price_change_pct=max_price_change_pct,
            verbose=verbose,
        )
        self.params = ahlfeldt_params
        self.eta_floor = eta_floor
        self.eta_wage = eta_wage
        self.max_price_change_pct_wage = (
            float(max_price_change_pct_wage)
            if max_price_change_pct_wage is not None
            else max_price_change_pct
        )

        # Endogenous-agglomeration state. Kernels lazily precomputed the
        # first time clear() runs (we need the environment's tt matrix,
        # which isn't available here at __init__). agglomeration_damping is
        # read from the scenario params.
        self.endogenous_agglomeration: bool = bool(
            getattr(ahlfeldt_params, "endogenous_agglomeration", False)
        )
        self.agglomeration_damping: float = float(
            getattr(ahlfeldt_params, "agglomeration_damping", 0.5)
        )
        if not (0.0 < self.agglomeration_damping <= 1.0):
            raise ValueError(
                f"agglomeration_damping must be in (0, 1]; got "
                f"{self.agglomeration_damping}"
            )
        self.K_prod: Optional[np.ndarray] = None
        self.K_amen: Optional[np.ndarray] = None

        # Endogenous-land-use state. When True, the floor market collapses
        # Q and q into a single P_i that clears residential + commercial
        # demand against total floor supply H_i. Default eta_floor bumps
        # from (1-β)·ε ≈ 1.67 to 2.0 to reflect the blended elasticity of
        # combined res + com demand (Decision 4 in design.md).
        self.endogenous_land_use: bool = bool(
            getattr(ahlfeldt_params, "endogenous_land_use", False)
        )
        # Clearing method: "foc_direct" uses pack's closed-form FOC inversion
        # (Q = ((1-α)Y + (1-β)vv)/L), converges in ~10 iterations and anchors
        # the absolute price level. "tatonnement" is the legacy share-based
        # elasticity update. FOC is default for endogenous_land_use scenarios.
        self.clearing_method: str = str(
            getattr(ahlfeldt_params, "clearing_method", None)
            or ("foc_direct" if self.endogenous_land_use else "tatonnement")
        )
        if self.clearing_method not in ("foc_direct", "tatonnement"):
            raise ValueError(
                f"clearing_method must be 'foc_direct' or 'tatonnement'; got "
                f"{self.clearing_method!r}"
            )
        if self.endogenous_land_use and ahlfeldt_params.eta_floor_override is None:
            # Override the parent-set eta_floor with the combined-market default.
            self.eta_floor = 2.0
            self.price_elasticity = 2.0

        # Opt-in per-iteration progress hook. Default is None so existing
        # callers (V1/V2/V3) are unaffected. The hook fires once per
        # tatonnement iteration after residuals are computed with args
        # (iter_idx, max_floor_excess, max_labor_excess, elapsed_seconds).
        self._iteration_callback: Optional[Callable[[int, float, float, float], None]] = None

    def set_iteration_callback(
        self,
        fn: Optional[Callable[[int, float, float, float], None]],
    ) -> None:
        """Install or clear a per-iteration progress callback.

        Called after every tatonnement iteration with
        ``fn(iter_idx, max_floor_excess, max_labor_excess, elapsed_seconds)``.
        Exceptions raised by the callback are suppressed so progress
        rendering cannot break market clearing.
        """
        self._iteration_callback = fn

    # ------------------------------------------------------------------
    # Endogenous-agglomeration helpers
    # ------------------------------------------------------------------
    def _ensure_spillover_kernels(self, environment: Environment, zone_names: list[str]) -> None:
        """Precompute K_prod = exp(-delta·tt) and K_amen = exp(-rho·tt) once.

        Called lazily on first clear() because __init__ doesn't receive the
        environment. Idempotent: subsequent calls are no-ops if kernels
        already have the right shape.
        """
        if not self.endogenous_agglomeration:
            return
        N = len(zone_names)
        if self.K_prod is not None and self.K_prod.shape == (N, N):
            return
        # Validate that the scenario has at least SOME non-zero fundamentals.
        # At block scale the pack's data includes ~2000-3000 peripheral blocks
        # with zero activity (no workplace employment and/or no residents); those
        # legitimately have a_i = b_i = 0. We tolerate these by clamping at
        # kernel-product time (the agglomeration loop's np.maximum(1e-12)).
        # We only fail if EVERY zone is zero (indicates a regeneration bug).
        total_zones = len(zone_names)
        zero_a = sum(
            1 for z in zone_names
            if float(environment.get_zone(z).productivity_fundamental_a) == 0.0
        )
        zero_b = sum(
            1 for z in zone_names
            if float(environment.get_zone(z).amenity_fundamental_b) == 0.0
        )
        if zero_a == total_zones or zero_b == total_zones:
            from agent_urban_planning.data.loaders import ConfigError
            raise ConfigError(
                f"All {total_zones} zones have zero productivity_fundamental_a "
                f"({zero_a} zones) or amenity_fundamental_b ({zero_b} zones). "
                f"endogenous_agglomeration requires a_i, b_i primitives from "
                f"Ahlfeldt 2015. Regenerate the scenario YAML with "
                f"scripts/build_berlin_scenario_yaml.py."
            )
        # Build the travel-time matrix in zone-order
        if environment.transport_matrix is not None and environment.transport_matrix_index:
            idx = [environment._matrix_index_map[z] for z in zone_names]
            tt = environment.transport_matrix[np.ix_(idx, idx)].astype(
                np.float64, copy=False
            )
        else:
            tt = np.zeros((N, N), dtype=np.float64)
            for i, zi in enumerate(zone_names):
                for j, zj in enumerate(zone_names):
                    tt[i, j] = environment.travel_time(zi, zj)
        delta = float(self.params.delta)
        rho = float(self.params.rho)
        dtype_str = getattr(self.params, "dtype", "float64")
        kernel_dtype = np.float32 if dtype_str == "float32" else np.float64
        self.K_prod = np.exp(-delta * tt).astype(kernel_dtype, copy=False)
        self.K_amen = np.exp(-rho * tt).astype(kernel_dtype, copy=False)
        # Precision sanity: the PEAK kernel value (at τ near 0) must be
        # positive — that confirms we haven't catastrophically lost all
        # resolution. Individual long-range entries can legitimately
        # underflow to zero under float32 at large τ·δ (e.g. pubtt06
        # values >100 min with rho=0.76 give exp(-76) ≈ 1e-33, below
        # float32 normal range). Zero kernel values are semantically
        # "no spillover at this distance" and the agglomeration update
        # clamps the aggregate Υ, Ω to 1e-12 downstream.
        assert float(self.K_prod.max()) > 0, "K_prod is all zeros after dtype cast"
        assert float(self.K_amen.max()) > 0, "K_amen is all zeros after dtype cast"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def clear(
        self,
        population: AgentPopulation,
        environment: Environment,
        engine: DecisionEngine,
        resume_state: Optional[dict] = None,  # noqa: ARG002 — not supported for Ahlfeldt
        checkpoint_callback: Optional[Callable[[dict], None]] = None,  # noqa: ARG002
        cache_path: Optional[str] = None,
    ) -> MarketResult:
        """Run joint Q/w tatonnement and return an Ahlfeldt ``MarketResult``.

        ``resume_state`` and ``checkpoint_callback`` are accepted for
        protocol compatibility with :class:`HousingMarket` but not used
        — Ahlfeldt runs are typically short enough (≤200 iterations at
        Ortsteile resolution) that in-process execution is sufficient.
        """
        zone_names = environment.zone_names
        agents_list = list(population)
        N = len(zone_names)
        total_weight = sum(a.weight for a in agents_list) or 1.0
        _clear_start_time = _time.monotonic()

        # ---- Endogenous-agglomeration setup (lazy on first clear) -------
        self._ensure_spillover_kernels(environment, zone_names)

        # ---- Supply shares (fixed-θ path) -----------------------------
        residential_supply = {
            z: float(environment.get_zone(z).residential_floor_area) for z in zone_names
        }
        total_residential_supply = sum(residential_supply.values()) or 1.0
        residential_share = {
            z: residential_supply[z] / total_residential_supply for z in zone_names
        }

        # ---- Total floor supply (endogenous-land-use path) ------------
        # Under endogenous_land_use, the scenario YAML carries per-zone
        # total_floor_area H_i (or synthesized from residential + commercial).
        # The single-price market clears combined demand against H_i.
        H_supply = np.array(
            [float(environment.get_zone(z).total_floor_area) for z in zone_names],
            dtype=np.float64,
        )
        # Normalized to shares for the elasticity-based update.
        total_H = float(H_supply.sum()) or 1.0
        H_share_vec = H_supply / total_H

        # ---- Initialize prices -----------------------------------------
        Q = {
            z: float(environment.get_zone(z).floor_price_observed)
            or float(environment.get_zone(z).housing_base_price)
            or 1.0
            for z in zone_names
        }
        wages = {
            z: float(environment.get_zone(z).wage_observed) or 1.0 for z in zone_names
        }
        # Geometric-mean target for Q-level anchoring. Share-based tatonnement
        # drives spatial pattern toward equilibrium but leaves the absolute
        # level unpinned (any uniform rescaling preserves shares). After each
        # Q update, rescale Q so its geometric mean matches this target,
        # preserving the observed absolute price level.
        _Q_init_vec = np.array(list(Q.values()), dtype=np.float64)
        _log_Q_init_mean = float(np.mean(np.log(np.maximum(_Q_init_vec, 1e-12))))

        # ---- Build LaborMarket helper ----------------------------------
        A_map = {z: float(environment.get_zone(z).productivity_A) for z in zone_names}
        L_map = {z: float(environment.get_zone(z).commercial_floor_area) for z in zone_names}
        labor_market = LaborMarket(
            alpha=self.params.alpha,
            A=A_map,
            L=L_map,
            wage_observed=wages,
            max_price_change_pct=self.max_price_change_pct_wage,
        )

        # Observed employment shares (targets for labor market).
        # When observed empwpl is absent we fall back to equal shares.
        empwpl_obs = np.array(
            [float(environment.get_zone(z).job_density) for z in zone_names],
            dtype=np.float64,
        )
        if empwpl_obs.sum() <= 0.0:
            empwpl_obs = np.ones(N, dtype=np.float64) / N
        empwpl_share = empwpl_obs / empwpl_obs.sum()

        # ---- Cache plumbing (for LLM-wrapped engines) -------------------
        if cache_path:
            cache = DiskBackedLLMCallCache(base_prices=Q, path=cache_path)
        else:
            cache = LLMCallCache(base_prices=Q)
        if hasattr(engine, "set_cache"):
            try:
                engine.set_cache(cache)
            except Exception:
                pass

        # Wage-injection hook (supported by AhlfeldtUtilityEngine)
        supports_wages = hasattr(engine, "set_current_wages") and callable(
            engine.set_current_wages
        )
        supports_agglomeration = (
            self.endogenous_agglomeration
            and hasattr(engine, "set_current_productivity")
            and hasattr(engine, "set_current_amenity")
        )

        # ---- Endogenous-agglomeration state ----------------------------
        # Warm-start A, B from the pack's post-agglomeration equilibrium
        # values (stored as Zone.productivity_A / amenity_B). Raw
        # fundamentals a, b are the zone primitives that stay fixed.
        A_prev = np.array(
            [float(environment.get_zone(z).productivity_A) for z in zone_names],
            dtype=np.float64,
        )
        B_prev = np.array(
            [float(environment.get_zone(z).amenity_B) for z in zone_names],
            dtype=np.float64,
        )
        a_fund = np.array(
            [float(environment.get_zone(z).productivity_fundamental_a) for z in zone_names],
            dtype=np.float64,
        )
        b_fund = np.array(
            [float(environment.get_zone(z).amenity_fundamental_b) for z in zone_names],
            dtype=np.float64,
        )
        K_area = np.array(
            [
                float(environment.get_zone(z).commercial_floor_area)
                + float(environment.get_zone(z).residential_floor_area)
                or 1.0
                for z in zone_names
            ],
            dtype=np.float64,
        )
        delta_A_history: list[float] = []
        delta_B_history: list[float] = []
        agglomeration_trajectory: list[tuple[dict[str, float], dict[str, float]]] = []
        theta_trajectory: list[dict[str, float]] = []

        # Inject warm-start A, B on iteration 0 so the engine uses them
        # from the first decide_batch.
        if supports_agglomeration:
            engine.set_current_productivity(
                {z: float(A_prev[i]) for i, z in enumerate(zone_names)}
            )
            engine.set_current_amenity(
                {z: float(B_prev[i]) for i, z in enumerate(zone_names)}
            )

        # ---- Independent damping state machines ------------------------
        lambda_floor = self.initial_damping
        lambda_wage = self.initial_damping
        floor_history: list[float] = []
        wage_history: list[float] = []

        best_joint_excess = float("inf")
        best_Q = dict(Q)
        best_w = dict(wages)
        best_allocations: dict[int, ZoneChoice] = {}
        history: list[MarketSnapshot] = []

        converged_floor = False
        converged_labor = False
        converged_agglomeration = not self.endogenous_agglomeration  # vacuously True when off
        iteration = 0

        for iteration in range(self.max_iterations):
            # ---- Phase 1: inject wages, decide ---------------------------
            if supports_wages:
                engine.set_current_wages(wages)

            choices: list[LocationChoice]
            if hasattr(engine, "decide_batch") and callable(engine.decide_batch):
                choices = engine.decide_batch(agents_list, environment, zone_names, Q)
            else:
                choices = [engine.decide(a, environment, zone_names, Q) for a in agents_list]

            allocations: dict[int, ZoneChoice] = {}
            residential_demand_weight = {z: 0.0 for z in zone_names}
            labor_supply_weight = {z: 0.0 for z in zone_names}
            # residence_income: income FLOWING TO residence zone i from
            # workplaces j, weighted by wage_j × P[i,j]. Pack's vv_i.
            # This is the correct quantity for housing-expenditure aggregation
            # (D_R = (1-β)·vv/Q), not the agent-mass HR used previously.
            residence_income = {z: 0.0 for z in zone_names}
            # If the engine ran in deterministic mode, it exposes the full
            # P_ij choice-probability matrix. Under that mode every agent's
            # LocationChoice is nominal (modal); the ACTUAL demand must be
            # aggregated from P_ij so we preserve the continuum semantics
            # (zero Monte Carlo noise). Otherwise aggregate from per-agent
            # discrete choices as before.
            P = getattr(engine, "last_choice_probabilities", None)
            if P is not None:
                # Continuum aggregation: residential demand share at zone i
                # = Σ_j P[i, j]; labor supply share at zone j = Σ_i P[i, j].
                # Both are multiplied by the total agent weight.
                total_agent_weight = sum(a.weight for a in agents_list) or 1.0
                P_arr = np.asarray(P, dtype=np.float64)
                residence_shares = P_arr.sum(axis=1)
                workplace_shares = P_arr.sum(axis=0)
                # vv_i = Σ_j (wage_j × P[i,j]) × total_mass
                wage_arr = np.array(
                    [wages[z] for z in zone_names], dtype=np.float64
                )
                residence_income_vec = (P_arr @ wage_arr) * total_agent_weight
                for a, ch in zip(agents_list, choices):
                    allocations[a.agent_id] = ch
                for i_z, zname in enumerate(zone_names):
                    residential_demand_weight[zname] = float(
                        residence_shares[i_z] * total_agent_weight
                    )
                    labor_supply_weight[zname] = float(
                        workplace_shares[i_z] * total_agent_weight
                    )
                    residence_income[zname] = float(residence_income_vec[i_z])
            else:
                for agent, choice in zip(agents_list, choices):
                    allocations[agent.agent_id] = choice
                    residential_demand_weight[choice.residence] += agent.weight
                    labor_supply_weight[choice.workplace] += agent.weight
                    # vv_i: wage at agent's workplace × weight
                    residence_income[choice.residence] += (
                        float(wages.get(choice.workplace, 0.0)) * agent.weight
                    )

            # ---- Phase 2: floor excess (demand vs supply SHARES) -------
            # Labor quantities first (both paths need H_M_j for phase 3).
            w_vec = labor_market.to_array(wages)
            firm_demand = labor_market.compute_demand(w_vec)
            firm_demand_share = firm_demand / max(firm_demand.sum(), 1e-12)
            labor_supply_share = np.array(
                [labor_supply_weight[z] / total_weight for z in zone_names],
                dtype=np.float64,
            )

            theta_diagnostic: dict[str, float] = {}
            arbitrage_gap_by_zone: dict[str, float] = {}

            if self.endogenous_land_use:
                # Unified floor price clears combined demand against H_i.
                # Residential demand: D_R_i = (1-β)·vv_i/P_i where vv_i is
                # income flowing to residence zone i = Σ_j wage_j · P[i,j] · N.
                # Wages are heterogeneous across workplaces, so vv_i ≠ HR_i
                # (simple mass). Pack uses vv for housing-expenditure.
                beta = self.params.beta
                P_vec = np.array(
                    [float(Q[z]) for z in zone_names], dtype=np.float64
                )
                HR_vec = np.array(
                    [residential_demand_weight[z] for z in zone_names],
                    dtype=np.float64,
                )
                vv_vec = np.array(
                    [residence_income[z] for z in zone_names],
                    dtype=np.float64,
                )
                H_M_vec = np.array(
                    [labor_supply_weight[z] for z in zone_names],
                    dtype=np.float64,
                )
                # Residential demand: D_R_i = (1-β) · vv_i / Q_i
                D_R = (1.0 - beta) * vv_vec / np.maximum(P_vec, 1e-12)
                # Commercial demand via firm FOC (uses current A from
                # endogenous-agglomeration injection if active).
                A_vec_live = (
                    A_prev
                    if self.endogenous_agglomeration
                    else np.array(
                        [float(environment.get_zone(z).productivity_A) for z in zone_names],
                        dtype=np.float64,
                    )
                )
                L_com = labor_market.compute_commercial_floor_demand(
                    w_vec, P_vec, A_vec_live, H_M_vec
                )
                # Share-based excess drives the spatial pattern toward
                # equilibrium, but shares have infinitely many fixed points
                # differing only in the overall Q level. The absolute level
                # is anchored below via a geometric-mean renormalization.
                total_demand = D_R + L_com
                demand_share = total_demand / max(float(total_demand.sum()), 1e-12)
                supply_share = H_share_vec  # precomputed outside the loop
                excess_P = demand_share - supply_share
                max_floor_excess = float(np.max(np.abs(excess_P)))
                excess_floor_by_zone = {
                    z: float(excess_P[i]) for i, z in enumerate(zone_names)
                }
                # θ_i diagnostic: commercial share of total demand.
                total_demand = D_R + L_com
                theta_raw = L_com / np.maximum(total_demand, 1e-12)
                theta_diagnostic = {
                    z: float(np.clip(theta_raw[i], 0.0, 1.0))
                    for i, z in enumerate(zone_names)
                }
                # Arbitrage gap: |P_i - q_i^ZP| / q_i^ZP
                q_zp = labor_market.compute_commercial_price_diagnostic(w_vec)
                arbitrage_gap_vec = np.abs(P_vec - q_zp) / np.maximum(q_zp, 1e-12)
                arbitrage_gap_by_zone = {
                    z: float(arbitrage_gap_vec[i]) for i, z in enumerate(zone_names)
                }
            else:
                # Legacy fixed-θ path: residential demand share vs
                # residential supply share only.
                max_floor_excess = 0.0
                excess_floor_by_zone = {}
                for z in zone_names:
                    demand_share = residential_demand_weight[z] / total_weight
                    excess = demand_share - residential_share[z]
                    excess_floor_by_zone[z] = excess
                    if abs(excess) > max_floor_excess:
                        max_floor_excess = abs(excess)

            # ---- Phase 3: labor excess (firm demand vs supply) ---------
            labor_excess = firm_demand_share - labor_supply_share
            max_labor_excess = float(np.max(np.abs(labor_excess)))
            excess_labor_by_zone = {
                z: float(labor_excess[i]) for i, z in enumerate(zone_names)
            }

            # ---- Phase 3.5: endogenous-agglomeration update ------------
            # Recompute A_i = a_i * Υ_i^λ and B_i = b_i * Ω_i^η from current
            # density, then damped-blend with previous iteration. Skipped
            # entirely when the flag is off (kernels are None).
            max_delta_A_rel = 0.0
            max_delta_B_rel = 0.0
            if self.endogenous_agglomeration and self.K_prod is not None:
                HR_vec = np.array(
                    [residential_demand_weight[z] for z in zone_names],
                    dtype=np.float64,
                )
                HM_vec = np.array(
                    [labor_supply_weight[z] for z in zone_names],
                    dtype=np.float64,
                )
                density_res = HR_vec / K_area
                density_wpl = HM_vec / K_area
                # Spillover aggregates via kernel matmul
                Upsilon = self.K_prod @ density_wpl
                Omega = self.K_amen @ density_res
                # Zero-density guard
                Upsilon = np.maximum(Upsilon, 1e-12)
                Omega = np.maximum(Omega, 1e-12)
                # Zero-fundamental guard (peripheral blocks with a_i = b_i = 0
                # — allowed at block scale; clamp so Gumbel-softmax stays finite).
                a_safe = np.maximum(a_fund, 1e-12)
                b_safe = np.maximum(b_fund, 1e-12)
                A_new = a_safe * np.power(Upsilon, self.params.lambda_)
                B_new = b_safe * np.power(Omega, self.params.eta)
                # Damped blend: A = (1-d)·A_prev + d·A_new
                d = self.agglomeration_damping
                A_updated = (1.0 - d) * A_prev + d * A_new
                B_updated = (1.0 - d) * B_prev + d * B_new
                # Per-iteration max relative change
                max_delta_A_rel = float(
                    np.max(np.abs(A_updated - A_prev) / np.maximum(np.abs(A_prev), 1e-12))
                )
                max_delta_B_rel = float(
                    np.max(np.abs(B_updated - B_prev) / np.maximum(np.abs(B_prev), 1e-12))
                )
                # Inject into engine for the next iteration
                if supports_agglomeration:
                    A_dict = {z: float(A_updated[i]) for i, z in enumerate(zone_names)}
                    B_dict = {z: float(B_updated[i]) for i, z in enumerate(zone_names)}
                    engine.set_current_productivity(A_dict)
                    engine.set_current_amenity(B_dict)
                    agglomeration_trajectory.append((A_dict, B_dict))
                # Update for next iteration
                A_prev = A_updated
                B_prev = B_updated
                delta_A_history.append(max_delta_A_rel)
                delta_B_history.append(max_delta_B_rel)

            # ---- Phase 4: record snapshot ------------------------------
            snapshot_A = (
                {z: float(A_prev[i]) for i, z in enumerate(zone_names)}
                if self.endogenous_agglomeration
                else {}
            )
            snapshot_B = (
                {z: float(B_prev[i]) for i, z in enumerate(zone_names)}
                if self.endogenous_agglomeration
                else {}
            )
            snapshot = MarketSnapshot(
                iteration=iteration + 1,
                prices=dict(Q),
                demand=dict(residential_demand_weight),
                excess_demand=max_floor_excess,
                excess_demand_by_zone=dict(excess_floor_by_zone),
                wages=dict(wages),
                labor_demand={
                    z: float(firm_demand[i]) for i, z in enumerate(zone_names)
                },
                labor_supply=dict(labor_supply_weight),
                labor_excess_by_zone=dict(excess_labor_by_zone),
                max_labor_excess=max_labor_excess,
                productivity_A=snapshot_A,
                amenity_B=snapshot_B,
                max_delta_A_rel=max_delta_A_rel,
                max_delta_B_rel=max_delta_B_rel,
                theta_diagnostic=dict(theta_diagnostic),
                arbitrage_gap_by_zone=dict(arbitrage_gap_by_zone),
            )
            history.append(snapshot)
            if self.endogenous_land_use:
                theta_trajectory.append(dict(theta_diagnostic))

            joint_max = max(max_floor_excess, max_labor_excess)
            floor_history.append(max_floor_excess)
            wage_history.append(max_labor_excess)

            if joint_max < best_joint_excess:
                best_joint_excess = joint_max
                best_Q = dict(Q)
                best_w = dict(wages)
                best_allocations = dict(allocations)

            # ---- Opt-in progress callback -------------------------------
            if self._iteration_callback is not None:
                try:
                    self._iteration_callback(
                        iteration + 1,
                        float(max_floor_excess),
                        float(max_labor_excess),
                        _time.monotonic() - _clear_start_time,
                    )
                except Exception:
                    pass  # progress rendering must not break clearing

            # ---- Phase 5: convergence check ----------------------------
            converged_floor = max_floor_excess < self.convergence_threshold
            converged_labor = max_labor_excess < self.convergence_threshold
            if self.endogenous_agglomeration:
                converged_agglomeration = (
                    max_delta_A_rel < self.convergence_threshold
                    and max_delta_B_rel < self.convergence_threshold
                )
            else:
                converged_agglomeration = True
            if converged_floor and converged_labor and converged_agglomeration:
                if self.verbose:
                    agg_str = (
                        f", ΔA={max_delta_A_rel:.4f}, ΔB={max_delta_B_rel:.4f}"
                        if self.endogenous_agglomeration
                        else ""
                    )
                    print(
                        f"    AhlfeldtMarket converged at iter {iteration + 1} "
                        f"(floor={max_floor_excess:.4f}, labor={max_labor_excess:.4f}{agg_str})",
                        flush=True,
                    )
                break

            if self.verbose and (iteration % 25 == 0 or iteration == self.max_iterations - 1):
                agg_str = (
                    f"  ΔA={max_delta_A_rel:.4f}  ΔB={max_delta_B_rel:.4f}"
                    if self.endogenous_agglomeration
                    else ""
                )
                print(
                    f"    Iter {iteration + 1}/{self.max_iterations}  "
                    f"floor |excess|: {max_floor_excess:.4f} (λ_f={lambda_floor:.3f})  "
                    f"labor |excess|: {max_labor_excess:.4f} (λ_w={lambda_wage:.3f})"
                    f"{agg_str}",
                    flush=True,
                )

            # ---- Phase 6: independent adaptive damping ------------------
            lambda_floor = self._adapt_damping(floor_history, lambda_floor)
            lambda_wage = self._adapt_damping(wage_history, lambda_wage)

            # ---- Phase 7: price updates --------------------------------
            if self.clearing_method == "foc_direct" and self.endogenous_land_use:
                # Pack-style closed-form FOC update. From Ahlfeldt et al.
                # (2015) solver:
                #   Y_j    = A_j · HM_j^α · (θ_j · L_j)^(1-α)       (firm output)
                #   w_j    = α · Y_j / HM_j                          (firm wage FOC)
                #   Q_i    = ((1-α) Y_i + (1-β) vv_i) / L_i           (floor FOC)
                #   θ_i    = (1-α) Y_i / (Q_i · L_i)                  (endogenous share)
                # Pack's Matlab solver uses 0.25·new + 0.75·old damping
                # across Q, q, θ, wage simultaneously. This conservative
                # step prevents spatial pattern drift during iteration.
                alpha = self.params.alpha
                beta = self.params.beta
                damping = 0.25
                theta_vec = np.array(
                    [float(theta_diagnostic.get(z, 0.21)) for z in zone_names],
                    dtype=np.float64,
                )
                Y_vec = (
                    A_vec_live
                    * np.power(np.maximum(H_M_vec, 1e-12), alpha)
                    * np.power(
                        np.maximum(theta_vec * H_supply, 1e-12),
                        1.0 - alpha,
                    )
                )
                # Q FOC (mixed form — naturally reduces to residence-only
                # when Y_i=0, or commercial-only when vv_i=0):
                Q_new_vec = (
                    (1.0 - alpha) * Y_vec + (1.0 - beta) * vv_vec
                ) / np.maximum(H_supply, 1e-12)
                Q_new_vec = np.maximum(Q_new_vec, 1e-6)
                # Damped update (pack-style, no anchor — agent mass must
                # match pack's scale for Q to converge to observed).
                Q_cur_vec = np.array([Q[z] for z in zone_names], dtype=np.float64)
                Q_updated = damping * Q_new_vec + (1.0 - damping) * Q_cur_vec
                Q_updated = np.maximum(Q_updated, 1e-6)
                for i_z, z in enumerate(zone_names):
                    Q[z] = float(Q_updated[i_z])

                # Wage FOC (pack-style damped, no anchor)
                w_new_vec = alpha * Y_vec / np.maximum(H_M_vec, 1e-12)
                w_new_vec = np.maximum(w_new_vec, 1e-6)
                w_cur_vec = labor_market.to_array(wages)
                w_updated = damping * w_new_vec + (1.0 - damping) * w_cur_vec
                w_updated = np.maximum(w_updated, 1e-6)
                wages = {z: float(w_updated[i]) for i, z in enumerate(zone_names)}
            else:
                # Legacy tatonnement: Δ(ln Q) = (λ / η) · excess_ratio (capped).
                for i_z, z in enumerate(zone_names):
                    if self.endogenous_land_use:
                        sshare = float(H_share_vec[i_z])
                        if sshare <= 0:
                            continue
                        ratio = excess_floor_by_zone[z] / sshare
                    else:
                        supply_share = residential_share[z]
                        if supply_share <= 0:
                            continue
                        ratio = excess_floor_by_zone[z] / supply_share
                    delta = (lambda_floor / max(self.eta_floor, 1e-9)) * ratio * Q[z]
                    cap = self.max_price_change_pct * Q[z]
                    delta = max(-cap, min(delta, cap))
                    Q[z] = max(Q[z] + delta, 1e-6)

                # Q-level anchor: rescale Q so its geometric mean matches the
                # initial observed geometric mean.
                if self.endogenous_land_use:
                    _Q_vec = np.array([Q[z] for z in zone_names], dtype=np.float64)
                    _log_Q_mean = float(np.mean(np.log(np.maximum(_Q_vec, 1e-12))))
                    _rescale = float(np.exp(_log_Q_init_mean - _log_Q_mean))
                    for z in zone_names:
                        Q[z] = max(Q[z] * _rescale, 1e-6)

                # Wage update via LaborMarket
                supply_vec = labor_market.to_array(labor_supply_weight) / total_weight
                lm_result = labor_market.update_wages(
                    w_vec, supply_vec, eta_wage=self.eta_wage, lambda_wage=lambda_wage
                )
                wages = lm_result.wages

            # ---- Post-iteration hook (for subclasses like OpenCityAhlfeldtMarket) --
            # Called AFTER Q and wages update. Subclasses can override to adjust
            # population, agglomeration, or any other state that needs to respond
            # to the new equilibrium prices. Default is a no-op so Run 1 baseline
            # behaviour is unchanged.
            self._post_iter_hook(
                iteration=iteration,
                Q=Q,
                wages=wages,
                zone_names=zone_names,
                environment=environment,
                agents_list=agents_list,
            )

        # ---- Finalize --------------------------------------------------
        final_floor_excess = history[-1].excess_demand if history else 0.0
        final_labor_excess = history[-1].max_labor_excess if history else 0.0
        joint_converged = converged_floor and converged_labor and converged_agglomeration

        # Return the FINAL state, not best-seen. Best-seen can pin Q to the
        # warm-start (iter 0) when that happens to have locally-small residuals,
        # hiding the tatonnement's actual price movement.
        Q_out = Q
        w_out = wages
        alloc_out = allocations
        metric = max(final_floor_excess, final_labor_excess)

        # Compute final commercial-price diagnostic from exit wages
        q_diag = labor_market.compute_commercial_price_diagnostic(
            labor_market.to_array(w_out)
        )
        commercial_price_diag = {
            z: float(q_diag[i]) for i, z in enumerate(zone_names)
        }

        # Final A, B for the result (only populated when endogenous)
        final_A = (
            {z: float(A_prev[i]) for i, z in enumerate(zone_names)}
            if self.endogenous_agglomeration
            else {}
        )
        final_B = (
            {z: float(B_prev[i]) for i, z in enumerate(zone_names)}
            if self.endogenous_agglomeration
            else {}
        )
        # Final θ + arbitrage gap (only populated under endogenous_land_use)
        if self.endogenous_land_use and history:
            final_theta = history[-1].theta_diagnostic
            final_arb = history[-1].arbitrage_gap_by_zone
            max_arb = max(final_arb.values()) if final_arb else 0.0
        else:
            final_theta = {}
            final_arb = {}
            max_arb = 0.0

        return MarketResult(
            prices=Q_out,
            allocations=alloc_out,
            convergence_metric=metric,
            iterations=iteration + 1,
            converged=joint_converged,
            history=history,
            cache_hits=cache.hits,
            cache_misses=cache.misses,
            total_input_tokens=int(getattr(engine, "total_input_tokens", 0) or 0),
            total_output_tokens=int(getattr(engine, "total_output_tokens", 0) or 0),
            damping_final=lambda_floor,
            price_elasticity_used=self.eta_floor,
            wages=w_out,
            commercial_price_diagnostic=commercial_price_diag,
            damping_final_wage=lambda_wage,
            converged_floor=converged_floor,
            converged_labor=converged_labor,
            eta_wage_used=self.eta_wage,
            productivity_A=final_A,
            amenity_B=final_B,
            converged_agglomeration=converged_agglomeration,
            agglomeration_trajectory=agglomeration_trajectory,
            theta_diagnostic=final_theta,
            arbitrage_gap_by_zone=final_arb,
            max_arbitrage_gap=max_arb,
            theta_trajectory=theta_trajectory,
        )

    # ------------------------------------------------------------------
    # Subclass hook: called at end of each iteration (after Q and wages
    # updates). Default no-op preserves baseline Run 1 behaviour exactly.
    # Subclasses like OpenCityAhlfeldtMarket override to adjust population.
    # ------------------------------------------------------------------
    def _post_iter_hook(self, **kwargs) -> None:
        pass

    # ------------------------------------------------------------------
    # Adaptive damping helper (per-market state)
    # ------------------------------------------------------------------
    @staticmethod
    def _adapt_damping(history: list[float], lam: float) -> float:
        """Apply accelerate/brake rules to ``lam`` based on the last 3
        observations of ``history``. Floor 0.05, cap 0.8."""
        if len(history) < 3:
            return lam
        recent = history[-3:]
        if recent[2] < recent[1] < recent[0]:
            return min(lam * 1.1, 0.8)
        if (recent[2] - recent[1]) * (recent[1] - recent[0]) < 0:
            return max(lam * 0.7, 0.05)
        return lam
