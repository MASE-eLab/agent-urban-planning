"""YAML config loading and validation for scenarios, agents, and policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when a configuration file is invalid."""


# --- Data structures for parsed configs ---


@dataclass
class FacilityConfig:
    type: str
    capacity: int
    quality: float
    cost: float = 0.0  # only used in policy configs


@dataclass
class ZoneConfig:
    name: str
    housing_supply: int
    housing_base_price: float
    amenity_score: float
    facilities: list[FacilityConfig] = field(default_factory=list)
    job_density: float = 0.0
    private_supply: int = 0
    private_base_price: float = 0.0
    job_count: int = 0
    # Added in berlin-replication-abm: Ahlfeldt-style zone fundamentals.
    # All default to 0.0 so Singapore YAMLs remain fully backward-compatible.
    commercial_floor_area: float = 0.0
    residential_floor_area: float = 0.0
    productivity_A: float = 0.0
    amenity_B: float = 0.0
    wage_observed: float = 0.0
    floor_price_observed: float = 0.0
    # Added in endogenous-agglomeration: raw (pre-agglomeration) fundamentals.
    productivity_fundamental_a: float = 0.0
    amenity_fundamental_b: float = 0.0
    # Added in endogenous-land-use: total floor supply (optional; synthesized
    # from commercial + residential when 0.0).
    total_floor_area: float = 0.0


@dataclass
class AhlfeldtParams:
    """Structural parameters from Ahlfeldt et al. (2015) Section 7."""

    # 6 estimated structural parameters (pooled EThetaA)
    kappa_eps: float  # = kappa * epsilon, scaled commuting friction
    epsilon: float    # Fréchet shape / CES elasticity
    lambda_: float    # productivity agglomeration elasticity (trailing _ avoids keyword clash)
    delta: float      # productivity spillover spatial decay
    eta: float        # amenity agglomeration elasticity
    rho: float        # amenity spillover spatial decay
    # 2 fixed parameters (locked by model design; do NOT load as free)
    alpha: float = 0.80  # labor share
    beta: float = 0.75   # residential expenditure share
    # Optional tatonnement overrides
    eta_floor_override: float | None = None
    eta_wage_override: float | None = None
    # Added in endogenous-agglomeration. When True the market recomputes
    # productivity A and amenity B each iteration from per-zone density.
    endogenous_agglomeration: bool = False
    # Damping factor for the A, B update (Jacobi-style blend). Range (0, 1].
    agglomeration_damping: float = 0.5
    # Added in endogenous-land-use. When True the floor market collapses
    # Q and q into a single price P_i that clears combined residential +
    # commercial demand against total floor supply H_i.
    endogenous_land_use: bool = False
    # Added in block-level-replication. Hot-path N² matrix precision.
    dtype: str = "float64"
    # Added in block-level-replication. Anderson-m convergence acceleration.
    # 0 disables; 2-8 is typical sweet spot. See design.md Decision 9.
    anderson_m: int = 0
    # Added in block-level-replication. When True, the engine uses
    # expected-demand (continuum) mode instead of Fréchet sampling.
    deterministic: bool = False
    # Market clearing method. "foc_direct" uses pack's closed-form FOC
    # (Q = ((1-α)Y + (1-β)vv)/L), stable and fast. "tatonnement" is the
    # legacy share-based elasticity update. None falls back to
    # "foc_direct" when endogenous_land_use else "tatonnement".
    clearing_method: str | None = None

    @property
    def kappa(self) -> float:
        """Derived: pure commuting-cost decay per minute."""
        return self.kappa_eps / self.epsilon if self.epsilon > 0 else 0.0


@dataclass
class TransportRouteConfig:
    from_zone: str
    to_zone: str
    mode: str
    time_minutes: float
    cost_dollars: float


@dataclass
class SimulationParams:
    market_max_iterations: int = 1000
    market_convergence_threshold: float = 0.05
    random_seed: int | None = None


@dataclass
class ScenarioConfig:
    name: str
    zones: list[ZoneConfig]
    transport: list[TransportRouteConfig]
    simulation: SimulationParams
    description: str = ""
    # Added in berlin-replication-abm. When present, the simulator builds an
    # AhlfeldtMarket instead of a HousingMarket and uses AhlfeldtUtilityEngine.
    # Always None for Singapore scenarios.
    ahlfeldt_params: AhlfeldtParams | None = None
    # Optional path to an NPZ file containing an N×N travel-time matrix
    # (keyed by 'tt'). When present and ahlfeldt_params is set, this matrix
    # replaces the edge-list transport for commute-cost computation.
    transport_matrix_path: str | None = None


@dataclass
class DistributionConfig:
    type: str
    params: dict[str, Any]


@dataclass
class AgentDistributionalConfig:
    mode: str  # "distributional" or "explicit"
    num_types: int
    distributions: dict[str, DistributionConfig]
    preference_rules: dict[str, Any] | None = None  # DEPRECATED: unused by all engines. Kept for YAML backward compat only.
    explicit_agents: list[dict[str, Any]] | None = None
    zone_distributions: dict[str, dict] | None = None  # per-zone Census histograms


@dataclass
class TransitInvestment:
    route: list[str]  # [from_zone, to_zone]
    mode: str
    cost: float
    new_time_minutes: float
    new_cost_dollars: float = 0.0


@dataclass
class FacilityInvestment:
    zone: str
    type: str
    cost: float
    capacity: int
    quality: float


@dataclass
class PolicyConfig:
    name: str
    total_budget: float
    transit_investments: list[TransitInvestment] = field(default_factory=list)
    facility_investments: list[FacilityInvestment] = field(default_factory=list)
    description: str = ""
    budget_unit: str = ""


# --- Loading functions ---


def _load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a YAML mapping: {path}")
    return data


def _require(data: dict, key: str, path: str) -> Any:
    """Get a required key from a dict, raising ConfigError if missing."""
    if key not in data:
        raise ConfigError(f"Missing required field '{key}' in {path}")
    return data[key]


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load and validate a scenario configuration file."""
    data = _load_yaml(path)

    # Parse zones
    zones_data = _require(data, "zones", str(path))
    if not isinstance(zones_data, dict) or len(zones_data) == 0:
        raise ConfigError(f"'zones' must be a non-empty mapping in {path}")

    zones = []
    for name, z in zones_data.items():
        facilities = []
        for f in z.get("facilities", []):
            facilities.append(FacilityConfig(
                type=f["type"],
                capacity=f["capacity"],
                quality=f["quality"],
            ))
        zones.append(ZoneConfig(
            name=name,
            housing_supply=_require(z, "housing_supply", f"zones.{name}"),
            housing_base_price=_require(z, "housing_base_price", f"zones.{name}"),
            amenity_score=_require(z, "amenity_score", f"zones.{name}"),
            facilities=facilities,
            job_density=z.get("job_density", 0.0),
            private_supply=z.get("private_supply", 0),
            private_base_price=z.get("private_base_price", 0.0),
            job_count=z.get("job_count", 0),
            commercial_floor_area=z.get("commercial_floor_area", 0.0),
            residential_floor_area=z.get("residential_floor_area", 0.0),
            productivity_A=z.get("productivity_A", 0.0),
            amenity_B=z.get("amenity_B", 0.0),
            wage_observed=z.get("wage_observed", 0.0),
            floor_price_observed=z.get("floor_price_observed", 0.0),
            productivity_fundamental_a=z.get("productivity_fundamental_a", 0.0),
            amenity_fundamental_b=z.get("amenity_fundamental_b", 0.0),
            total_floor_area=z.get("total_floor_area", 0.0),
        ))

    # Parse transport. For Ahlfeldt scenarios the transport edge list may be
    # empty; the travel-time matrix is loaded from transport_matrix_path.
    transport_data = data.get("transport", [])
    transport = []
    for t in transport_data:
        transport.append(TransportRouteConfig(
            from_zone=t["from"],
            to_zone=t["to"],
            mode=t["mode"],
            time_minutes=t["time_minutes"],
            cost_dollars=t["cost_dollars"],
        ))

    # Parse simulation params
    sim_data = data.get("simulation", {})
    simulation = SimulationParams(
        market_max_iterations=sim_data.get("market_max_iterations", 200),
        market_convergence_threshold=sim_data.get("market_convergence_threshold", 0.01),
        random_seed=sim_data.get("random_seed"),
    )

    # Parse optional ahlfeldt_params block (Berlin scenarios)
    ahlfeldt_params = None
    if "ahlfeldt_params" in data:
        ap = data["ahlfeldt_params"]
        required_ap = ("kappa_eps", "epsilon", "lambda", "delta", "eta", "rho")
        missing = [k for k in required_ap if k not in ap]
        if missing:
            raise ConfigError(
                f"ahlfeldt_params in {path} missing required fields: {missing}"
            )
        ahlfeldt_params = AhlfeldtParams(
            kappa_eps=float(ap["kappa_eps"]),
            epsilon=float(ap["epsilon"]),
            lambda_=float(ap["lambda"]),
            delta=float(ap["delta"]),
            eta=float(ap["eta"]),
            rho=float(ap["rho"]),
            alpha=float(ap.get("alpha", 0.80)),
            beta=float(ap.get("beta", 0.75)),
            eta_floor_override=ap.get("eta_floor_override"),
            eta_wage_override=ap.get("eta_wage_override"),
            endogenous_agglomeration=bool(ap.get("endogenous_agglomeration", False)),
            agglomeration_damping=float(ap.get("agglomeration_damping", 0.5)),
            endogenous_land_use=bool(ap.get("endogenous_land_use", False)),
            dtype=str(ap.get("dtype", "float64")),
            anderson_m=int(ap.get("anderson_m", 0)),
            deterministic=bool(ap.get("deterministic", False)),
            clearing_method=ap.get("clearing_method"),
        )
        if ahlfeldt_params.dtype not in ("float32", "float64"):
            raise ConfigError(
                f"ahlfeldt_params.dtype must be 'float32' or 'float64'; "
                f"got '{ahlfeldt_params.dtype}' in {path}"
            )
        if not (0 <= ahlfeldt_params.anderson_m <= 10):
            raise ConfigError(
                f"ahlfeldt_params.anderson_m must be in [0, 10]; "
                f"got {ahlfeldt_params.anderson_m} in {path}"
            )

    scenario = ScenarioConfig(
        name=data.get("name", Path(path).stem),
        description=data.get("description", ""),
        zones=zones,
        transport=transport,
        simulation=simulation,
        ahlfeldt_params=ahlfeldt_params,
        transport_matrix_path=data.get("transport_matrix_path"),
    )

    # Skip Singapore-specific data-quality checks for Ahlfeldt scenarios —
    # they use completely different fields (floor_price_observed instead of
    # housing_base_price as the primary price signal).
    if ahlfeldt_params is not None:
        return scenario

    # REJECT configs with missing real-data fields (signs of old guessed data).
    # Small test scenarios (≤5 zones) are exempt — they're for unit tests.
    if len(zones) > 5:
        zones_missing_private = [z.name for z in zones if z.private_supply == 0 and z.housing_supply > 0]
        zones_missing_jobcount = [z.name for z in zones if z.job_count == 0 and z.job_density > 0]

        if len(zones_missing_private) > len(zones) * 0.5:
            raise ConfigError(
                f"Scenario '{scenario.name}': {len(zones_missing_private)}/{len(zones)} "
                f"zones have no private_supply data. This scenario uses old guessed "
                f"data without private housing — not acceptable. "
                f"Use config/scenarios/singapore_real_v2.yaml (real Census 2020 data) "
                f"or run the data fetcher to regenerate."
            )
        if len(zones_missing_jobcount) > len(zones) * 0.5:
            raise ConfigError(
                f"Scenario '{scenario.name}': {len(zones_missing_jobcount)}/{len(zones)} "
                f"zones have no job_count data. Employment data is missing — not "
                f"acceptable. Use config/scenarios/singapore_real_v2.yaml (real Census "
                f"2020 data) or run the data fetcher to regenerate."
            )

        # Check for suspiciously round HDB prices (sign of old hardcoded values)
        prices = [z.housing_base_price for z in zones]
        round_count = sum(1 for p in prices if p > 0 and p % 100 == 0)
        if round_count == len(prices):
            raise ConfigError(
                f"Scenario '{scenario.name}': ALL {len(prices)} HDB prices are round "
                f"hundreds (e.g., S${prices[0]:.0f}). These are old guessed values, "
                f"not real API data. Use config/scenarios/singapore_real_v2.yaml "
                f"(real HDB resale transaction prices) or run the data fetcher."
            )

    return scenario


def load_agents(path: str | Path) -> AgentDistributionalConfig:
    """Load and validate an agent demographics configuration file."""
    data = _load_yaml(path)

    mode = _require(data, "mode", str(path))
    if mode not in ("distributional", "explicit"):
        raise ConfigError(f"Agent mode must be 'distributional' or 'explicit', got '{mode}'")

    if mode == "distributional":
        num_types = _require(data, "num_types", str(path))
        dists_data = _require(data, "distributions", str(path))
        distributions = {}
        for name, d in dists_data.items():
            dist_type = _require(d, "type", f"distributions.{name}")
            params = {k: v for k, v in d.items() if k != "type"}
            distributions[name] = DistributionConfig(type=dist_type, params=params)

        return AgentDistributionalConfig(
            mode=mode,
            num_types=num_types,
            distributions=distributions,
            preference_rules=data.get("preference_rules"),
            zone_distributions=data.get("zone_distributions"),
        )
    else:
        agents_list = _require(data, "agents", str(path))
        return AgentDistributionalConfig(
            mode=mode,
            num_types=len(agents_list),
            distributions={},
            explicit_agents=agents_list,
        )


def load_policy(path: str | Path) -> PolicyConfig:
    """Load and validate a policy configuration file."""
    data = _load_yaml(path)

    total_budget = _require(data, "total_budget", str(path))

    transit_investments = []
    for t in data.get("transit_investments", []):
        transit_investments.append(TransitInvestment(
            route=t["route"],
            mode=t["mode"],
            cost=t["cost"],
            new_time_minutes=t["new_time_minutes"],
            new_cost_dollars=t.get("new_cost_dollars", 0.0),
        ))

    facility_investments = []
    for f in data.get("facility_investments", []):
        facility_investments.append(FacilityInvestment(
            zone=f["zone"],
            type=f["type"],
            cost=f["cost"],
            capacity=f["capacity"],
            quality=f["quality"],
        ))

    # Validate budget
    total_cost = (
        sum(t.cost for t in transit_investments)
        + sum(f.cost for f in facility_investments)
    )
    if total_cost > total_budget:
        raise ConfigError(
            f"Policy '{data.get('name', path)}' investments total {total_cost} "
            f"exceeds budget {total_budget}"
        )

    return PolicyConfig(
        name=data.get("name", Path(path).stem),
        description=data.get("description", ""),
        total_budget=total_budget,
        transit_investments=transit_investments,
        facility_investments=facility_investments,
        budget_unit=data.get("budget_unit", ""),
    )
