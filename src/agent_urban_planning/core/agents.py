"""Heterogeneous household agent model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from agent_urban_planning.data.loaders import AgentDistributionalConfig


@dataclass
class PreferenceWeights:
    """Four-axis preference weights for an agent or archetype.

    Captures the relative importance an agent assigns to housing
    affordability, commute, services/facilities, and amenities. Values
    are stored unnormalized; call :meth:`normalized` to get a copy
    summing to 1.

    Attributes:
        alpha: Weight on housing affordability.
        beta: Weight on commute disutility.
        gamma: Weight on services / facilities accessibility.
        delta: Weight on neighborhood amenities.

    Examples:
        >>> from agent_urban_planning.core.agents import PreferenceWeights
        >>> w = PreferenceWeights(alpha=2.0, beta=1.0, gamma=1.0, delta=1.0)
        >>> w.normalized().alpha
        0.4
    """

    alpha: float  # housing affordability
    beta: float   # commute
    gamma: float  # services/facilities
    delta: float  # amenities

    def normalized(self) -> "PreferenceWeights":
        """Return a normalized copy whose four weights sum to one.

        Returns:
            New :class:`PreferenceWeights` whose components are scaled
            so ``alpha + beta + gamma + delta == 1.0``. If all weights
            are zero, returns equal weights (0.25 each) to avoid a
            division-by-zero downstream.

        Examples:
            >>> from agent_urban_planning.core.agents import PreferenceWeights
            >>> PreferenceWeights(1.0, 1.0, 1.0, 1.0).normalized().alpha
            0.25
        """
        total = self.alpha + self.beta + self.gamma + self.delta
        if total == 0:
            # Edge case: all weights are zero → equal weights as safe fallback.
            # NOT a model parameter — just prevents division by zero.
            return PreferenceWeights(0.25, 0.25, 0.25, 0.25)
        return PreferenceWeights(
            self.alpha / total,
            self.beta / total,
            self.gamma / total,
            self.delta / total,
        )


@dataclass
class Agent:
    """One representative household type (a weighted demographic record).

    Each agent represents a slice of the population. Demographic fields
    drive both the closed-form utility computation (via income and
    location) and the LLM-elicited persona used by V5 engines (through
    :func:`persona_summary`). The optional richer demographic fields
    (education, migration, employment, tenure) are populated only when
    the agent is sampled from the 10D Berlin joint distribution; legacy
    engines (V1..V4-B) ignore them.

    Attributes:
        agent_id: Stable integer identifier across runs.
        household_size: Number of persons in the household.
        age_head: Age of the household head (constrained 21-85 in
            sampling).
        has_children: Whether the household has any members aged < 18.
        has_elderly: Whether the household has any members aged >= 65.
        income: Monthly household income in scenario-currency units.
        savings: Liquid savings (default ``income * 6``).
        job_location: Zone name where the agent works (for Singapore
            scenarios; ignored when the engine optimizes workplace
            jointly with residence).
        car_owner: Whether the household owns a car.
        weight: Population share assigned to this type. Across the full
            :class:`AgentPopulation` these weights must sum to 1.
        preferences: Optional :class:`PreferenceWeights` recorded in
            the JSON output. Most engines ignore this — it is preserved
            for diagnostic output.
        home_zone: Planning area where the agent resides (used by
            engines that need an outside-option reference).
        education: ``"low"``, ``"mid"``, ``"high"``, or ``None``.
        migration_background: ``"none"``, ``"EU"``, ``"non-EU"``, or
            ``None``.
        employment_status: ``"employed"``, ``"self-employed"``,
            ``"unemployed"``, ``"retired_or_student"``, or ``None``.
        tenure: ``"owner"``, ``"renter"``, or ``None``.

    Examples:
        >>> from agent_urban_planning.core.agents import Agent
        >>> a = Agent(
        ...     agent_id=0, household_size=3, age_head=42, has_children=True,
        ...     has_elderly=False, income=5000.0, savings=30000.0,
        ...     job_location="CBD", car_owner=False, weight=0.001,
        ... )
        >>> a.income
        5000.0
    """

    agent_id: int
    household_size: int
    age_head: int
    has_children: bool
    has_elderly: bool
    income: float
    savings: float
    job_location: str
    car_owner: bool
    weight: float  # population share, all weights sum to 1.0
    # Preference weights (α,β,γ,δ). Historical field — NOT read by any
    # decision engine (UtilityEngine and EstimatedUtilityEngine use the
    # linear β model with literature coefficients). Kept for recording
    # in simulation results JSON. Default equal weights.
    preferences: PreferenceWeights = field(
        default_factory=lambda: PreferenceWeights(0.25, 0.25, 0.25, 0.25)
    )
    home_zone: str = ""  # planning area where the agent resides (for outside option)
    # ---- Richer demographic fields (zensus-richer-demographics) -----------
    # Optional. Populated when agent is sampled from the richer 10D joint
    # (joint_2011_richer.npz) via sample_ortsteile_agent_types.py
    # --joint-version richer. All legacy engines (V1..V4-B) ignore these.
    # V5-hierarchical consumes them through persona_summary().
    education: Optional[str] = None              # "low" | "mid" | "high" | None
    migration_background: Optional[str] = None   # "none" | "EU" | "non-EU" | None
    employment_status: Optional[str] = None      # "employed" | "self-employed" | "unemployed" | "retired_or_student" | None
    tenure: Optional[str] = None                 # "owner" | "renter" | None


# ---------------------------------------------------------------------------
# Persona summary helper (used by V5-hierarchical prompts + diagnostics).
# ---------------------------------------------------------------------------

# Fixed ordering so the same agent always yields the same string. Omits any
# field set to None so pre-richer agents (V1..V4-B) produce a shorter but
# still-valid persona.
_PERSONA_FIELD_ORDER: tuple[str, ...] = (
    "age_head", "household_size", "has_children", "has_elderly",
    "income", "education", "migration_background", "employment_status",
    "tenure", "car_owner",
)


def persona_summary(agent: "Agent") -> str:
    """Produce a stable, human-readable one-line persona from an agent's demographics.

    Fields appear in a fixed order; ``None`` fields are omitted. Output
    format is stable across calls (deterministic given the same agent),
    suitable for use as a cluster-label diagnostic or as the persona
    block in an LLM prompt.

    Args:
        agent: An :class:`Agent` instance whose demographics are
            rendered.

    Returns:
        Comma-separated persona string. Example:
        ``"38y, 3-person household, has children, mid income, renter,
        high education, EU background, employed, no car"``.

    Examples:
        >>> from agent_urban_planning.core.agents import Agent, persona_summary
        >>> a = Agent(
        ...     agent_id=0, household_size=2, age_head=30, has_children=False,
        ...     has_elderly=False, income=5000.0, savings=30000.0,
        ...     job_location="x", car_owner=True, weight=1.0,
        ... )
        >>> persona_summary(a)  # doctest: +ELLIPSIS
        '30y, 2-person household, no children, ... car owner'
    """
    parts: list[str] = []
    for field_name in _PERSONA_FIELD_ORDER:
        v = getattr(agent, field_name, None)
        if v is None:
            continue
        if field_name == "age_head":
            parts.append(f"{int(v)}y")
        elif field_name == "household_size":
            parts.append(f"{int(v)}-person household")
        elif field_name == "has_children":
            parts.append("has children" if v else "no children")
        elif field_name == "has_elderly":
            if v:
                parts.append("has elderly")
        elif field_name == "income":
            parts.append(_income_bucket_label(float(v)))
        elif field_name == "education":
            parts.append(f"{v} education")
        elif field_name == "migration_background":
            parts.append("native" if v == "none" else f"{v} background")
        elif field_name == "employment_status":
            parts.append(v.replace("_", " "))
        elif field_name == "tenure":
            parts.append(str(v))
        elif field_name == "car_owner":
            parts.append("car owner" if v else "no car")
    return ", ".join(parts)


def _income_bucket_label(inc: float) -> str:
    """Categorize a numeric income into the fetcher's income brackets."""
    # Thresholds between low/mid/high tertiles; midpoints are 900 / 1700 / 3200.
    if inc < 1300:
        return "low income"
    if inc < 2400:
        return "mid income"
    return "high income"


class AgentPopulation:
    """Collection of weighted representative agent types.

    Holds the full list of :class:`Agent` instances comprising the
    simulation's population, indexed by integer position. Each agent
    carries a ``weight`` that represents its population share; weights
    must sum to 1.0 across the population. Construct via
    :meth:`from_config` from a parsed agent YAML, which handles
    distributional sampling (per-zone Census or single national) or
    explicit per-agent records.

    Args:
        agents: List of :class:`Agent` instances. The constructor
            validates that their weights sum to 1.0 within tolerance.

    Raises:
        ValueError: If ``agents`` weights do not sum to 1 within
            ``1e-6``.

    Examples:
        >>> import agent_urban_planning as aup
        >>> # config = aup.data.builtin.load_agents("singapore_real_v2")
        >>> # pop = aup.AgentPopulation.from_config(config)
        >>> # len(pop)  # number of representative types
        >>> # for agent in pop:
        >>> #     ... # iterate over the population
    """

    def __init__(self, agents: list[Agent]):
        self.agents = agents
        self._validate_weights()

    def _validate_weights(self):
        total = sum(a.weight for a in self.agents)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Agent weights must sum to 1.0, got {total:.6f}"
            )

    def __len__(self) -> int:
        return len(self.agents)

    def __iter__(self):
        return iter(self.agents)

    def __getitem__(self, idx: int) -> Agent:
        return self.agents[idx]

    @classmethod
    def from_config(
        cls,
        config: AgentDistributionalConfig,
        rng: Optional[np.random.RandomState] = None,
        strict: bool = True,
    ) -> "AgentPopulation":
        """Generate an agent population from a configuration object.

        Dispatches on ``config.mode``: ``"distributional"`` samples
        agents from per-zone Census distributions (or, for unit-test
        configs with ``strict=False``, a single national distribution),
        otherwise loads explicitly declared agent records.

        Args:
            config: Parsed :class:`AgentDistributionalConfig` from a
                YAML file.
            rng: Optional ``numpy.random.RandomState`` for reproducible
                sampling. If ``None``, a fresh state is created.
            strict: If ``True`` (default), reject configs without
                per-zone Census data. Set to ``False`` only in unit
                tests that use small synthetic configs.

        Returns:
            A new :class:`AgentPopulation`. Weights are normalized to
            sum to 1.

        Raises:
            RuntimeError: When ``strict=True`` and the config lacks
                per-zone ``zone_distributions``.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # config = aup.data.builtin.load_agents("singapore_real_v2")
            >>> # pop = aup.AgentPopulation.from_config(config, strict=True)
        """
        if config.mode == "distributional":
            # Per-zone Census sampling if zone_distributions is present
            if config.zone_distributions and len(config.zone_distributions) > 0:
                return cls._generate_from_zone_distributions(config, rng)
            if strict:
                raise RuntimeError(
                    "Agent config has no zone_distributions. Cannot generate "
                    "agents without per-zone Census data — a single national "
                    "distribution produces spatially uniform demographics with "
                    "no inter-zone income variation, which is not acceptable. "
                    "Use config/agents/singapore_real_v2.yaml (has per-zone "
                    "Census 2020 data) or run the data fetcher to generate it."
                )
            # strict=False: allow old single-distribution path (for unit tests only)
            return cls._generate_from_distributions(config, rng)
        else:
            return cls._load_explicit(config)

    @classmethod
    def _generate_from_distributions(
        cls,
        config: AgentDistributionalConfig,
        rng: Optional[np.random.RandomState] = None,
    ) -> "AgentPopulation":
        if rng is None:
            rng = np.random.RandomState()

        n = config.num_types
        dists = config.distributions

        # Sample each feature
        income = _sample_distribution(dists["income"], n, rng)
        age_head = _sample_distribution(dists["age_head"], n, rng).astype(int)
        household_size = _sample_distribution(dists["household_size"], n, rng).astype(int)
        has_children = _sample_distribution(dists["has_children"], n, rng).astype(bool)
        has_elderly = _sample_distribution(dists["has_elderly"], n, rng).astype(bool)
        car_owner = _sample_distribution(dists["car_owner"], n, rng).astype(bool)
        job_location = _sample_distribution(dists["job_location"], n, rng)

        # Equal weights for sampled types
        weight = 1.0 / n

        agents = []
        for i in range(n):
            agents.append(Agent(
                agent_id=i,
                household_size=int(household_size[i]),
                age_head=int(age_head[i]),
                has_children=bool(has_children[i]),
                has_elderly=bool(has_elderly[i]),
                income=float(income[i]),
                savings=float(income[i]) * 6,  # rough default: 6 months income
                job_location=str(job_location[i]),
                car_owner=bool(car_owner[i]),
                weight=weight,
            ))

        return cls(agents)

    @classmethod
    def _generate_from_zone_distributions(
        cls,
        config: AgentDistributionalConfig,
        rng: Optional[np.random.RandomState] = None,
    ) -> "AgentPopulation":
        """Generate agents from per-zone Census 2020 distributions.

        Each agent's demographics are sampled from the histogram of their
        home zone, producing a population that statistically matches each
        zone's real Census profile.
        """
        if rng is None:
            rng = np.random.RandomState()

        n = config.num_types
        zd = config.zone_distributions
        dists = config.distributions

        # 1. Compute population-proportional agent counts per zone
        total_pop = sum(z.get("population", 0) for z in zd.values()) or 1
        zone_agent_counts: dict[str, int] = {}
        remaining = n
        zone_names = sorted(zd.keys())
        for i, zone_name in enumerate(zone_names):
            z = zd[zone_name]
            if i == len(zone_names) - 1:
                zone_agent_counts[zone_name] = remaining  # last zone gets remainder
            else:
                count = max(1, round(n * z.get("population", 0) / total_pop))
                count = min(count, remaining)
                zone_agent_counts[zone_name] = count
                remaining -= count

        # 2. Sample job_location weights from employment-density distribution
        job_dist = dists.get("job_location")
        if job_dist:
            job_values = job_dist.params["values"]
            job_weights = np.array(job_dist.params["weights"], dtype=float)
            job_weights /= job_weights.sum()
        else:
            job_values = zone_names
            job_weights = np.ones(len(zone_names)) / len(zone_names)

        # 3. Get car_owner rate (national — no per-zone data)
        car_pct = dists["car_owner"].params["p"] if "car_owner" in dists else 0.28

        # 3b. Compute median employment for agglomeration wage premium.
        #     Employment counts are embedded in job_location weights (proportional).
        #     We use population as a proxy if per-zone employment isn't available.
        zone_populations = [z.get("population", 0) for z in zd.values()]
        median_pop = float(np.median([p for p in zone_populations if p > 0])) if zone_populations else 1.0
        # Build a lookup: job_location → zone population (proxy for employment)
        zone_pop_lookup = {zone: zd[zone].get("population", 0) for zone in zd}

        # 4. Generate agents per zone
        agents = []
        agent_id = 0
        weight = 1.0 / n

        for zone_name in zone_names:
            z = zd[zone_name]
            zone_n = zone_agent_counts[zone_name]
            if zone_n <= 0:
                continue

            income_brackets = z.get("income_brackets", [])
            hh_hist = z.get("household_size_hist", [])
            age_brackets = z.get("age_brackets", [])

            for _ in range(zone_n):
                # Sample income from Census bracket histogram
                sampled_income = _sample_census_income(income_brackets, rng)

                # Sample household size from Census histogram
                sampled_hh = _sample_census_hh_size(hh_hist, rng)

                # Sample age from Census age brackets
                sampled_age = _sample_census_age(age_brackets, rng)

                # Derive has_children / has_elderly from zone age distribution
                children_pct = _derive_children_pct(age_brackets)
                elderly_pct = _derive_elderly_pct(age_brackets)
                has_children = bool(rng.random() < children_pct)
                has_elderly = bool(rng.random() < elderly_pct)

                # Sample job_location from employment weights
                job_idx = rng.choice(len(job_values), p=job_weights)
                job_loc = str(job_values[job_idx])

                # Sample car_owner from national rate
                car_owner = bool(rng.random() < car_pct)

                # Apply agglomeration wage premium based on job_location
                from agent_urban_planning.core.constraints import compute_effective_income
                job_zone_pop = zone_pop_lookup.get(job_loc, median_pop)
                effective_income = compute_effective_income(
                    sampled_income, int(job_zone_pop), median_pop
                )

                agents.append(Agent(
                    agent_id=agent_id,
                    household_size=sampled_hh,
                    age_head=sampled_age,
                    has_children=has_children,
                    has_elderly=has_elderly,
                    income=effective_income,
                    savings=effective_income * 6,
                    job_location=job_loc,
                    car_owner=car_owner,
                    weight=weight,
                    home_zone=zone_name,
                ))
                agent_id += 1

        return cls(agents)

    @classmethod
    def _load_explicit(cls, config: AgentDistributionalConfig) -> "AgentPopulation":
        agents = []
        for i, ad in enumerate(config.explicit_agents):
            agents.append(Agent(
                agent_id=i,
                household_size=ad["household_size"],
                age_head=ad["age_head"],
                has_children=ad["has_children"],
                has_elderly=ad["has_elderly"],
                income=ad["income"],
                savings=ad.get("savings", ad["income"] * 6),
                job_location=ad["job_location"],
                car_owner=ad["car_owner"],
                weight=ad["weight"],
                # Richer demographic fields (zensus-richer-demographics).
                # Default to None so pre-richer YAMLs still load.
                education=ad.get("education"),
                migration_background=ad.get("migration_background"),
                employment_status=ad.get("employment_status"),
                tenure=ad.get("tenure"),
            ))
        return cls(agents)


def _sample_distribution(dist_config, n: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample n values from a distribution config."""
    dtype = dist_config.type
    params = dist_config.params

    if dtype == "lognormal":
        # Convert mean/sigma to lognormal params
        mean = params["mean"]
        sigma = params["sigma"]
        mu_ln = np.log(mean) - 0.5 * sigma**2
        samples = rng.lognormal(mu_ln, sigma, n)
        return np.maximum(samples, 100)  # floor at 100

    elif dtype == "truncated_normal":
        mean = params["mean"]
        std = params["std"]
        lo = params.get("min", mean - 4 * std)
        hi = params.get("max", mean + 4 * std)
        a = (lo - mean) / std
        b = (hi - mean) / std
        return stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=n, random_state=rng)

    elif dtype == "categorical":
        values = params["values"]
        weights = params["weights"]
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        indices = rng.choice(len(values), size=n, p=weights)
        return np.array([values[i] for i in indices])

    elif dtype == "bernoulli":
        p = params["p"]
        return rng.random(n) < p

    else:
        raise ValueError(f"Unknown distribution type: {dtype}")


# ------------------------------------------------------------------
# Census histogram sampling helpers
# ------------------------------------------------------------------

# Income bracket label → (lower_bound, upper_bound) in SGD
_INCOME_BRACKET_RANGES: dict[str, tuple[float, float]] = {
    "NoEmployedPerson": (0, 0),
    "Below_1_000": (200, 1000),
    "1_000_1_999": (1000, 2000),
    "2_000_2_999": (2000, 3000),
    "3_000_3_999": (3000, 4000),
    "4_000_4_999": (4000, 5000),
    "5_000_5_999": (5000, 6000),
    "6_000_6_999": (6000, 7000),
    "7_000_7_999": (7000, 8000),
    "8_000_8_999": (8000, 9000),
    "9_000_9_999": (9000, 10000),
    "10_000_10_999": (10000, 11000),
    "11_000_11_999": (11000, 12000),
    "12_000_12_999": (12000, 13000),
    "13_000_13_999": (13000, 14000),
    "14_000_14_999": (14000, 15000),
    "15_000_17_499": (15000, 17500),
    "17_500_19_999": (17500, 20000),
    "20_000andOver": (20000, 35000),
}


def _sample_census_income(
    brackets: list, rng: np.random.RandomState
) -> float:
    """Sample one income from a Census income bracket histogram.

    Each bracket is [label, count]. We weighted-random a bracket,
    then uniform-sample within the bracket range.
    """
    if not brackets:
        return 5000.0  # national median fallback

    labels = [b[0] if isinstance(b, (list, tuple)) else b for b in brackets]
    counts = [b[1] if isinstance(b, (list, tuple)) else 0 for b in brackets]

    # Filter to brackets with positive count and known range
    valid = []
    weights = []
    for label, count in zip(labels, counts):
        if count > 0 and label in _INCOME_BRACKET_RANGES:
            lo, hi = _INCOME_BRACKET_RANGES[label]
            if lo == 0 and hi == 0:
                continue  # skip NoEmployedPerson
            valid.append((lo, hi))
            weights.append(count)

    if not valid:
        return 5000.0

    weights_arr = np.array(weights, dtype=float)
    weights_arr /= weights_arr.sum()
    idx = rng.choice(len(valid), p=weights_arr)
    lo, hi = valid[idx]
    return float(rng.uniform(lo, hi))


def _sample_census_hh_size(
    hist: list, rng: np.random.RandomState
) -> int:
    """Sample one household size from a Census histogram.

    Each entry is [size, count]. The last bucket (8) represents 8+.
    """
    if not hist:
        return 3

    sizes = [int(h[0]) if isinstance(h, (list, tuple)) else int(h) for h in hist]
    counts = [int(h[1]) if isinstance(h, (list, tuple)) else 0 for h in hist]

    total = sum(counts)
    if total <= 0:
        return 3

    weights = np.array(counts, dtype=float) / total
    idx = rng.choice(len(sizes), p=weights)
    return max(1, sizes[idx])


def _sample_census_age(
    brackets: list, rng: np.random.RandomState
) -> int:
    """Sample one age from a Census age bracket histogram.

    Each bracket is [label, count] where label is like "25_29" or "90+Over".
    We sample the head-of-household age (constrained to 21-85).
    """
    if not brackets:
        return 42

    valid = []
    weights = []
    for b in brackets:
        label = str(b[0]) if isinstance(b, (list, tuple)) else str(b)
        count = int(b[1]) if isinstance(b, (list, tuple)) else 0
        if count <= 0:
            continue
        # Parse age range from label like "25_29", "0_4", "90+Over"
        lo, hi = _parse_age_bracket(label)
        if lo < 21:
            lo = 21  # head of household minimum
        if hi > 85:
            hi = 85
        if lo > hi:
            continue
        valid.append((lo, hi))
        weights.append(count)

    if not valid:
        return 42

    weights_arr = np.array(weights, dtype=float)
    weights_arr /= weights_arr.sum()
    idx = rng.choice(len(valid), p=weights_arr)
    lo, hi = valid[idx]
    return int(rng.randint(lo, hi + 1))


def _parse_age_bracket(label: str) -> tuple[int, int]:
    """Parse Census age bracket label to (lo, hi) ages."""
    label = label.strip().replace("+Over", "").replace("+", "")
    parts = label.split("_")
    try:
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        elif len(parts) == 1:
            age = int(parts[0])
            return age, age + 4  # e.g., "90" → 90-94
    except ValueError:
        pass
    return 21, 85  # safe fallback


def _derive_children_pct(age_brackets: list) -> float:
    """Estimate has_children probability from zone's age distribution.

    Uses share of population aged 0-17 as proxy: if a zone has many
    children, its households are more likely to have children.
    """
    if not age_brackets:
        return 0.37  # national default

    total = 0
    children = 0
    for b in age_brackets:
        label = str(b[0])
        count = int(b[1]) if isinstance(b, (list, tuple)) else 0
        lo, _ = _parse_age_bracket(label)
        total += count
        if lo < 18:
            children += count

    return children / total if total > 0 else 0.37


def _derive_elderly_pct(age_brackets: list) -> float:
    """Estimate has_elderly probability from zone's age distribution.

    Uses share of population aged 65+ as proxy.
    """
    if not age_brackets:
        return 0.22  # national default

    total = 0
    elderly = 0
    for b in age_brackets:
        label = str(b[0])
        count = int(b[1]) if isinstance(b, (list, tuple)) else 0
        lo, _ = _parse_age_bracket(label)
        total += count
        if lo >= 65:
            elderly += count

    return elderly / total if total > 0 else 0.22


