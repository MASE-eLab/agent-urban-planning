"""Agent clustering and ClusterizedDecisionEngine wrapper.

Provides three pluggable clustering algorithms (k-means, hierarchical,
decision-tree) plus a special `none` mode that maps each agent to a unique
cluster (= no compression). All algorithms operate on a configurable list
of agent features, with one-hot encoding for categorical features.

The `ClusterizedDecisionEngine` wraps any inner `DecisionEngine` and
applies clustering before delegating to the inner engine. Clustering is
done once on first call and reused across all market iterations so cluster
identities are stable.

Two within-cluster assignment modes:
  * deterministic: build one centroid representative per cluster, call the
    inner engine once, assign the resulting choice to all members.
  * probabilistic: sample N agents per cluster, call the inner engine N
    times, build a discrete decision distribution, and assign each cluster
    member a decision sampled from the distribution using the run RNG.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from agent_urban_planning.core.agents import Agent, PreferenceWeights
from agent_urban_planning.decisions.base import LocationChoice, ZoneChoice
from agent_urban_planning.core.environment import Environment


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

DEFAULT_NUMERIC_FEATURES = [
    "income",
    "age_head",
    "household_size",
    "savings",
]
DEFAULT_BOOL_FEATURES = ["has_children", "has_elderly", "car_owner"]
DEFAULT_CATEGORICAL_FEATURES = ["job_location"]
DEFAULT_FEATURES = (
    DEFAULT_NUMERIC_FEATURES + DEFAULT_BOOL_FEATURES + DEFAULT_CATEGORICAL_FEATURES
)


def _extract_feature_matrix(
    agents: list[Agent],
    features: list[str],
) -> np.ndarray:
    """Build a (n_agents, n_features_expanded) matrix.

    Numeric features pass through. Bools become 0/1. Categorical features
    are one-hot encoded across all observed values in the population.
    """
    if not agents or not features:
        return np.zeros((len(agents), 0))

    cols: list[np.ndarray] = []

    for feat in features:
        sample = getattr(agents[0], feat)
        if isinstance(sample, bool):
            col = np.array([float(getattr(a, feat)) for a in agents])
            cols.append(col.reshape(-1, 1))
        elif isinstance(sample, (int, float)):
            col = np.array([float(getattr(a, feat)) for a in agents], dtype=float)
            cols.append(col.reshape(-1, 1))
        elif isinstance(sample, str):
            # One-hot encode
            values = sorted({getattr(a, feat) for a in agents})
            for v in values:
                col = np.array([1.0 if getattr(a, feat) == v else 0.0 for a in agents])
                cols.append(col.reshape(-1, 1))
        else:
            raise TypeError(
                f"Unsupported feature type for '{feat}': {type(sample).__name__}"
            )

    matrix = np.hstack(cols)
    return matrix


def _standardize(matrix: np.ndarray) -> np.ndarray:
    """Z-score normalize each column. Constant columns are zeroed."""
    if matrix.size == 0:
        return matrix
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return (matrix - mean) / std


# ------------------------------------------------------------------
# Clustering algorithm interface and implementations
# ------------------------------------------------------------------


class ClusteringAlgorithm:
    """Base class for clustering algorithms."""

    name: str = "base"

    def cluster(
        self,
        agents: list[Agent],
        k: int,
        features: list[str],
        seed: Optional[int] = None,
    ) -> dict[int, int]:
        """Map each agent.agent_id to a cluster_id in [0, k)."""
        raise NotImplementedError


class NoneClustering(ClusteringAlgorithm):
    """No clustering — each agent is its own cluster."""

    name = "none"

    def cluster(self, agents, k=None, features=None, seed=None):
        return {a.agent_id: i for i, a in enumerate(agents)}


class KMeansClustering(ClusteringAlgorithm):
    """K-means clustering on standardized features."""

    name = "kmeans"

    def cluster(self, agents, k, features=None, seed=None):
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if k >= len(agents):
            return NoneClustering().cluster(agents)

        from sklearn.cluster import KMeans

        feats = features or DEFAULT_FEATURES
        matrix = _standardize(_extract_feature_matrix(agents, feats))
        km = KMeans(n_clusters=k, random_state=seed if seed is not None else 0, n_init=10)
        labels = km.fit_predict(matrix)
        return {a.agent_id: int(labels[i]) for i, a in enumerate(agents)}


class HierarchicalClustering(ClusteringAlgorithm):
    """Agglomerative hierarchical clustering with Ward linkage."""

    name = "hierarchical"

    def cluster(self, agents, k, features=None, seed=None):
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if k >= len(agents):
            return NoneClustering().cluster(agents)

        from sklearn.cluster import AgglomerativeClustering

        feats = features or DEFAULT_FEATURES
        matrix = _standardize(_extract_feature_matrix(agents, feats))
        ac = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = ac.fit_predict(matrix)
        return {a.agent_id: int(labels[i]) for i, a in enumerate(agents)}


class DecisionTreeClustering(ClusteringAlgorithm):
    """Decision-tree partitioning that minimizes within-cluster utility variance.

    Strategy: use the existing UtilityEngine to compute per-zone utilities for
    each agent at the environment's *base* prices. Treat the per-zone utility
    vectors as the target signal. Fit a regression tree on (features →
    utility vector) with max_leaf_nodes = k, then use the leaf id as the
    cluster label.

    This produces clusters whose members have similar decision-relevant
    utility patterns, which is more useful than purely demographic clustering
    when the goal is to compress LLM calls without losing decision diversity.
    """

    name = "decision_tree"

    def __init__(self, environment: Optional[Environment] = None):
        self.environment = environment

    def cluster(self, agents, k, features=None, seed=None):
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if k >= len(agents):
            return NoneClustering().cluster(agents)

        from sklearn.tree import DecisionTreeRegressor
        from agent_urban_planning.decisions.utility import UtilityEngine

        feats = features or DEFAULT_FEATURES
        X = _standardize(_extract_feature_matrix(agents, feats))

        # Compute target: per-zone utilities at base prices.
        # If we have no environment, fall back to k-means style on features.
        if self.environment is None:
            return KMeansClustering().cluster(agents, k, features, seed)

        zone_names = self.environment.zone_names
        base_prices = {
            name: self.environment.get_zone(name).housing_base_price
            for name in zone_names
        }
        utility_engine = UtilityEngine()

        Y = np.zeros((len(agents), len(zone_names)))
        for i, agent in enumerate(agents):
            choice = utility_engine.decide(agent, self.environment, zone_names, base_prices)
            for j, z in enumerate(zone_names):
                Y[i, j] = choice.zone_utilities.get(z, 0.0)

        # Fit a regression tree with at most k leaves.
        tree = DecisionTreeRegressor(
            max_leaf_nodes=max(2, k),
            random_state=seed if seed is not None else 0,
        )
        tree.fit(X, Y)
        leaf_ids = tree.apply(X)
        # Re-label leaves as 0..k-1
        unique_leaves = sorted(set(leaf_ids))
        relabel = {old: new for new, old in enumerate(unique_leaves)}
        return {a.agent_id: relabel[int(leaf_ids[i])] for i, a in enumerate(agents)}


# ------------------------------------------------------------------
# Algorithm registry
# ------------------------------------------------------------------


_ALGORITHM_FACTORIES: dict[str, Callable[..., ClusteringAlgorithm]] = {
    "none": lambda **kwargs: NoneClustering(),
    "kmeans": lambda **kwargs: KMeansClustering(),
    "hierarchical": lambda **kwargs: HierarchicalClustering(),
    "decision_tree": lambda environment=None, **kwargs: DecisionTreeClustering(
        environment=environment
    ),
}


def register_clustering(name: str, factory: Callable[..., ClusteringAlgorithm]):
    _ALGORITHM_FACTORIES[name] = factory


def get_clustering(name: str, **kwargs) -> ClusteringAlgorithm:
    if name not in _ALGORITHM_FACTORIES:
        available = ", ".join(sorted(_ALGORITHM_FACTORIES.keys()))
        raise KeyError(f"Unknown clustering algorithm '{name}'. Available: {available}")
    return _ALGORITHM_FACTORIES[name](**kwargs)


def list_clustering_algorithms() -> list[str]:
    return sorted(_ALGORITHM_FACTORIES.keys())


# ------------------------------------------------------------------
# Centroid construction
# ------------------------------------------------------------------


def _weighted_mode(values: list, weights: list[float]):
    """Return the value with the highest accumulated weight."""
    if not values:
        return None
    totals: dict = {}
    for v, w in zip(values, weights):
        totals[v] = totals.get(v, 0.0) + w
    return max(totals.items(), key=lambda kv: kv[1])[0]


def _build_centroid_agent(
    cluster_id: int,
    members: list[Agent],
) -> Agent:
    """Construct a representative agent from cluster members.

    Numeric: weighted mean. Bool/categorical: weighted mode.
    Preference weights: weighted mean of each component.
    """
    if not members:
        raise ValueError("Cannot build centroid from empty cluster")

    weights = [m.weight for m in members]
    total_w = sum(weights) or 1.0

    def wmean(attr: str) -> float:
        return sum(getattr(m, attr) * m.weight for m in members) / total_w

    def wmode(attr: str):
        return _weighted_mode([getattr(m, attr) for m in members], weights)

    prefs = PreferenceWeights(
        alpha=sum(m.preferences.alpha * m.weight for m in members) / total_w,
        beta=sum(m.preferences.beta * m.weight for m in members) / total_w,
        gamma=sum(m.preferences.gamma * m.weight for m in members) / total_w,
        delta=sum(m.preferences.delta * m.weight for m in members) / total_w,
    )

    return Agent(
        agent_id=10**9 + cluster_id,  # synthetic ID outside normal agent_id range
        household_size=int(round(wmean("household_size"))),
        age_head=int(round(wmean("age_head"))),
        has_children=bool(wmode("has_children")),
        has_elderly=bool(wmode("has_elderly")),
        income=wmean("income"),
        savings=wmean("savings"),
        job_location=str(wmode("job_location")),
        car_owner=bool(wmode("car_owner")),
        weight=total_w,
        preferences=prefs,
    )


# ------------------------------------------------------------------
# ClusterizedDecisionEngine wrapper
# ------------------------------------------------------------------


@dataclass
class ClusteringConfig:
    algo: str = "none"
    k: int = 50
    features: Optional[list[str]] = None
    samples_per_archetype: int = 1
    within_cluster_assignment: str = "deterministic"  # or "probabilistic"
    seed: Optional[int] = None


class ClusterizedDecisionEngine:
    """Wraps an inner DecisionEngine with opt-in agent clustering.

    The wrapper computes cluster assignments on first call (or via
    ``prepare(agents)``) and reuses them across all market iterations. Each
    iteration calls the inner engine once per cluster (deterministic) or
    N times per cluster (probabilistic), then maps decisions back to all
    cluster members.
    """

    def __init__(
        self,
        inner: object,
        config: ClusteringConfig,
        environment: Optional[Environment] = None,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.inner = inner
        self.config = config
        self.environment = environment
        self.rng = rng or np.random.RandomState(config.seed)
        self.cluster_assignments: Optional[dict[int, int]] = None
        self._members_by_cluster: Optional[dict[int, list[Agent]]] = None
        self._centroid_by_cluster: Optional[dict[int, Agent]] = None
        self._algorithm: Optional[ClusteringAlgorithm] = None

    # ------------------------------------------------------------------
    # Cache and inner-engine pass-through
    # ------------------------------------------------------------------

    def set_cache(self, cache) -> None:
        if hasattr(self.inner, "set_cache"):
            self.inner.set_cache(cache)

    @property
    def total_input_tokens(self) -> int:
        return int(getattr(self.inner, "total_input_tokens", 0) or 0)

    @property
    def total_output_tokens(self) -> int:
        return int(getattr(self.inner, "total_output_tokens", 0) or 0)

    # ------------------------------------------------------------------
    # Clustering bootstrap
    # ------------------------------------------------------------------

    def prepare(self, agents: list[Agent]):
        """Compute cluster assignments. Idempotent — only runs once."""
        if self.cluster_assignments is not None:
            return
        algo = get_clustering(
            self.config.algo,
            environment=self.environment,
        )
        self._algorithm = algo
        self.cluster_assignments = algo.cluster(
            agents,
            k=self.config.k,
            features=self.config.features,
            seed=self.config.seed,
        )
        # Group members by cluster
        members: dict[int, list[Agent]] = {}
        agent_by_id = {a.agent_id: a for a in agents}
        for agent_id, cluster_id in self.cluster_assignments.items():
            members.setdefault(cluster_id, []).append(agent_by_id[agent_id])
        self._members_by_cluster = members
        # Build centroid per cluster (deterministic mode uses these)
        self._centroid_by_cluster = {
            cid: _build_centroid_agent(cid, mems) for cid, mems in members.items()
        }

    # ------------------------------------------------------------------
    # decide / decide_batch
    # ------------------------------------------------------------------

    def decide(self, agent, environment, zone_options, prices):
        # Single-agent fallback: just delegate to inner engine
        return self.inner.decide(agent, environment, zone_options, prices)

    def decide_batch(self, agents, environment, zone_options, prices):
        if not agents:
            return []
        self.prepare(agents)

        if self.config.within_cluster_assignment == "deterministic":
            return self._decide_deterministic(agents, environment, zone_options, prices)
        elif self.config.within_cluster_assignment == "probabilistic":
            return self._decide_probabilistic(agents, environment, zone_options, prices)
        else:
            raise ValueError(
                f"Unknown within_cluster_assignment: {self.config.within_cluster_assignment}"
            )

    def _decide_deterministic(self, agents, environment, zone_options, prices):
        # Build one centroid per cluster, send to inner engine in one batch
        cluster_ids = sorted(self._members_by_cluster.keys())
        centroids = [self._centroid_by_cluster[cid] for cid in cluster_ids]

        choices = self.inner.decide_batch(centroids, environment, zone_options, prices)
        cluster_to_choice = {cid: choice for cid, choice in zip(cluster_ids, choices)}

        # Assign each agent the choice from its cluster.
        # If target agent has its own job_location (Singapore-style), override
        # the cluster centroid's workplace with the target's own workplace
        # so zone_employment metrics remain accurate.
        result = []
        for agent in agents:
            cid = self.cluster_assignments[agent.agent_id]
            src = cluster_to_choice[cid]
            result.append(_specialize_workplace(src, agent))
        return result

    def _decide_probabilistic(self, agents, environment, zone_options, prices):
        n_samples = max(1, self.config.samples_per_archetype)
        cluster_ids = sorted(self._members_by_cluster.keys())

        # Build N sample agents per cluster (sampled from cluster members)
        all_samples: list[Agent] = []
        sample_cluster_ids: list[int] = []
        for cid in cluster_ids:
            members = self._members_by_cluster[cid]
            if len(members) <= n_samples:
                samples = list(members)
            else:
                indices = self.rng.choice(len(members), size=n_samples, replace=False)
                samples = [members[i] for i in indices]
            for s in samples:
                all_samples.append(s)
                sample_cluster_ids.append(cid)

        # Single batched call for all sample agents
        sample_choices = self.inner.decide_batch(
            all_samples, environment, zone_options, prices
        )

        # Group choices back by cluster
        cluster_distributions: dict[int, list[ZoneChoice]] = {}
        for cid, choice in zip(sample_cluster_ids, sample_choices):
            cluster_distributions.setdefault(cid, []).append(choice)

        # Assign each cluster member by sampling from the cluster's
        # distribution of zone choices. Workplace is specialized to the
        # target agent's own job_location when the agent has one.
        result = []
        for agent in agents:
            cid = self.cluster_assignments[agent.agent_id]
            choices = cluster_distributions[cid]
            if len(choices) == 1:
                src = choices[0]
            else:
                idx = int(self.rng.randint(0, len(choices)))
                src = choices[idx]
            result.append(_specialize_workplace(src, agent))
        return result


def _specialize_workplace(src: LocationChoice, target: Agent) -> LocationChoice:
    """Return a copy of ``src`` with workplace replaced by ``target.job_location``
    when the target agent has a meaningful job_location.

    For Singapore scenarios, job_location is an agent attribute, so the
    cluster-centroid's workplace should be overridden with the target's
    own so aggregate employment metrics remain accurate.

    For Berlin scenarios, homogeneous agents have an empty job_location;
    in that case the inner engine's chosen workplace is preserved.
    """
    target_job = getattr(target, "job_location", "") or ""
    if not target_job:
        return src
    if src.workplace == target_job:
        return src
    return LocationChoice(
        residence=src.residence,
        workplace=target_job,
        utility=src.utility,
        zone_utilities=src.zone_utilities,
    )
