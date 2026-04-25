"""Per-run metadata tracking and cost estimation.

Captures everything needed to make a simulation run reproducible from the
output JSON: scenario / policy / seed, LLM provider details, performance
counters (call count, cache hit rate, wall-clock time), clustering
configuration (algorithm + k + assignments), and an estimated USD cost
based on a static per-model price table.

Saved automatically alongside results when LLM mode or clustering is used,
or on demand via `--save-metadata`.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ------------------------------------------------------------------
# Cost table
# ------------------------------------------------------------------

# Approximate USD per 1k tokens. Numbers are rough public list prices and
# should be treated as order-of-magnitude estimates only. Format:
#   {(provider, model): (input_per_1k, output_per_1k)}
LLM_COST_TABLE: dict[tuple[str, str], tuple[float, float]] = {
    # Z.ai GLM family — community/coding tier estimates
    ("zai-coding", "glm-4.7"): (0.0006, 0.0022),
    ("zai-coding", "glm-4.6"): (0.0006, 0.0022),
    ("zai-coding", "glm-4.5"): (0.0005, 0.0015),
    ("zai-coding", "glm-4.5-air"): (0.0002, 0.0006),
    # Anthropic
    ("anthropic", "claude-haiku-4-5-20251001"): (0.001, 0.005),
    ("anthropic", "claude-sonnet-4-6"): (0.003, 0.015),
    ("anthropic", "claude-opus-4-6"): (0.015, 0.075),
    # OpenAI
    ("openai", "gpt-4o-mini"): (0.00015, 0.0006),
    ("openai", "gpt-4o"): (0.0025, 0.010),
}


def compute_cost(
    provider: Optional[str],
    model: Optional[str],
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate USD cost from token counts. Returns 0.0 for unknown models."""
    if not provider or not model:
        return 0.0
    key = (provider.lower(), model.lower())
    if key not in LLM_COST_TABLE:
        return 0.0
    in_per_1k, out_per_1k = LLM_COST_TABLE[key]
    return (input_tokens / 1000.0) * in_per_1k + (output_tokens / 1000.0) * out_per_1k


# ------------------------------------------------------------------
# RunMetadata dataclass
# ------------------------------------------------------------------


def _new_run_id() -> str:
    """Generate a unique run id: timestamp + short random suffix."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{ts}_{suffix}"


@dataclass
class RunMetadata:
    """Metadata for a single simulation run.

    Every field is optional so this can be incrementally populated during a
    run. JSON-serializable for output alongside results.
    """

    run_id: str = field(default_factory=_new_run_id)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    scenario_name: Optional[str] = None
    policy_name: Optional[str] = None
    seed: Optional[int] = None

    # LLM info
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_temperature: Optional[float] = None
    llm_concurrency: Optional[int] = None

    # Performance counters
    total_llm_calls: int = 0
    cached_llm_calls: int = 0
    cache_hit_rate: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    wall_clock_seconds: float = 0.0
    estimated_cost_usd: float = 0.0

    # Reliability counters (pure LLM mode)
    llm_retry_count: int = 0          # total retries triggered by transient errors
    llm_failed_calls: int = 0          # calls that failed after all retries exhausted
    llm_success_rate: float = 1.0      # successful / (successful + failed_final)

    # Clustering
    clustering_algo: str = "none"
    num_archetypes: Optional[int] = None
    samples_per_archetype: int = 1
    within_cluster_assignment: str = "deterministic"
    cluster_features: list[str] = field(default_factory=list)
    cluster_assignments: Optional[dict[int, int]] = None

    # Market clearing
    price_elasticity_used: Optional[float] = None
    damping_final: Optional[float] = None
    market_iterations_actual: Optional[int] = None
    convergence_achieved: Optional[bool] = None

    # Other
    decision_engine_name: Optional[str] = None
    notes: Optional[str] = None

    # ------------------------------------------------------------------
    # Computed helpers
    # ------------------------------------------------------------------

    def update_cost(self):
        """Recompute estimated_cost_usd from the cost table."""
        self.estimated_cost_usd = compute_cost(
            self.llm_provider,
            self.llm_model,
            self.total_input_tokens,
            self.total_output_tokens,
        )

    def update_cache_hit_rate(self):
        total = self.total_llm_calls + self.cached_llm_calls
        if total > 0:
            self.cache_hit_rate = self.cached_llm_calls / total
        else:
            self.cache_hit_rate = 0.0

    def update_llm_success_rate(self):
        """Compute the LLM success rate from successful and failed counters.

        In pure LLM mode, this SHOULD be 1.0 — any value less than 1.0 means
        some agents' decisions could not be made by the LLM. By design the
        simulation aborts rather than falling back to utility, so a <1.0
        value only occurs if manual failure injection or an incomplete run.
        """
        total = self.total_llm_calls + self.llm_failed_calls
        if total > 0:
            self.llm_success_rate = self.total_llm_calls / total
        else:
            self.llm_success_rate = 1.0

    # ------------------------------------------------------------------
    # JSON I/O
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert int keys in cluster_assignments to strings for JSON
        if self.cluster_assignments is not None:
            d["cluster_assignments"] = {
                str(k): int(v) for k, v in self.cluster_assignments.items()
            }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunMetadata":
        ca = data.get("cluster_assignments")
        if ca is not None:
            data = dict(data)
            data["cluster_assignments"] = {int(k): int(v) for k, v in ca.items()}
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> "RunMetadata":
        return cls.from_dict(json.loads(s))

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())

    @classmethod
    def load(cls, path) -> "RunMetadata":
        return cls.from_json(Path(path).read_text())


# ------------------------------------------------------------------
# Convenience: a small timer helper
# ------------------------------------------------------------------


class WallClock:
    """Context manager that measures wall-clock time."""

    def __init__(self):
        self.elapsed = 0.0
        self._start: Optional[float] = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - (self._start or time.time())
        return False
