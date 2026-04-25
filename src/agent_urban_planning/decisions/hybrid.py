"""Unified ``HybridDecisionEngine`` API class — V4-B via kwargs.

Implements the V4-B pattern: an LLM elicits per-agent preference weights
(β, κ); a closed-form mixed-logit then computes discrete zone choice.
The library exposes this as a single first-class API class so that user
code instantiates one symbol regardless of the underlying implementation.

Internally delegates to :class:`AhlfeldtArgmaxHybridEngine`, ported from
the dev repo.

Example::

    import agent_urban_planning as aup

    engine = aup.HybridDecisionEngine(
        params=scenario.ahlfeldt_params,
        llm_client=aup.llm.CodexCliClient(),
        cluster_k=50,
        num_agents=1_000_000,
        seed=42,
    )
"""
from __future__ import annotations

from typing import Any

from agent_urban_planning.decisions.ahlfeldt_argmax_hybrid_engine import (
    AhlfeldtArgmaxHybridEngine,
)


class HybridDecisionEngine:
    """LLM-elicited preferences + closed-form mixed-logit choice (V4-B).

    The hybrid pattern: an LLM is queried *once per agent cluster* to
    elicit the cluster's per-agent preference weights (β: housing share;
    κ: commute disutility). The closed-form mixed-logit choice then
    computes discrete zone selection deterministically. This pattern
    keeps the LLM in the role of *parameter provider* rather than
    *decision maker* — much cheaper than the full-LLM V5 approach
    while still capturing demographic-driven preference heterogeneity.

    Args:
        params: An ``AhlfeldtParams`` instance carrying structural
            elasticities (``alpha``, ``beta``, ``epsilon``, ``kappa_eps``).
        llm_client: An :class:`agent_urban_planning.llm.LLMClient`
            instance (or anything with a ``.complete(user, system="")``
            method returning a string).
        **kwargs: Forwarded to the underlying
            :class:`AhlfeldtArgmaxHybridEngine`. Common kwargs:
            ``cluster_k`` (default 50), ``clustering_algo`` (default
            ``"kmeans"``), ``num_agents``, ``batch_size``, ``seed``,
            ``llm_concurrency``, ``cache_dir``.

    Raises:
        ValueError: If ``llm_client`` is None.

    Examples:
        V4-B reproduction with codex-cli::

            >>> import agent_urban_planning as aup
            >>> engine = aup.HybridDecisionEngine(
            ...     params=scenario.ahlfeldt_params,
            ...     llm_client=aup.llm.CodexCliClient(),
            ...     cluster_k=50,
            ...     num_agents=1_000_000,
            ...     seed=42,
            ... )
            >>> # sim = aup.SimulationEngine(scenario, agent_config, engine=engine)
            >>> # results = sim.run()

        V4-B with claude-code (alternate provider)::

            >>> engine = aup.HybridDecisionEngine(
            ...     params=scenario.ahlfeldt_params,
            ...     llm_client=aup.llm.ClaudeCodeClient(),
            ... )

    See Also:
        :class:`agent_urban_planning.UtilityEngine` — closed-form V1/V2/V3
        baselines (no LLM involvement).
        :class:`agent_urban_planning.LLMDecisionEngine` — V5.0/V5.4
        full-LLM-as-decision-maker pattern.
    """

    def __init__(
        self,
        params: Any,
        elicitor: Any = None,
        *,
        llm_client: Any = None,
        **kwargs: Any,
    ) -> None:
        # V4-B's underlying engine takes a high-level "elicitor" object that
        # wraps an LLM client + caching + per-type β/κ extraction. The simple
        # `llm_client` argument is accepted for API symmetry with
        # LLMDecisionEngine, but an elicitor must be provided for the engine
        # to actually call the LLM. Users who only have an llm_client should
        # construct an elicitor themselves; see the dev repo's
        # `AhlfeldtElicitor` for the reference implementation.
        if elicitor is None and llm_client is None:
            raise ValueError(
                "HybridDecisionEngine requires either an elicitor or an "
                "llm_client; got neither. The V4-B pattern uses an elicitor "
                "to extract per-type preference weights from the LLM. See "
                "aup.research.berlin for AhlfeldtElicitor (the reference "
                "implementation)."
            )
        if elicitor is None:
            raise NotImplementedError(
                "HybridDecisionEngine currently requires a pre-built elicitor "
                "object. Building one from a raw llm_client is supported in "
                "the dev repo via simulator.decisions.elicitation but not yet "
                "exposed in the public API. Pass `elicitor=` directly for now."
            )
        self._impl = AhlfeldtArgmaxHybridEngine(
            params, elicitor=elicitor, **kwargs,
        )

    def decide_batch(self, *args: Any, **kwargs: Any) -> Any:
        """Forward to the underlying implementation's ``decide_batch``."""
        return self._impl.decide_batch(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._impl, name)

    def __repr__(self) -> str:
        return f"HybridDecisionEngine(_impl={type(self._impl).__name__})"
