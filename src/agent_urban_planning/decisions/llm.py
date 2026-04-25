"""Unified ``LLMDecisionEngine`` API class — V5.0 / V5.4 via kwargs.

Full LLM-as-decision-maker hierarchical engine. The headline contribution
of the paper. Configure via constructor kwargs to reproduce V5.0 (top-5
ranking) or V5.4 (score-all-96 with rebalance instruction and stage-2
top-K cap).

Internally delegates to :class:`AhlfeldtHierarchicalLLMEngine`, ported
from the dev repo.

Example::

    import agent_urban_planning as aup

    # V5.4 reproduction (paper's headline LLM-ABM):
    engine = aup.LLMDecisionEngine(
        params=scenario.ahlfeldt_params,
        llm_client=aup.llm.CodexCliClient(),
        response_format="score_all",
        rebalance_instruction=True,
        stage2_top_k_residences=10,
        cluster_k=50,
        num_agents=1_000_000,
        seed=42,
    )
"""
from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from agent_urban_planning.decisions.ahlfeldt_hierarchical_llm_engine import (
    AhlfeldtHierarchicalLLMEngine,
)


_VALID_RESPONSE_FORMATS = ("top5", "score_all")


def _select_prompt_and_validator(
    response_format: str,
    rebalance_instruction: bool,
) -> tuple[Optional[Callable], Optional[Callable]]:
    """Choose (prompt_builder, response_validator) callables for the given config.

    Returns ``(None, None)`` for the default case so the underlying engine
    can pick its own defaults.
    """
    from agent_urban_planning.llm.prompts.hierarchical import (
        build_stage1_prompt,
        build_stage1_prompt_score_all,
        build_stage1_prompt_rank_rebalanced,
        validate_top5_response,
        validate_all_scores_response,
    )

    if response_format == "score_all":
        return build_stage1_prompt_score_all, validate_all_scores_response
    if response_format == "top5":
        if rebalance_instruction:
            return build_stage1_prompt_rank_rebalanced, validate_top5_response
        return build_stage1_prompt, validate_top5_response
    raise ValueError(
        f"response_format={response_format!r} is not valid; expected one of "
        f"{_VALID_RESPONSE_FORMATS}"
    )


class LLMDecisionEngine:
    """Full LLM-as-decision-maker hierarchical engine (V5.0, V5.4).

    The LLM is queried per agent cluster per market iteration to make
    discrete location decisions directly: stage 1 selects a residence,
    stage 2 selects a workplace conditional on residence. This is the
    paper's headline contribution and the ``aup`` library's core
    extensibility point.

    Configure via constructor kwargs to reproduce V5.0 or V5.4 from the
    paper:

    +---------+--------------------------+-----------------------------+
    | Variant | response_format          | rebalance + stage2 cap      |
    +=========+==========================+=============================+
    | V5.0    | ``"top5"``                | Default (no rebalance)      |
    +---------+--------------------------+-----------------------------+
    | V5.4    | ``"score_all"``           | ``rebalance_instruction=    |
    |         |                          | True``, ``stage2_top_k_     |
    |         |                          | residences=10`` (paper      |
    |         |                          | headline)                   |
    +---------+--------------------------+-----------------------------+

    Args:
        params: ``AhlfeldtParams`` instance carrying structural
            elasticities.
        llm_client: An :class:`agent_urban_planning.llm.LLMClient` instance
            (or any object with a ``.complete(user, system="")`` method).
        response_format: ``"top5"`` for V5.0 (LLM emits top-5 ranking).
            ``"score_all"`` for V5.4 (LLM scores all N zones; the paper's
            headline). Default ``"score_all"`` (V5.4).
        rebalance_instruction: If ``True``, the stage-1 prompt includes
            an explicit "weight affordability ≥ amenity" instruction.
            Validated in the V5.3 ablation. Default ``False`` (the V5.4
            paper run sets this to ``True``).
        stage2_top_k_residences: When set to an int, stage-2 fan-out is
            capped at the top-K residences per cluster (by stage-1 score),
            preventing the cost blowup from score-all stage-1 producing
            96 residences per cluster. ``None`` disables the cap.
            Default ``None``; the V5.4 paper run uses ``10``.
        prompt_builder: Optional override for the stage-1 prompt builder
            callable. When provided, takes precedence over the
            ``response_format`` selection.
        response_validator: Optional override for the response validator
            callable. When provided, takes precedence over the format
            selection.
        **kwargs: Forwarded to :class:`AhlfeldtHierarchicalLLMEngine`.
            Common kwargs: ``cluster_k`` (default 50), ``clustering_algo``,
            ``zone_name_map``, ``cache_dir``, ``softmax_T``, ``num_agents``,
            ``batch_size``, ``seed``, ``llm_concurrency``,
            ``progress_callback``, ``max_retries``.

    Raises:
        ValueError: If ``response_format`` is not one of ``{"top5",
            "score_all"}``, or if ``llm_client`` is None.

    Examples:
        V5.4 reproduction (paper's headline LLM-ABM)::

            >>> import agent_urban_planning as aup
            >>> engine = aup.LLMDecisionEngine(
            ...     params=scenario.ahlfeldt_params,
            ...     llm_client=aup.llm.CodexCliClient(),
            ...     response_format="score_all",
            ...     rebalance_instruction=True,
            ...     stage2_top_k_residences=10,
            ...     cluster_k=50,
            ...     num_agents=1_000_000,
            ...     seed=42,
            ... )

        V5.0 reproduction (legacy top-5 hierarchical, no rebalance)::

            >>> engine = aup.LLMDecisionEngine(
            ...     params=scenario.ahlfeldt_params,
            ...     llm_client=aup.llm.CodexCliClient(),
            ...     response_format="top5",
            ...     rebalance_instruction=False,
            ...     cluster_k=50,
            ... )

        Custom prompt builder (research extensibility)::

            >>> def my_prompt(persona, zones_info, *, prompt_version):
            ...     return ("system", f"persona: {persona}; rank top 3.")
            >>> engine = aup.LLMDecisionEngine(
            ...     params=scenario.ahlfeldt_params,
            ...     llm_client=aup.llm.CodexCliClient(),
            ...     prompt_builder=my_prompt,
            ...     response_validator=my_validator,
            ... )

    Notes:
        The V5.4 score-all configuration produces 96-zone stage-1
        distributions, which would cause stage-2 fan-out to explode
        (50 clusters × 96 residences = 4800 stage-2 LLM calls per
        market iteration). The ``stage2_top_k_residences`` kwarg caps
        this at top-K residences (default for V5.4 production: 10);
        residences outside the top-K fall back to the sampler's
        uniform-workplace default in
        :class:`AhlfeldtHierarchicalLLMEngine`.

    See Also:
        :class:`agent_urban_planning.UtilityEngine` — closed-form V1/V2/V3
        baselines.
        :class:`agent_urban_planning.HybridDecisionEngine` — V4-B
        (LLM elicits preferences, closed-form choice).

    References:
        Ahlfeldt, G. M., Redding, S. J., Sturm, D. M., Wolf, N. (2015).
        The economics of density: Evidence from the Berlin Wall.
        *Econometrica*, 83(6), 2127-2189.
    """

    def __init__(
        self,
        params: Any,
        llm_client: Any,
        *,
        response_format: Literal["top5", "score_all"] = "score_all",
        rebalance_instruction: bool = False,
        stage2_top_k_residences: Optional[int] = None,
        prompt_builder: Optional[Callable] = None,
        response_validator: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        if llm_client is None:
            raise ValueError(
                "LLMDecisionEngine requires an llm_client; got None. "
                "See aup.llm for available client wrappers."
            )
        if response_format not in _VALID_RESPONSE_FORMATS:
            raise ValueError(
                f"response_format={response_format!r} is not valid; "
                f"expected one of {_VALID_RESPONSE_FORMATS}"
            )

        # Determine which prompt builder + validator to use.
        if prompt_builder is None or response_validator is None:
            sel_pb, sel_rv = _select_prompt_and_validator(
                response_format, rebalance_instruction,
            )
            if prompt_builder is None:
                prompt_builder = sel_pb
            if response_validator is None:
                response_validator = sel_rv

        self._response_format = response_format
        self._rebalance_instruction = rebalance_instruction

        self._impl = AhlfeldtHierarchicalLLMEngine(
            params,
            llm_client=llm_client,
            prompt_builder_stage1=prompt_builder,
            response_validator_stage1=response_validator,
            stage2_top_k_residences=stage2_top_k_residences,
            **kwargs,
        )

    @property
    def response_format(self) -> str:
        """The configured response format: ``"top5"`` or ``"score_all"``."""
        return self._response_format

    @property
    def rebalance_instruction(self) -> bool:
        """Whether the stage-1 prompt includes the V5.3-validated rebalance instruction."""
        return self._rebalance_instruction

    def decide_batch(self, *args: Any, **kwargs: Any) -> Any:
        """Forward to the underlying implementation's ``decide_batch``."""
        return self._impl.decide_batch(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._impl, name)

    def __repr__(self) -> str:
        return (
            f"LLMDecisionEngine(response_format={self._response_format!r}, "
            f"rebalance_instruction={self._rebalance_instruction}, "
            f"_impl={type(self._impl).__name__})"
        )
