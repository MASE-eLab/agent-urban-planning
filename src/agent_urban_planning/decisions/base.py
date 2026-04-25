"""Decision engine interface.

A `DecisionEngine` chooses locations for individual agents (`decide`) or whole
populations (`decide_batch`). The batch interface lets engines optimize
across many agents at once — for example, the LLM engine batches calls via
asyncio for parallelism. UtilityEngine implements `decide_batch` as a simple
loop because per-agent computation is already cheap.

Engines may optionally accept an `LLMCallCache` via `set_cache(cache)` so
that the market loop can hand them a per-clearing cache. Engines that do
not need a cache simply ignore the call.

A choice is returned as a `LocationChoice` with both `residence` and
`workplace` zone IDs. For Singapore scenarios where workplace is fixed as
an agent attribute, engines set `workplace = agent.job_location`. For
Berlin/Ahlfeldt scenarios, the engine genuinely optimizes over joint
(R, W) pairs. The legacy `ZoneChoice` name remains importable as a
backward-compatible thin subclass that accepts `zone_name=` in its
constructor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from agent_urban_planning.core.agents import Agent
from agent_urban_planning.core.environment import Environment


@dataclass
class LocationChoice:
    """An agent's chosen (residence, workplace) pair plus diagnostic utilities.

    The canonical output of any :class:`DecisionEngine`. For Singapore-style
    scenarios where workplace is fixed as an agent attribute, ``workplace``
    is set to ``agent.job_location``; for Berlin-style scenarios where
    workplace is a genuine choice output, it reflects the engine's
    decision. The diagnostic ``zone_utilities`` mapping supports both flat
    (single-zone) and namespaced (joint R/W) layouts.

    Attributes:
        residence: Zone ID where the agent lives (their chosen residence).
        workplace: Zone ID where the agent works. For Singapore-style
            scenarios equals ``agent.job_location``; for Berlin-style
            scenarios it is a genuine choice output.
        utility: Realized utility at the chosen pair.
        zone_utilities: Per-zone diagnostic utilities. In single-zone
            engines, keys are residence zone IDs. In joint-choice engines,
            keys may optionally be namespaced ``"R:<zone>"`` /
            ``"W:<zone>"``.

    Examples:
        >>> from agent_urban_planning import LocationChoice
        >>> choice = LocationChoice(
        ...     residence="Mitte",
        ...     workplace="Charlottenburg",
        ...     utility=2.71,
        ...     zone_utilities={"R:Mitte": 1.4, "W:Charlottenburg": 1.31},
        ... )
        >>> choice.zone_name  # backward-compat alias for residence
        'Mitte'
    """

    residence: str
    workplace: str
    utility: float
    zone_utilities: dict[str, float] = field(default_factory=dict)

    @property
    def zone_name(self) -> str:
        """Backward-compatible alias for ``residence``.

        Existing code that reads ``choice.zone_name`` continues to work
        against the new dataclass. Prefer ``residence`` in new code.
        """
        return self.residence


class ZoneChoice(LocationChoice):
    """Backward-compatible alias for :class:`LocationChoice`.

    Accepts the legacy keyword-argument style ``ZoneChoice(zone_name=...,
    utility=..., zone_utilities=...)`` and maps ``zone_name`` to both
    ``residence`` and ``workplace`` (since Singapore scenarios conflated
    them). Also accepts the new ``residence=`` / ``workplace=`` keywords
    for forward compatibility. New code should use
    :class:`LocationChoice` directly; this class remains so that the
    large existing codebase (utility engines, LLM engines, clustering,
    cache serialization) continues to work unchanged.

    Examples:
        >>> from agent_urban_planning import ZoneChoice
        >>> legacy = ZoneChoice(zone_name="Mitte", utility=1.5)
        >>> legacy.residence == legacy.workplace == "Mitte"
        True
        >>> # New code SHOULD use the explicit residence/workplace kwargs:
        >>> modern = ZoneChoice(residence="Mitte", workplace="Charlottenburg",
        ...                     utility=2.1)
        >>> modern.workplace
        'Charlottenburg'
    """

    def __init__(
        self,
        zone_name: Optional[str] = None,
        utility: float = 0.0,
        zone_utilities: Optional[dict[str, float]] = None,
        *,
        residence: Optional[str] = None,
        workplace: Optional[str] = None,
    ):
        r = residence if residence is not None else zone_name
        if r is None:
            raise TypeError(
                "ZoneChoice requires either zone_name (legacy) or residence (new)"
            )
        w = workplace if workplace is not None else r
        super().__init__(
            residence=r,
            workplace=w,
            utility=utility,
            zone_utilities=zone_utilities if zone_utilities is not None else {},
        )


@runtime_checkable
class DecisionEngine(Protocol):
    """Protocol every decision engine implements.

    Implementations MUST provide :meth:`decide`. They MAY override
    :meth:`decide_batch` for optimized population-level processing;
    otherwise the default implementation calls :meth:`decide` in a
    loop. They MAY override :meth:`set_cache` to accept an optional
    ``LLMCallCache`` from the market loop. The four canonical
    library engines (closed-form, hybrid, LLM, clusterized) all
    satisfy this protocol.

    Examples:
        >>> import agent_urban_planning as aup
        >>> # Any of the public engines can be used wherever
        >>> # `DecisionEngine` is required:
        >>> # engine: aup.DecisionEngine = aup.UtilityEngine(params)
    """

    def decide(
        self,
        agent: Agent,
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> LocationChoice:
        """Choose a location for one agent given the environment and current prices.

        Implementations decide on the agent's residence (and workplace,
        if joint) given the current state of the market. The choice is
        returned as a :class:`LocationChoice` carrying both selected
        zones, the realized utility, and per-zone diagnostic utilities.

        Args:
            agent: The :class:`Agent` whose decision is being made.
            environment: The :class:`Environment` in which the agent
                operates.
            zone_options: Allowed residence (and, where joint,
                workplace) zones at this market step.
            prices: Mapping ``zone -> current price`` seen by the agent.

        Returns:
            A :class:`LocationChoice` describing the agent's chosen
            (residence, workplace) pair and realized utility.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # engine = aup.UtilityEngine(params)
            >>> # choice = engine.decide(agent, env, zones, prices)  # doctest: +SKIP
        """
        ...

    def decide_batch(
        self,
        agents: list[Agent],
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> list[LocationChoice]:
        """Choose locations for many agents at once.

        Default implementation loops over :meth:`decide` so legacy
        engines keep working. Override for optimized batching (e.g.
        async LLM calls or vectorized per-agent matrix evaluations).

        Args:
            agents: List of :class:`Agent` instances to decide for.
            environment: The :class:`Environment` they operate in.
            zone_options: Allowed zones at this step.
            prices: Mapping ``zone -> current price``.

        Returns:
            List of :class:`LocationChoice`, one per input agent and
            in the same order.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # choices = engine.decide_batch(list(pop), env, zones, prices)
        """
        ...

    def set_cache(self, cache) -> None:
        """Inject an optional cache from the market loop.

        Engines that need not cache anything ignore the call (the
        default is no-op). LLM-backed engines typically wire the cache
        into their request layer here.

        Args:
            cache: An :class:`agent_urban_planning.llm.LLMCallCache`
                or compatible object.

        Returns:
            None.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # engine.set_cache(cache)
        """
        ...


def default_decide_batch(
    engine,
    agents: list[Agent],
    environment: Environment,
    zone_options: list[str],
    prices: dict[str, float],
) -> list[LocationChoice]:
    """Reusable default implementation: call `decide` once per agent.

    Engines that subclass this module's classes (or just use it as a mixin)
    can call this helper from their own `decide_batch` to fall back to the
    sequential path.
    """
    return [engine.decide(agent, environment, zone_options, prices) for agent in agents]


class BaseDecisionEngine:
    """Optional base class providing default `decide_batch` and `set_cache`.

    Concrete engines may inherit from this to get the loop-based batch
    implementation for free. Engines that need custom batching override
    `decide_batch`.
    """

    def decide(
        self,
        agent: Agent,
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> LocationChoice:
        raise NotImplementedError

    def decide_batch(
        self,
        agents: list[Agent],
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> list[LocationChoice]:
        return default_decide_batch(self, agents, environment, zone_options, prices)

    def set_cache(self, cache) -> None:
        # Default: ignore the cache
        return None
