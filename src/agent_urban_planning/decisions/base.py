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

    Fields:
        residence: zone ID where the agent lives.
        workplace: zone ID where the agent works. For Singapore-style
            scenarios this equals ``agent.job_location``; for Berlin-style
            scenarios it is a genuine choice output.
        utility: realized utility at the chosen pair.
        zone_utilities: per-zone diagnostic utilities. In single-zone
            engines keys are residence zone IDs. In joint-choice engines
            keys may optionally be namespaced ``"R:<zone>"`` / ``"W:<zone>"``.
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
    them). Also accepts the new ``residence=``/``workplace=`` keywords
    for forward compatibility.

    New code SHOULD use :class:`LocationChoice` directly. This class
    remains only so that the large existing codebase (utility engines,
    LLM engines, clustering, cache serialization) continues to work
    unchanged.
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
    """Interface for agent decision engines.

    Implementations MUST provide `decide`. They MAY override `decide_batch`
    for optimized population-level processing; otherwise the default
    implementation calls `decide` in a loop. They MAY override `set_cache`
    to accept an optional LLMCallCache from the market loop.
    """

    def decide(
        self,
        agent: Agent,
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> LocationChoice:
        """Choose a location for the agent given the environment and current prices."""
        ...

    def decide_batch(
        self,
        agents: list[Agent],
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> list[LocationChoice]:
        """Choose locations for many agents at once.

        Default implementation loops over `decide` so legacy engines keep
        working. Override for optimized batching (e.g. async LLM calls).
        """
        ...

    def set_cache(self, cache) -> None:
        """Inject an optional cache from the market loop. Default is no-op."""
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
