"""Unified ``UtilityEngine`` API class — V1 / V2 / V3 via kwargs.

This module exposes the public :class:`UtilityEngine` class. Configure via
constructor kwargs to reproduce the paper's three baseline-family variants:

  * ``mode='softmax'``                       → V1 (Baseline-softmax)
  * ``mode='argmax', noise='frechet'``       → V2 (Baseline-ABM argmax)
  * ``mode='argmax', noise='normal'``        → V3 (Normal-ABM argmax)

Internally delegates to :class:`AhlfeldtUtilityEngine` (softmax) or
:class:`AhlfeldtABMEngine` (argmax), both ported from the dev repo. The
public class is a thin compatibility/dispatch wrapper so that user code
references one symbol — ``aup.UtilityEngine`` — regardless of which
underlying implementation handles the call.

Example::

    import agent_urban_planning as aup

    # V1 reproduction:
    engine = aup.UtilityEngine(scenario.ahlfeldt_params, mode="softmax")

    # V2 reproduction:
    engine = aup.UtilityEngine(
        scenario.ahlfeldt_params, mode="argmax", noise="frechet",
    )

    # V3 reproduction:
    engine = aup.UtilityEngine(
        scenario.ahlfeldt_params, mode="argmax", noise="normal",
    )
"""
from __future__ import annotations

from typing import Any, Literal

from agent_urban_planning.decisions.ahlfeldt_abm_engine import AhlfeldtABMEngine
from agent_urban_planning.decisions.ahlfeldt_utility import AhlfeldtUtilityEngine


_VALID_MODES = ("softmax", "argmax")
_VALID_NOISE = ("frechet", "normal")


class UtilityEngine:
    """Closed-form Cobb-Douglas + Fréchet utility decision engine.

    Configure via constructor kwargs to reproduce V1 (Baseline-softmax),
    V2 (Baseline-ABM argmax with Fréchet noise), or V3 (Normal-ABM argmax
    with Gaussian noise) from the paper.

    Args:
        params: An ``AhlfeldtParams`` instance from
            :mod:`agent_urban_planning.data.loaders` carrying the model's
            structural elasticities (``alpha``, ``beta``, ``epsilon``,
            ``kappa_eps``, etc.).
        mode: ``"softmax"`` for the deterministic V1 pattern (closed-form
            softmax over Fréchet utility). ``"argmax"`` for the V2/V3 ABM
            pattern (per-agent draw + argmax). Default ``"softmax"``.
        noise: When ``mode="argmax"``, selects the per-agent shock
            distribution. ``"frechet"`` for V2; ``"normal"`` for V3.
            Ignored when ``mode="softmax"``. Default ``"frechet"``.
        **kwargs: Forwarded to the underlying implementation
            (:class:`AhlfeldtUtilityEngine` for softmax,
            :class:`AhlfeldtABMEngine` for argmax). Common kwargs:
            ``num_agents``, ``batch_size``, ``seed``, ``dtype``.

    Raises:
        ValueError: If ``mode`` is not one of ``{"softmax", "argmax"}``,
            or if ``noise`` is not one of ``{"frechet", "normal"}``.

    Examples:
        V1 reproduction (Baseline-softmax)::

            >>> import agent_urban_planning as aup
            >>> engine = aup.UtilityEngine(params, mode="softmax")
            >>> # Use as you would any DecisionEngine.
            >>> # sim = aup.SimulationEngine(scenario=sc, agent_config=ag, engine=engine)
            >>> # results = sim.run()

        V2 reproduction (Baseline-ABM argmax, Fréchet shocks)::

            >>> engine = aup.UtilityEngine(
            ...     params, mode="argmax", noise="frechet",
            ...     num_agents=1_000_000, seed=42,
            ... )

        V3 reproduction (Normal-ABM argmax, Gaussian shocks)::

            >>> engine = aup.UtilityEngine(
            ...     params, mode="argmax", noise="normal",
            ...     num_agents=1_000_000, seed=42,
            ... )

    Notes:
        Internally this class is a dispatch wrapper around two
        implementation classes (:class:`AhlfeldtUtilityEngine` and
        :class:`AhlfeldtABMEngine`). All other attribute and method
        access is forwarded transparently to the underlying implementation
        via ``__getattr__``, so any feature documented on those classes is
        usable on a ``UtilityEngine`` instance.

    See Also:
        :class:`agent_urban_planning.HybridDecisionEngine` — V4-B (LLM
        elicits per-agent preference weights, then closed-form choice).
        :class:`agent_urban_planning.LLMDecisionEngine` — V5.0 / V5.4
        (full LLM-as-decision-maker hierarchical engine).

    References:
        Ahlfeldt, G. M., Redding, S. J., Sturm, D. M., Wolf, N. (2015).
        The economics of density: Evidence from the Berlin Wall.
        *Econometrica*, 83(6), 2127-2189.
    """

    def __init__(
        self,
        params: Any,
        *,
        mode: Literal["softmax", "argmax"] = "softmax",
        noise: Literal["frechet", "normal"] = "frechet",
        **kwargs: Any,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode={mode!r} is not valid; expected one of {_VALID_MODES}"
            )
        if noise not in _VALID_NOISE:
            raise ValueError(
                f"noise={noise!r} is not valid; expected one of {_VALID_NOISE}"
            )
        self._mode = mode
        self._noise = noise
        if mode == "softmax":
            self._impl = AhlfeldtUtilityEngine(params, **kwargs)
        else:  # argmax
            self._impl = AhlfeldtABMEngine(
                params, shock_distribution=noise, **kwargs,
            )

    @property
    def mode(self) -> str:
        """The configured mode: ``"softmax"`` or ``"argmax"``."""
        return self._mode

    @property
    def noise(self) -> str:
        """The configured per-agent noise distribution (only relevant for argmax mode)."""
        return self._noise

    def decide_batch(self, *args: Any, **kwargs: Any) -> Any:
        """Forward to the underlying implementation's ``decide_batch``.

        Transparently forwards to either
        :meth:`AhlfeldtUtilityEngine.decide_batch` (softmax) or
        :meth:`AhlfeldtABMEngine.decide_batch` (argmax) depending on
        the configured mode.

        Args:
            *args: Positional arguments forwarded unchanged.
            **kwargs: Keyword arguments forwarded unchanged.

        Returns:
            List of :class:`agent_urban_planning.LocationChoice`, one
            per input agent and in the same order.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # engine = aup.UtilityEngine(params)
            >>> # choices = engine.decide_batch(agents, env, zones, prices)
        """
        return self._impl.decide_batch(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Forward unknown attribute access to the wrapped implementation.
        # Note: __getattr__ is only called when attribute lookup fails on
        # the instance itself, so this doesn't interfere with `_impl` etc.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._impl, name)

    def __repr__(self) -> str:
        return (
            f"UtilityEngine(mode={self._mode!r}, noise={self._noise!r}, "
            f"_impl={type(self._impl).__name__})"
        )
