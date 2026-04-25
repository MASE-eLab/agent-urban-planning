"""Pluggable decision engines.

Three first-class API classes (configurable via constructor kwargs to reproduce
V1-V5 paper variants):

  * :class:`UtilityEngine` — closed-form Cobb-Douglas + Fréchet (V1, V2, V3).
  * :class:`HybridDecisionEngine` — LLM-elicited preferences + closed-form
    choice (V4).
  * :class:`LLMDecisionEngine` — full LLM-as-decision-maker (V5.0, V5).

Phase 3 of the package extraction populates this subpackage; until then it is
intentionally empty.
"""
