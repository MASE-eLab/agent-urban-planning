"""Prompt templates + response validators for LLM-as-decision-maker engines.

Includes the V5 score-all-96 hierarchical prompt (paper headline) plus a
legacy top-5 ranking variant, both with their JSON-schema validators.
Selected by the ``response_format`` kwarg of
:class:`agent_urban_planning.LLMDecisionEngine`.
"""
