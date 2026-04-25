"""YAML scenario / agent loaders + bundled builtin scenarios.

Subpackages:
  * :mod:`agent_urban_planning.data.builtin` — small bundled scenarios for
    quick experimentation (Singapore, Berlin Ortsteile).
  * (root-level here) :func:`load_scenario`, :func:`load_agents` — load arbitrary
    YAML files matching the schemas in :mod:`agent_urban_planning.data.schemas`.

Phase 2 of the package extraction populates the loader functions; Phase 3 fills
in :mod:`builtin`.
"""
