API reference
=============

.. currentmodule:: agent_urban_planning

The library's public API. All names listed here are importable directly
from the top-level ``agent_urban_planning`` package (or its short alias ``aup``)::

    import agent_urban_planning as aup

Core simulation
---------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   SimulationEngine
   Environment
   Zone
   Agent
   AgentPopulation
   PreferenceWeights
   HousingMarket
   AhlfeldtMarket
   MarketResult
   WelfareMetrics
   SimulationResults
   AgentResult
   RunMetadata

.. autosummary::
   :toctree: _autosummary

   persona_summary
   compute_metrics

Decision engines (public API)
-----------------------------

The three first-class decision-engine classes — configurable via
constructor kwargs to reproduce V1-V5.4 from the paper:

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   DecisionEngine
   UtilityEngine
   HybridDecisionEngine
   LLMDecisionEngine
   LocationChoice
   ZoneChoice

Decision engines (paper-internal — advanced)
--------------------------------------------

The underlying paper-internal classes that the public API delegates to.
Most users should not need to import these directly.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   AhlfeldtUtilityEngine
   AhlfeldtABMEngine
   AhlfeldtArgmaxHybridEngine
   AhlfeldtHierarchicalLLMEngine
