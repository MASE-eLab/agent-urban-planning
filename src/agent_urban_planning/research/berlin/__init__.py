"""Berlin replication helpers (Ahlfeldt et al. 2015).

This subpackage contains paper-specific implementations:
  * Ahlfeldt-style closed-form market clearing (Cobb-Douglas + Fréchet).
  * East-West Express τ shock application (Route-C min-of-paths).
  * Block-to-zone aggregation utilities.
  * Calibration helpers.

These modules are imported by the :class:`agent_urban_planning.UtilityEngine`,
:class:`HybridDecisionEngine`, and :class:`LLMDecisionEngine` public classes
to provide Berlin-specific behavior. End users typically do not import them
directly.
"""
