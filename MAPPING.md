# Public-API mapping

This file records the relationship between the public-facing API class names and the variant naming used in the accompanying paper.

> **TBD — paper title** (NeurIPS Datasets & Benchmarks 2026).

## Variant table

| Paper variant | Public API |
|---|---|
| V1 (Baseline-softmax) | `aup.UtilityEngine(mode='softmax')` |
| V2 (Baseline-ABM argmax) | `aup.UtilityEngine(mode='argmax', noise='frechet')` |
| V3 (Normal-ABM argmax) | `aup.UtilityEngine(mode='argmax', noise='normal')` |
| V4 (Hybrid-ABM) | `aup.HybridDecisionEngine(...)` |
| V5 (LLM-ABM) | `aup.LLMDecisionEngine(response_format='score_all', rebalance_instruction=True, stage2_top_k_residences=10)` |

The public-API classes use generic names with kwargs to select variants. The kwargs are documented in each class docstring and in the API documentation.
