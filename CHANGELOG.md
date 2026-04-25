# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-04-25

Initial public release accompanying the NeurIPS Datasets & Benchmarks 2026 submission.

### Added

- Core simulation framework: `Environment`, `AgentPopulation`, `Market`, `SimulationEngine`, `WelfareMetrics`, `Results`.
- Decision engines as first-class API classes:
  - `UtilityEngine` — closed-form Cobb-Douglas + Fréchet (V1, V2, V3 via `mode` and `noise` kwargs).
  - `HybridDecisionEngine` — V4-B pattern: LLM-elicited preference weights + closed-form choice.
  - `LLMDecisionEngine` — full LLM-as-decision-maker hierarchical engine with `response_format`, `rebalance_instruction`, `stage2_top_k_residences` kwargs (V5.0 / V5.4).
- LLM client abstraction: `CodexCliClient`, `ClaudeCodeClient`, `ZaiCodingClient`, `AnthropicClient`, `OpenAIClient`.
- YAML scenario + agent loader, schema dataclasses.
- Bundled Singapore + Berlin builtin scenarios (Berlin Ortsteile NPZ files in repo, excluded from PyPI sdist).
- Sphinx + Furo + 11-extension API documentation hosted on ReadTheDocs.
- 4 tutorials: quickstart, custom decision engine, full LLM V5, Berlin replication.
- 5 Berlin V1-V5.4 reproduction scripts under `examples/03_berlin_replication/`.
- Reproducibility tier documentation (Tier 1-4).
- MIT license, CITATION.cff, GitHub Actions CI / docs build / PyPI publish workflows.

### Notes

- Berlin shapefiles attributed to Ahlfeldt et al. (2015) "The Economics of Density: Evidence from the Berlin Wall" *Econometrica* 83(6); see `data/berlin/README.md`.
