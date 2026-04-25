# Repo provenance

This repo (`agent-urban-planning`) is the publishable version of the simulator from the paper:

> **TBD — paper title** (NeurIPS Datasets & Benchmarks 2026).

Paper-development history, ablation experiments, intermediate artifacts, OpenSpec change-records, and one-off paper-internal scripts live in the development repo:

```
multi-agent-simulator   (private dev repo)
```

If you're reproducing the paper's results — **this is the right place**.
If you're reading the paper-development history (V5 ablations, openspec changes, etc.) — see the dev repo.

## Phase mapping

| Paper variant | Dev repo class | Public API |
|---|---|---|
| V1 (Baseline-softmax) | `simulator.decisions.AhlfeldtUtilityEngine` (mode='softmax') | `aup.UtilityEngine(mode='softmax')` |
| V2 (Baseline-ABM argmax) | `simulator.decisions.AhlfeldtABMEngine` (Fréchet) | `aup.UtilityEngine(mode='argmax', noise='frechet')` |
| V3 (Normal-ABM argmax) | `simulator.decisions.AhlfeldtABMEngine` (Normal) | `aup.UtilityEngine(mode='argmax', noise='normal')` |
| V4 (Hybrid-ABM) | `simulator.decisions.AhlfeldtArgmaxHybridEngine` | `aup.HybridDecisionEngine(...)` |
| V5 (LLM-ABM) | `simulator.decisions.AhlfeldtHierarchicalLLMEngine` (score-all + rebalance) | `aup.LLMDecisionEngine(response_format='score_all', rebalance_instruction=True, stage2_top_k_residences=10)` |

The dev repo's class names are paper-internal (`Ahlfeldt*`); the public API uses generic class names with kwargs to select variants. Internally, the public classes delegate to the same code.

## Key dev-repo references

- Final V5 baseline + shock CSVs: `output/berlin_v5_score_all_*/per_zone.csv`
- Comparison-moments table: `geo_graph_plot/results/table_berlin/`
- Choropleth figures: `geo_graph_plot/results/berlin_*_observed_and_log_change.png`
- Paper-development docs: `docs/berlin_v5_score_all.md`, `docs/berlin_v5_hierarchical.md`
- Submission-archive tag in dev repo: TBD (will be tagged at paper submission)
