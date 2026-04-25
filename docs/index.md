# agent-urban-planning

Open-source Python library for **agent-based urban planning simulation** with
closed-form, hybrid, and full-LLM decision engines. Companion to the
NeurIPS Datasets & Benchmarks 2026 paper.

::::{grid} 2
:::{grid-item-card} 🚀 Quickstart
:link: tutorials/01_quickstart
:link-type: doc

`pip install agent-urban-planning` and run a 5-variant smoke test in
under 10 seconds.
:::

:::{grid-item-card} 📊 Berlin replication
:link: tutorials/04_berlin_replication
:link-type: doc

Reproduce the paper's V1-V5 Berlin baseline + East-West Express shock.
:::

:::{grid-item-card} 🤖 Full LLM (V5)
:link: tutorials/03_full_llm_v5
:link-type: doc

Deep-dive on the score-all-96 + rebalance + stage-2 cap pattern (paper
headline).
:::

:::{grid-item-card} 🔧 Custom engines
:link: tutorials/02_custom_decision_engine
:link-type: doc

Subclass {class}`agent_urban_planning.DecisionEngine` for research
extensions.
:::
::::

## What it is

A modular agent-based modeling framework for spatial-equilibrium urban
policy simulation. Five reference decision-engine variants ship as
first-class API classes, configurable via kwargs:

| Paper variant | One-line description | API call |
|---|---|---|
| **V1** Baseline-softmax | Closed-form Cobb-Douglas + Fréchet softmax | `aup.UtilityEngine(mode="softmax")` |
| **V2** Baseline-ABM argmax | ABM with Fréchet idiosyncratic shocks | `aup.UtilityEngine(mode="argmax", noise="frechet")` |
| **V3** Normal-ABM argmax | ABM with Gaussian shocks | `aup.UtilityEngine(mode="argmax", noise="normal")` |
| **V4** Hybrid-ABM | LLM-elicited preferences + closed-form choice | `aup.HybridDecisionEngine(elicitor=...)` |
| **V5** LLM-ABM (paper headline) | Full LLM-as-decision-maker, score-all-96 | `aup.LLMDecisionEngine(response_format="score_all", rebalance_instruction=True, stage2_top_k_residences=10)` |

## Citation

If you use this software in your research, please cite the accompanying
paper. See {file}`CITATION.cff` in the repo root for full metadata.

```{toctree}
:hidden:
:caption: Tutorials

tutorials/01_quickstart
tutorials/02_custom_decision_engine
tutorials/03_full_llm_v5
tutorials/04_berlin_replication
```

```{toctree}
:hidden:
:caption: Concepts

concepts/architecture
concepts/decision_engines
concepts/llm_integration
```

```{toctree}
:hidden:
:caption: Reproducibility

reproducibility/berlin_v1_v5
reproducibility/shock_analysis
```

```{toctree}
:hidden:
:caption: API reference

api/index
```
