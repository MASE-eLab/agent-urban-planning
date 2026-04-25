# agent-urban-planning

[![PyPI](https://img.shields.io/pypi/v/agent-urban-planning.svg)](https://pypi.org/project/agent-urban-planning/)
[![Documentation Status](https://readthedocs.org/projects/agent-urban-planning/badge/?version=latest)](https://agent-urban-planning.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)

Open-source Python library for **agent-based urban planning simulation** with closed-form, hybrid, and full-LLM decision engines. Companion to the NeurIPS Datasets & Benchmarks 2026 paper:

> **TBD — paper title**

## What it is

A modular ABM framework for simulating spatial-equilibrium urban policies. Five reference decision-engine variants ship as first-class API classes, configurable via kwargs:

| Paper variant | One-line description | API call |
|---|---|---|
| **V1** Baseline-softmax | Closed-form Cobb-Douglas + Fréchet softmax | `aup.UtilityEngine(mode="softmax")` |
| **V2** Baseline-ABM argmax | ABM with Fréchet idiosyncratic shocks | `aup.UtilityEngine(mode="argmax", noise="frechet")` |
| **V3** Normal-ABM argmax | ABM with Gaussian shocks | `aup.UtilityEngine(mode="argmax", noise="normal")` |
| **V4-B** Hybrid-ABM | LLM-elicited preferences + closed-form choice | `aup.HybridDecisionEngine(...)` |
| **V5.4** LLM-ABM (paper headline) | Full LLM-as-decision-maker, score-all-96 | `aup.LLMDecisionEngine(response_format="score_all", rebalance_instruction=True, stage2_top_k_residences=10)` |

## Quickstart

```bash
pip install agent-urban-planning
```

```python
import agent_urban_planning as aup

# Load a bundled scenario.
scenario = aup.data.builtin.load("singapore_real_v2")
agents = aup.data.builtin.load_agents("singapore_real_v2")

# Configure a decision engine and run.
engine = aup.UtilityEngine(scenario.ahlfeldt_params, mode="softmax")
sim = aup.SimulationEngine(scenario=scenario, agent_config=agents, engine=engine)
results = sim.run(policy=None)

# Inspect welfare metrics.
print(results.metrics.avg_utility)
print(results.metrics.gini_coefficient)
```

## Reproducibility tiers

| Tier | What | Wall-clock | Prerequisites |
|---|---|---|---|
| **1** | `pip install agent-urban-planning` + `import` | <30 s | Python 3.9+ |
| **2** | `examples/01_quickstart_two_zone.py` | <10 s | Tier 1 |
| **3** | `examples/03_berlin_replication/run_v1_softmax.py` | ~3 hr | Tier 2 + bundled Berlin data (in git) |
| **4** | `examples/03_berlin_replication/run_v5_4_score_all.py` | ~10 hr | Tier 3 + LLM provider credits |

See `docs/reproducibility/berlin_v1_v5_4.md` for full details.

## Documentation

Comprehensive API documentation, tutorials, and concept guides at:

→ [agent-urban-planning.readthedocs.io](https://agent-urban-planning.readthedocs.io)

## Repository layout

```
agent-urban-planning/
├── src/agent_urban_planning/      Library source
│   ├── core/                       Environment / Agents / Market / Engine / Metrics
│   ├── decisions/                  UtilityEngine / HybridDecisionEngine / LLMDecisionEngine
│   ├── data/                       Loaders + builtin scenarios
│   ├── analysis/                   Welfare + plotting
│   ├── llm/                        LLM client wrappers
│   └── research/                   Paper-specific helpers (Berlin)
├── examples/                       Runnable example scripts
│   ├── 01_quickstart_two_zone.py
│   ├── 02_singapore_policy_compare/
│   └── 03_berlin_replication/
├── tests/                          unit + integration + examples tests
├── docs/                           Sphinx documentation source
├── data/                           Bundled Berlin + Singapore data (git, not PyPI)
└── pyproject.toml
```

## Citation

If you use this software in your research, please cite the accompanying paper (see `CITATION.cff`).

## License

[MIT License](LICENSE) © 2026 The agent-urban-planning Authors

## Acknowledgements

The Berlin replication is built on the Ahlfeldt, Redding, Sturm, Wolf (2015) "The Economics of Density: Evidence from the Berlin Wall" *Econometrica* 83(6) data and methodology. We are grateful for the authors' public release of the replication package; the data files in `data/berlin/` and `data/shapefiles/` derive from that release. See `data/berlin/README.md` for full attribution.
