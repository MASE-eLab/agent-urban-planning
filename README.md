# agent-urban-planning

[![PyPI](https://img.shields.io/pypi/v/agent-urban-planning.svg)](https://pypi.org/project/agent-urban-planning/)
[![Documentation Status](https://readthedocs.org/projects/agent-urban-planning/badge/?version=latest)](https://agent-urban-planning.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)

Open-source Python library for **agent-based urban planning simulation** with closed-form, hybrid, and full-LLM decision engines. Companion to the NeurIPS Datasets & Benchmarks 2026 paper:

> **TBD — paper title**

## Workflow

![workflow](figures/workflow.png)

The simulator is organized as a five-stage pipeline. **(1) Environment** — a city is described by zones, transportation graphs, land-use parameters, housing, amenities, and wages; an optional policy shock perturbs this state (e.g., a new rapid-transit line). **(2) Agents** — a population of heterogeneous household types (young professionals, families with kids, retirees, …) with demographic and preference attributes. **(3) Decision** — each agent picks a (residence, workplace) pair via one of three swappable engines: a closed-form `UtilityEngine` (V1–V3), a `HybridDecisionEngine` that elicits LLM-side preferences and resolves choice in closed form (V4-B), or a full `LLMDecisionEngine` that consults a language model end-to-end (V5.4 — the paper headline). **(4) Market & Allocation** — the macro layer aggregates choices, clears the housing/labor markets via tâtonnement (rents, wages, congestion), and allocates agents to zones. **(5) Equilibrium** — a fixed-point solver iterates 1→4 until prices and choices stop moving. Output is a bundle of welfare metrics, choropleth maps, and per-agent traces with a recorded seed for replay.

## What it is

A modular ABM framework for simulating spatial-equilibrium urban policies. Five reference decision-engine variants ship as first-class API classes, configurable via kwargs:

| Paper variant | One-line description | API call |
|---|---|---|
| **V1** Baseline-softmax | Closed-form Cobb-Douglas + Fréchet softmax | `aup.UtilityEngine(mode="softmax")` |
| **V2** Baseline-ABM argmax | ABM with Fréchet idiosyncratic shocks | `aup.UtilityEngine(mode="argmax", noise="frechet")` |
| **V3** Normal-ABM argmax | ABM with Gaussian shocks | `aup.UtilityEngine(mode="argmax", noise="normal")` |
| **V4-B** Hybrid-ABM | LLM-elicited preferences + closed-form choice | `aup.HybridDecisionEngine(elicitor=...)` |
| **V5.4** LLM-ABM (paper headline) | Full LLM-as-decision-maker, score-all-96 | `aup.LLMDecisionEngine(response_format="score_all", rebalance_instruction=True, stage2_top_k_residences=10)` |

## Quickstart

```bash
pip install agent-urban-planning
```

```python
import agent_urban_planning as aup
from agent_urban_planning.data.loaders import AhlfeldtParams

# Ahlfeldt (2015) Berlin structural parameters.
params = AhlfeldtParams(
    kappa_eps=0.0987, epsilon=6.6941,
    lambda_=0.071, delta=0.362,
    eta=0.155, rho=0.759,
)

# V1 — closed-form softmax (deterministic).
v1 = aup.UtilityEngine(params, mode="softmax")

# V5.4 — full LLM-as-decision-maker (paper headline).
v5_4 = aup.LLMDecisionEngine(
    params,
    llm_client=...,                     # any OpenAI-compatible client
    response_format="score_all",
    rebalance_instruction=True,
    stage2_top_k_residences=10,
    num_agents=10, batch_size=10, seed=42, cluster_k=2,
)
```

End-to-end runnable script: [`examples/01_quickstart_two_zone.py`](examples/01_quickstart_two_zone.py) — instantiates all five variants in <10 seconds.

## Berlin replication results

We benchmark all five variants on a 96-zone Berlin scenario (Ahlfeldt et al. 2015 calibration) under a hypothetical East–West Express rapid-transit shock (4 stations, 5-min between adjacent).

### Choropleth: log-change in housing prices Δlog Q under the shock

![Berlin Δlog Q](figures/berlin_dlogQ.png)

The structural family (V1, V2, V4-B) produces a visibly *focal* response — price effects concentrate at the four station catchments. **LLM-ABM (V5.4)** instead spreads the response broadly across the city, picking up the gradient-flattening + agglomeration effects that exogenous-productivity models forbid.

### Choropleth: log-change in wages Δlog w under the shock

![Berlin Δlog w](figures/berlin_dlogW.png)

The wage maps tell the same story at the workplace side: structural variants forbid the agglomeration channel by construction (productivity `A_i` is exogenous), while LLM-ABM produces a coherent compensating-differential pattern via its training-data priors on urban economics.

### Cross-variant moments

Distribution-shape moments across the 96 zones (baseline → shock):

| Variant | μ ΔlogQ | σ ΔlogQ | p95 \|ΔlogQ\| | ΔY% | Δ Q̄ | Δ⟨U⟩ |
|---|---:|---:|---:|---:|---:|---:|
| V1   Baseline-softmax     | +0.0004 | 0.0070 | 0.0083 | +0.0007 | +0.0008 | −0.0032 |
| V2   Baseline-ABM argmax  | +0.0004 | 0.0071 | 0.0080 | −0.0000 | +0.0007 | −0.0031 |
| V3   Normal-ABM argmax    | +0.0004 | 0.0042 | 0.0048 | +0.0132 | +0.0001 | −0.0027 |
| V4-B Hybrid-ABM           | +0.0004 | 0.0068 | 0.0077 | +0.0002 | +0.0007 | −0.0031 |
| **V5.4 LLM-ABM**          | **+0.0016** | **0.0083** | **0.0172** | **+0.0299** | **+0.0111** | **−0.0056** |

Full table + interpretation: [`figures/comparison_moments.csv`](figures/comparison_moments.csv).

**Three takeaways.**

1. **The structural family agrees.** V1, V2, V4-B (and V3 modulo Gaussian-vs-Fréchet tail differences) cluster tightly on `μ ≈ +0.0004`, `σ ≈ 0.007`, `Δ⟨U⟩ ≈ −0.0031`. They share the Cobb-Douglas + Fréchet architecture; the noise model and clustering wash out at zone-level moments. **Architecturally distinct simulators give the same answer to a small policy shock — a useful sanity check for the structural family.**
2. **LLM-ABM is qualitatively different.** Its mean log-change in housing price is 4× the structural consensus; spread (σ, p95) is 1.2–2× larger; mean rent change is 14× larger; aggregate productivity gain (+3%) is the only sizable positive value. Welfare drop is ~2× the structural family's.
3. **All five agree on welfare direction.** Δ⟨U⟩ < 0 across the board under the Baseline-softmax welfare ruler — the shock is a (small) net welfare loss. The mechanisms differ: structural variants attribute the drop to commute/wage compensation; LLM-ABM picks up the same direction at ~2× magnitude because its equilibrium reroutes more agents to lower-utility configurations under the structural ruler's lens.

The full paper §6 discusses why LLM-ABM diverges (gradient-flattening + agglomeration, both forbidden in exogenous-productivity structural models).

## Reproducibility tiers

| Tier | What | Wall-clock | Prerequisites |
|---|---|---|---|
| **1** | `pip install agent-urban-planning` + `import` | <30 s | Python 3.9+ |
| **2** | `examples/01_quickstart_two_zone.py` | <10 s | Tier 1 |
| **3** | `examples/03_berlin_replication/run_v1_softmax.py` | ~3 hr | Tier 2 + bundled Berlin data (in git, see `data/README.md`) |
| **4** | `examples/03_berlin_replication/run_v5_4_score_all.py` | ~10 hr | Tier 3 + LLM provider credits |

See [`docs/reproducibility/berlin_v1_v5_4.md`](docs/reproducibility/berlin_v1_v5_4.md) for full details.

## Documentation

Comprehensive API reference, tutorials, and concept guides:

→ <https://agent-urban-planning.readthedocs.io>

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
├── examples/
│   ├── 01_quickstart_two_zone.py   All 5 variants in <60 LOC, <10s wall-clock
│   └── 03_berlin_replication/      End-to-end V1–V5.4 Berlin runs
├── data/                           Bundled Berlin + Singapore data (git, not PyPI sdist)
├── docs/                           Sphinx documentation source
├── figures/                        README assets (workflow, choropleths, tables)
└── pyproject.toml
```

## Citation

If you use this software in your research, please cite the accompanying paper (see [`CITATION.cff`](CITATION.cff)).

## License

[MIT License](LICENSE) © 2026 The agent-urban-planning Authors

## Acknowledgements

The Berlin replication is built on the Ahlfeldt, Redding, Sturm, Wolf (2015) "The Economics of Density: Evidence from the Berlin Wall" *Econometrica* 83(6) data and methodology. We are grateful for the authors' public release of the replication package; the data files in `data/berlin/` and `data/shapefiles/` derive from that release. See `data/README.md` for full attribution.
