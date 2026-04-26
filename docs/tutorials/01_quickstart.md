# Quickstart

Goal: get from `pip install` to a running end-to-end simulation
pipeline in under a minute.

## Install

```bash
pip install agent-urban-planning
```

For the LLM-using variants (V4, V5), also install the LLM extras:

```bash
pip install "agent-urban-planning[llm]"
```

To reproduce the paper's Berlin numbers, you also need the bundled
data, which means a `git clone` rather than `pip install`:

```bash
git clone https://github.com/MASE-eLab/agent-urban-planning.git
cd agent-urban-planning
pip install -e ".[llm,plot,berlin]"
```

## Run the pipeline walkthrough

The {file}`examples/01_quickstart.py` script walks through the five
workflow stages in the paper's Figure 1 — Environment → Agents →
Decision → Market → Equilibrium — and runs a single baseline
simulation with V1 (closed-form Cobb-Douglas + Fréchet softmax).

```bash
python examples/01_quickstart.py
```

Expected output (numbers vary slightly with NumPy / BLAS version):

```
agent-urban-planning version: 0.1.0

[1/5] Loading Environment from scenario YAML...
      zones: 96, scenario: berlin_2006_ortsteile
[2/5] Loading AgentPopulation (downsized for quickstart)...
      agent types: 1000
[3/5] Constructing DecisionEngine (V1 — closed-form softmax)...
      engine: UtilityEngine(mode='softmax', noise='frechet', _impl=AhlfeldtUtilityEngine)
[4/5] Running SimulationEngine baseline (5 market iterations)...
[5/5] Pipeline complete. Welfare metrics from this short run:
      avg utility:           -0.8340
      Gini (utility):         0.2300

      top 3 zones by clearing price (Q):
        z96_029: Q=0.3801, pop share=0.0000
        z96_003: Q=0.3706, pop share=0.0000
        z96_088: Q=0.3257, pop share=0.0008

This is an API-surface walkthrough; numbers are NOT paper-faithful
(only 5 market iterations, 1k agents, no L-override).
For paper-faithful V1-V5 reproduction at full scale, see
examples/02_berlin_replication/ (Tier 3+4).
```

If you see this output, your install is healthy and the pipeline
is wired correctly.

## What just happened

The script demonstrates the five-stage pipeline:

```python
import agent_urban_planning as aup
from agent_urban_planning.data.loaders import load_scenario, load_agents

# Stage 1 — Environment (zones, transport, fundamentals).
scenario = load_scenario("data/berlin/scenarios/berlin_2006_ortsteile.yaml")

# Stage 2 — Agents (heterogeneous household population).
agent_config = load_agents("data/berlin/agents/berlin_ortsteile_richer_10k.yaml")
agent_config.num_types = 1_000  # downsized for the quickstart

# Stage 3 — Decision engine (V1 closed-form softmax).
engine = aup.UtilityEngine(scenario.ahlfeldt_params, mode="softmax")

# Stage 4 — SimulationEngine + market clearing.
scenario.simulation.market_max_iterations = 5
sim = aup.SimulationEngine(scenario=scenario, agent_config=agent_config,
                           engine=engine, seed=42)
result = sim.run(policy=None)

# Stage 5 — Equilibrium results.
print(result.metrics.avg_utility, result.metrics.gini_coefficient)
```

The other 4 paper variants (V2, V3, V4, V5) just swap stage 3 — same
five-stage pipeline, different decision engine:

```python
# V2 — Baseline-ABM argmax (Fréchet shocks)
engine = aup.UtilityEngine(params, mode="argmax", noise="frechet")

# V3 — Normal-ABM argmax (Gaussian shocks)
engine = aup.UtilityEngine(params, mode="argmax", noise="normal")

# V4 — Hybrid-ABM (LLM-elicited preferences)
engine = aup.HybridDecisionEngine(params, elicitor=...)

# V5 — LLM-ABM (paper headline)
engine = aup.LLMDecisionEngine(
    params, llm_client=...,
    response_format="score_all",
    rebalance_instruction=True,
    stage2_top_k_residences=10,
)
```

Configure variants via constructor kwargs; no subclassing required.

## Next steps

- {doc}`02_custom_decision_engine` — write your own decision engine.
- {doc}`03_full_llm_v5` — deep-dive on the V5 LLM-ABM pattern.
- {doc}`04_berlin_replication` — reproduce the paper's V1-V5 results
  end-to-end with paper-faithful numbers.
