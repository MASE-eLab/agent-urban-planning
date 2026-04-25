# Architecture

The library is organized around five core abstractions plus pluggable
decision engines.

```{mermaid}
flowchart TB
    Config[YAML Configs<br/>scenarios / agents / policies] --> Loader[load_scenario / load_agents]
    Loader --> Sim[SimulationEngine]
    Sim --> Env[Environment<br/>zones, transport, facilities]
    Sim --> Pop[AgentPopulation<br/>N agents with demographics]
    Sim --> Engine[DecisionEngine<br/>pluggable]
    Engine --> Util[UtilityEngine<br/>V1, V2, V3]
    Engine --> Hybrid[HybridDecisionEngine<br/>V4]
    Engine --> LLM[LLMDecisionEngine<br/>V5.0, V5]
    Sim --> Market[Market<br/>tâtonnement clearing]
    Market --> Results[Results<br/>metrics + traces + history]
```

## Core abstractions

### `Environment`
Spatial setting: zones, transport-time matrix, optional facilities. Mutable
via {meth}`Environment.apply_policy` for shock simulations.

### `AgentPopulation`
Heterogeneous household population sampled from demographic distributions.
N agents (typically 10,000-1,000,000), each with attributes (income, age,
household size, etc.) and a weight in [0, 1] summing to 1.

### `DecisionEngine` (pluggable)
Discrete-choice mechanism. Three first-class implementations:

- {class}`UtilityEngine` — closed-form Cobb-Douglas + Fréchet (V1/V2/V3)
- {class}`HybridDecisionEngine` — LLM elicits preferences + closed-form
  choice (V4)
- {class}`LLMDecisionEngine` — full LLM as decision maker (V5.0/V5)

Custom subclasses welcome (see
{doc}`/tutorials/02_custom_decision_engine`).

### `Market`
Tâtonnement clearing for housing prices and wages. Iterates until ΔQ and
Δw fall below configured thresholds. The {class}`AhlfeldtMarket` subclass
implements the two-market clearing pattern from Ahlfeldt et al. (2015) used
for the Berlin replication.

### `SimulationEngine`
Orchestrates Environment + AgentPopulation + DecisionEngine + Market for a
single run. Returns a {class}`SimulationResults` object with welfare
metrics, agent traces, price history, and run metadata.

## Data flow

```{mermaid}
sequenceDiagram
    participant User
    participant Sim as SimulationEngine
    participant Env as Environment
    participant Pop as AgentPopulation
    participant Eng as DecisionEngine
    participant Mkt as Market

    User->>Sim: run(policy)
    Sim->>Env: apply_policy(policy)
    Sim->>Pop: sample agents
    loop until convergence
        Sim->>Eng: decide_batch(agents, env, zones, prices)
        Eng-->>Sim: list[LocationChoice]
        Sim->>Mkt: update prices from demand
        Mkt-->>Sim: new prices
    end
    Sim->>User: SimulationResults
```

## Layered structure

```
┌─────────────────────────────────────────────────────────────────────┐
│  agent_urban_planning  (top-level package)                          │
│  ──────────────────────                                              │
│  __init__.py re-exports SimulationEngine, UtilityEngine, ...        │
│                                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────┐  ┌─────────┐ ┌──────┐ │
│  │   core/     │  │  decisions/  │  │ data/│  │analysis/│ │ llm/ │ │
│  │             │  │              │  │      │  │         │ │      │ │
│  │ environment │  │  base        │  │ load │  │ moments │ │client│ │
│  │ agents      │  │  utility     │  │ schemas│ │ plotting│ │async │ │
│  │ market      │  │  hybrid      │  │ builtin│ │ welfare │ │ cache│ │
│  │ engine      │  │  llm         │  │      │  │         │ │ prompts│
│  │ metrics     │  │  clustering  │  │      │  │         │ │      │ │
│  │ results     │  │              │  │      │  │         │ │      │ │
│  └─────────────┘  └──────────────┘  └──────┘  └─────────┘ └──────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  research/      Paper-specific helpers (advanced)               │ │
│  │   ├── berlin/   Ahlfeldt-style replication, railway shocks      │ │
│  │   └── singapore/ Singapore data fetcher                          │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

The `research/` subpackage is clearly labeled as paper-specific. The
headline public API in `agent_urban_planning` does not depend on anything
there.

## See also

- {doc}`decision_engines` — the V1-V5 variants in depth
- {doc}`llm_integration` — how to plug in different LLM providers
