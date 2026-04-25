# Decision engines

Five paper variants, three API classes, kwargs-driven configuration. This
page maps each paper variant to its API call and explains the conceptual
differences.

## The five variants

| # | Paper name | Mechanism | API call |
|---|---|---|---|
| V1 | Baseline-softmax | Closed-form softmax over Cobb-Douglas + Fréchet utility | `aup.UtilityEngine(mode="softmax")` |
| V2 | Baseline-ABM argmax | ABM, argmax, Fréchet idiosyncratic shocks | `aup.UtilityEngine(mode="argmax", noise="frechet")` |
| V3 | Normal-ABM argmax | ABM, argmax, Gaussian shocks | `aup.UtilityEngine(mode="argmax", noise="normal")` |
| V4-B | Hybrid-ABM | LLM elicits per-agent (β, κ); closed-form choice | `aup.HybridDecisionEngine(elicitor=...)` |
| V5.4 | LLM-ABM (paper headline) | Full LLM as decision maker; score-all-96 | `aup.LLMDecisionEngine(response_format="score_all", rebalance_instruction=True, stage2_top_k_residences=10)` |

## Conceptual layers

```{mermaid}
flowchart LR
    subgraph V1V2V3 ["UtilityEngine"]
        v1[V1 softmax]
        v2[V2 argmax + Fréchet]
        v3[V3 argmax + Normal]
    end
    subgraph V4B ["HybridDecisionEngine"]
        v4b[V4-B Hybrid]
    end
    subgraph V54 ["LLMDecisionEngine"]
        v50[V5.0 top-5]
        v54[V5.4 score-all-96]
    end
```

Each box is a single API class. Variants within a class differ only in
constructor kwargs.

## When to use which

```{list-table}
:header-rows: 1

* - Use case
  - Recommended
  - Why
* - Quick policy comparison, no LLM access
  - V1 (Baseline-softmax)
  - Deterministic, fastest, captures structural mechanisms
* - Stochastic ABM with realistic shocks
  - V2 / V3
  - Captures Fréchet (V2) or Gaussian (V3) idiosyncratic noise
* - LLM brings contextual reasoning, but funct. form is fine
  - V4-B (Hybrid-ABM)
  - LLM elicits parameters once per cluster; cheap, interpretable
* - Research question: "what if agents reason like an LLM?"
  - V5.4 (LLM-ABM)
  - Full LLM-as-decision-maker; captures cultural/identity/agglomeration
    mechanisms structural models forbid
* - Reproduce paper's main results
  - V1, V2, V3, V4-B, V5.4 all needed
  - Cross-version comparison is the paper's central artifact
```

## V5.4 — paper headline

The score-all-96 + rebalance + stage-2 cap configuration:

```python
engine = aup.LLMDecisionEngine(
    params=scenario.ahlfeldt_params,
    llm_client=aup.llm.CodexCliClient(),
    response_format="score_all",       # ask LLM to score every zone
    rebalance_instruction=True,        # add affordability ≥ amenity instruction
    stage2_top_k_residences=10,        # cap stage-2 fan-out
    cluster_k=50,
    num_agents=1_000_000,
    seed=42,
)
```

See {doc}`/tutorials/03_full_llm_v5` for the full walkthrough including
why each kwarg matters.

## Custom variants

```python
class MyEngine(aup.DecisionEngine):
    def decide_batch(self, agents, environment, zone_options, prices):
        # ... your logic ...
        return [LocationChoice(...) for agent in agents]

# Use it like any built-in:
sim = aup.SimulationEngine(scenario, agent_config, engine=MyEngine())
```

See {doc}`/tutorials/02_custom_decision_engine` for the full pattern.

## See also

- {doc}`/api/index` — full API reference
- {doc}`llm_integration` — LLM provider configuration
- {doc}`/tutorials/03_full_llm_v5` — V5.0/V5.4 deep dive
