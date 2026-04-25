# Full LLM hierarchical engine (V5)

The {class}`agent_urban_planning.LLMDecisionEngine` is the library's
**headline contribution**: a full LLM-as-decision-maker hierarchical
engine that queries an LLM per agent cluster per market iteration to
make discrete location decisions directly. The paper's V5 (score-all-96
+ rebalance + stage-2 cap) is reproducible via constructor kwargs.

## The hierarchical pattern

```
┌──────────────────────────────────────────────────────────────────┐
│  Per agent cluster c, per market iteration t:                    │
│                                                                  │
│  ┌────────────────────────┐                                      │
│  │  Stage 1 — residence    │                                     │
│  │  Prompt: persona +      │   ── one LLM call per cluster       │
│  │    96 zones with         │                                    │
│  │    Q, w, B, A            │                                    │
│  │  → {"scores": [...]}    │                                     │
│  └─────────────┬──────────┘                                      │
│                │                                                 │
│                ▼  pick top-K residences (typically K=10)         │
│                                                                  │
│  ┌────────────────────────┐                                      │
│  │  Stage 2 — workplace    │   ── K LLM calls per cluster        │
│  │  Prompt: persona +      │      (one per residence picked)     │
│  │    96 zones with         │                                    │
│  │    w, commute_min(r→j)   │                                    │
│  │  → {"top_5": [...]}     │                                     │
│  └─────────────┬──────────┘                                      │
│                │                                                 │
│                ▼                                                 │
│   sample N agents per cluster from joint (residence, workplace) │
│   distribution → choices fed back into market clearing           │
└──────────────────────────────────────────────────────────────────┘
```

## V5 reproduction (score-all-96, paper headline)

```python
import agent_urban_planning as aup

engine = aup.LLMDecisionEngine(
    params=scenario.ahlfeldt_params,
    llm_client=aup.llm.CodexCliClient(),
    response_format="score_all",         # ask LLM to score every zone
    rebalance_instruction=True,          # affordability ≥ amenity prompt
    stage2_top_k_residences=10,          # cost cap on stage-2 fan-out
    cluster_k=50,
    num_agents=1_000_000,
    seed=42,
)
```

The four kwargs that matter:

| kwarg | Paper value | What it does |
|---|---|---|
| `response_format` | `"score_all"` | Ask the LLM to assign a score to every zone (full demand-curve signal) |
| `rebalance_instruction` | `True` | Add "weight affordability ≥ amenity" sentence to the prompt |
| `stage2_top_k_residences` | `10` | Cap stage-2 fan-out to top-K residences per cluster |
| `cluster_k` | `50` | Number of agent clusters (one LLM call per cluster per iteration) |

## Why score-all-96?

A naïve top-5 ranking approach has a structural limitation: 91 of 96
zones are *invisible* to the market every iteration. Demand changes
from price shocks can only re-rank within the top-5; the other zones
contribute zero to the demand-curve signal at all. The market literally
can't price zones the LLM ignores.

The `score_all` format asks the LLM to score every zone, giving each a
non-trivial probability mass. This makes the demand curve fully
populated and the tâtonnement converges. See the paper's §5 (V5
ablation) for the empirical evidence.

## Why the stage-2 cap?

Score-all-96 produces 96 residences per cluster in stage-1. Without a
cap, stage-2 fan-out becomes 50 clusters × 96 residences = 4,800 LLM
calls per market iteration — infeasible in practice. The
`stage2_top_k_residences=10` kwarg keeps stage-2 fan-out at 50 × 10 =
500 calls per iter while preserving ~85-95% of the residence
probability mass per cluster.

## Custom prompts

For research extensions, override the prompt builder + response
validator:

```python
def my_stage1_prompt(persona, zones_info, *, prompt_version):
    return ("system message", f"persona: {persona}; rank zones...")

def my_validator(raw, allowed_zone_names, **kwargs):
    # Parse raw LLM output → list of (zone_name, score)
    return [...]

engine = aup.LLMDecisionEngine(
    params,
    llm_client=aup.llm.CodexCliClient(),
    prompt_builder=my_stage1_prompt,
    response_validator=my_validator,
)
```

## Next steps

- {doc}`04_berlin_replication` — reproduce the paper's V5 baseline + shock.
- {doc}`/api/index` — full API reference for `LLMDecisionEngine`.
