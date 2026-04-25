# Quickstart

Goal: get from `pip install` to a running 5-variant smoke test in <30 seconds.

## Install

```bash
pip install agent-urban-planning
```

For the LLM-using variants (V4-B, V5.4), also install the LLM extras:

```bash
pip install "agent-urban-planning[llm]"
```

## Smoke test the 5 paper variants

The {file}`examples/01_quickstart_two_zone.py` script instantiates all 5
paper variants (V1, V2, V3, V5.0, V5.4) using stub LLM clients. Total
runtime: <1 second.

```bash
python examples/01_quickstart_two_zone.py
```

Expected output:

```
agent-urban-planning version: 0.1.0

  V1   Baseline-softmax    : UtilityEngine(mode='softmax', noise='frechet', _impl=AhlfeldtUtilityEngine)
  V2   Baseline-ABM argmax : UtilityEngine(mode='argmax', noise='frechet', _impl=AhlfeldtABMEngine)
  V3   Normal-ABM argmax   : UtilityEngine(mode='argmax', noise='normal', _impl=AhlfeldtABMEngine)
  V5.0 LLM top-5           : LLMDecisionEngine(response_format='top5', rebalance_instruction=False, _impl=AhlfeldtHierarchicalLLMEngine)
  V5.4 LLM score-all-96    : LLMDecisionEngine(response_format='score_all', rebalance_instruction=True, _impl=AhlfeldtHierarchicalLLMEngine)

All 5 paper variants instantiated successfully.
```

If you see this output, your install is healthy.

## What just happened

The script demonstrates the `agent_urban_planning` library's three
first-class API classes:

```python
import agent_urban_planning as aup

# V1 — Baseline-softmax
v1 = aup.UtilityEngine(params, mode="softmax")

# V2 — Baseline-ABM argmax (Fréchet shocks)
v2 = aup.UtilityEngine(params, mode="argmax", noise="frechet")

# V3 — Normal-ABM argmax (Gaussian shocks)
v3 = aup.UtilityEngine(params, mode="argmax", noise="normal")

# V5.4 — LLM-ABM (paper headline)
v5_4 = aup.LLMDecisionEngine(
    params, llm_client=...,
    response_format="score_all",
    rebalance_instruction=True,
    stage2_top_k_residences=10,
)
```

Configure variants via constructor kwargs; no subclassing required.

## Next steps

- {doc}`02_custom_decision_engine` — write your own decision engine.
- {doc}`03_full_llm_v5` — deep-dive on the V5.4 LLM-ABM pattern.
- {doc}`04_berlin_replication` — reproduce the paper's V1-V5.4 results.
