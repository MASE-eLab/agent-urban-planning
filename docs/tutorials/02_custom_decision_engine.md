# Custom decision engine

Write your own decision engine by subclassing
{class}`agent_urban_planning.DecisionEngine`. Useful for research extensions
that don't fit the V1-V5.4 templates.

## The protocol

A decision engine implements a single method:

```python
def decide_batch(
    self,
    agents: list[Agent],
    environment: Environment,
    zone_options: list[str],
    prices: dict[str, float],
) -> list[LocationChoice]:
    ...
```

Given a batch of agents and current market state (zones, prices), return
a `LocationChoice` for each agent.

## Minimal example

```python
import agent_urban_planning as aup
from agent_urban_planning import DecisionEngine, LocationChoice


class RandomEngine(DecisionEngine):
    """Picks a random zone for each agent. Useful as a control baseline."""

    def __init__(self, seed=None):
        import numpy as np
        self.rng = np.random.default_rng(seed)

    def decide_batch(self, agents, environment, zone_options, prices):
        choices = []
        for agent in agents:
            zone = self.rng.choice(zone_options)
            choices.append(LocationChoice(
                residence=zone,
                workplace=zone,  # toy: same zone for both
                utility=0.0,
                zone_utilities={},
            ))
        return choices


# Use it just like any built-in engine
engine = RandomEngine(seed=42)
sim = aup.SimulationEngine(scenario=sc, agent_config=ag, engine=engine)
results = sim.run()
```

## Tips

- For LLM-based variants, prefer extending `LLMDecisionEngine` (or its
  internal helpers) over subclassing `DecisionEngine` directly — the
  built-in async batching and caching layer saves significant work.
- For closed-form variants, look at the source of
  {class}`agent_urban_planning.UtilityEngine` for reference patterns
  (Cobb-Douglas + Fréchet utility, softmax vs argmax dispatch).
- Custom engines can be passed to {class}`SimulationEngine` directly;
  no registration needed.

## Next steps

- {doc}`03_full_llm_v5` — the V5.4 LLM-ABM pattern in depth.
- {doc}`04_berlin_replication` — reproduce the paper's results.
