# Berlin V1-V5 reproducibility

End-to-end reproduction guidance for the paper's 5 model variants on the
synthetic 96-zone Berlin scenario.

## Reproducibility tiers

```{list-table}
:header-rows: 1

* - Tier
  - What
  - Wall-clock
  - Prerequisites
* - **Tier 1**
  - `pip install agent-urban-planning` + `import agent_urban_planning`
  - <30 s
  - Python 3.9+
* - **Tier 2**
  - `python examples/01_quickstart.py` (full V1 pipeline walkthrough)
  - <1 min
  - Tier 1 + bundled scenario YAML (in git, not PyPI)
* - **Tier 3a**
  - V1 baseline + shock
  - ~3 hr
  - Tier 2 + bundled Berlin data (in git, not PyPI)
* - **Tier 3b**
  - V2 / V3 baseline + shock
  - ~3 hr each
  - Tier 3a
* - **Tier 3c**
  - V4 baseline + shock
  - ~3 hr
  - Tier 3a + LLM credits (~$5)
* - **Tier 4 (cache replay)**
  - V5 baseline + shock from bundled cache
  - ~5 min
  - Tier 3a + bundled `data/berlin/llm_cache_v5/`
* - **Tier 4 (live)**
  - V5 baseline + shock with live LLM calls
  - ~10 hr
  - Tier 3a + LLM credits ($30-50)
```

## Step-by-step

### Tier 1: Install
```bash
pip install agent-urban-planning
python -c "import agent_urban_planning as aup; print(aup.__version__)"
# → 0.1.0
```

### Tier 2: Pipeline walkthrough (needs bundled scenario YAML)
```bash
python examples/01_quickstart.py
```
Walks through Environment → Agents → Decision → Market → Equilibrium
for V1 (closed-form softmax). Verifies the install + pipeline wiring
end-to-end.

### Tier 3+ requires git clone

The bundled Berlin Ortsteile NPZ files are git-only (excluded from PyPI
sdist). Tier 3 and Tier 4 require:

```bash
git clone https://anonymous.4open.science/r/agent-urban-planning-4B4D.git
cd agent-urban-planning
pip install -e ".[llm,plot,berlin]"
```

### Tier 3a: V1 (no LLM)
```bash
python examples/02_berlin_replication/run_v1_softmax.py
```
Outputs:
- `output/berlin_v1_softmax/per_zone.csv`
- `output/berlin_v1_shock_east_west/per_zone.csv`

Numerical match to dev repo's V1: within 1e-3 numerical tolerance (V1 is
deterministic at seed 42).

### Tier 3b: V2, V3 (no LLM)
```bash
python examples/02_berlin_replication/run_v2_argmax_frechet.py
python examples/02_berlin_replication/run_v3_argmax_normal.py
```
Each ~3 hr. V2 and V3 are stochastic but seeded — deterministic at seed 42.

### Tier 3c: V4 (LLM elicitation)
```bash
python examples/02_berlin_replication/run_v4_hybrid.py --llm-provider codex-cli
```
Requires `codex` CLI authenticated. Cost: ~$5 in API credits.

### Tier 4 (cache replay): V5 without LLM credits
```bash
python examples/02_berlin_replication/run_v5_score_all.py --no-llm
```
Replays bundled cache at `data/berlin/llm_cache_v5/`. ~5 min wall-clock.

### Tier 4 (live): V5 with live LLM
```bash
python examples/02_berlin_replication/run_v5_score_all.py --llm-provider codex-cli
```
Cost: $30-50. Wall-clock: ~10 hr. Reproduces baseline + shock from scratch.

## After all variants complete

Each `output/{variant}/per_zone.csv` and
`output/{variant}_shock_east_west/per_zone.csv` is a 96-row CSV with
columns `zone_id, Q_sim, HR_sim, HM_sim, wage_sim, Q_obs, HR_obs,
HM_obs, wage_obs`.

Two aggregation scripts under `examples/02_berlin_replication/`
consume the per-variant CSVs and produce the paper's headline
artefacts:

```bash
# Cross-variant moments (paper Tables 2 and 5)
python examples/02_berlin_replication/build_moments_table.py
# -> output/comparison/comparison_moments_abridged.csv  (Table 2)
# -> output/comparison/comparison_moments_full.csv      (Table 5)
# -> output/comparison/comparison_moments.md            (GFM table)

# Per-zone choropleths (paper Figures 3 and 4)
python examples/02_berlin_replication/plot_dlogQ_dlogw.py
# -> output/comparison/figure_dlogQ.png                 (Figure 3)
# -> output/comparison/figure_dlogw.png                 (Figure 4)
```

`plot_dlogQ_dlogw.py` requires `geopandas` and `matplotlib`; install
with `pip install -e ".[plot]"` if not already.

For custom analyses, the per-zone CSVs are usable directly with
`pandas`:

```python
import pandas as pd
v1 = pd.read_csv("output/berlin_v1_softmax/per_zone.csv")
v1_shock = pd.read_csv("output/berlin_v1_softmax_shock_east_west/per_zone.csv")
dlog_Q = (v1_shock.Q_sim / v1.Q_sim).pipe(lambda s: s.apply("log"))
print(dlog_Q.describe())
```

Pre-rendered versions of the paper's headline figures and tables are
also bundled at `figures/comparison_moments.csv`, `figures/berlin_dlogQ.png`,
and `figures/berlin_dlogW.png` for offline reference without rerunning
the pipeline.

## Numerical reproducibility expectations

| Variant | Seed-determinism | Tolerance vs dev repo |
|---|---|---|
| V1 (Baseline-softmax) | Fully deterministic | exact |
| V2 (Baseline-ABM argmax) | Seeded stochastic | <1e-3 numerical |
| V3 (Normal-ABM argmax) | Seeded stochastic | <1e-3 numerical |
| V4 (Hybrid-ABM) | Seeded + LLM cache | <1e-2 (LLM elicitation noise) |
| V5 (LLM-ABM, cache replay) | Cache-deterministic | exact |
| V5 (LLM-ABM, live) | Provider + temperature dependent | qualitative match only |

## Troubleshooting

### "Bundled Berlin data missing"
You ran `pip install` instead of `git clone`. Re-clone the repo.

### LLM provider not configured
```bash
# Verify codex-cli auth
codex login

# Or use Anthropic SDK
export ANTHROPIC_API_KEY=...
python ... --llm-provider anthropic
```

### Numerical divergence
- Verify seed=42 (the paper default).
- For V5: use the bundled cache (`--no-llm` mode) or the same provider + temperature as the paper (codex-cli, temperature=0).
- Live LLM runs at different seeds will not be bit-identical to the paper.

## See also

- {doc}`shock_analysis` — methodology for the East-West Express τ shock
- {doc}`/tutorials/04_berlin_replication` — task-oriented walkthrough
