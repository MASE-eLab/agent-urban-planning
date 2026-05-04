# Berlin V1-V5 replication

End-to-end reproduction of the paper's 5 model variants on the synthetic
96-zone Berlin scenario, baseline + East-West Express shock, at seed 42.

## Prerequisites

```bash
curl -L -o aup.zip 'https://anonymous.4open.science/api/repo/agent-urban-planning-4B4D/zip'
unzip aup.zip -d agent-urban-planning-4B4D
cd agent-urban-planning-4B4D
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[llm,plot,berlin]"
```

The bundled Berlin Ortsteile NPZ files at `data/berlin/ortsteile/` are
required. They ship in the git repo (~15 MB; see `data/README.md` for
size + license attribution to Ahlfeldt et al. 2015) but NOT in the PyPI
sdist. **`pip install` alone is not enough — you need the ZIP download.**

## Reproducibility tiers

| Tier | Variant | Wall-clock | LLM credits? |
|---|---|---|---|
| Tier 3a | V1 (Baseline-softmax) | ~3 hr | No |
| Tier 3b | V2 (Baseline-ABM argmax) | ~3 hr | No |
| Tier 3c | V3 (Normal-ABM argmax) | ~3 hr | No |
| Tier 3d | V4 (Hybrid-ABM, cache replay) | ~3 hr | No (bundled cache) |
| Tier 3d | V4 (Hybrid-ABM, live LLM) | ~3 hr | Yes (~$5) |
| Tier 4 | V5 (LLM-ABM, paper headline) | ~10 hr live / ~5 min cache-replay | Yes (~$30-50 live) / No (cache replay) |

A `--no-llm` mode on `run_v5_score_all.py` replays the bundled cached
LLM responses at `data/berlin/llm_cache_v5/` so Tier 4 can be reproduced
without any LLM credits.

## Run V1 (no LLM, simplest case)

```bash
python examples/02_berlin_replication/run_v1_softmax.py
```

Produces:
- `output/berlin_v1_softmax/per_zone.csv` (baseline)
- `output/berlin_v1_softmax_shock_east_west/per_zone.csv` (post-shock)

## Run V5 (paper headline, cache-replay mode)

```bash
python examples/02_berlin_replication/run_v5_score_all.py --no-llm
```

Replays cached LLM responses; ~5 min wall-clock. Produces:
- `output/berlin_v5_score_all/per_zone.csv`

## Run V5 (live LLM mode)

```bash
python examples/02_berlin_replication/run_v5_score_all.py --llm-provider codex-cli
```

Live mode requires the `codex` CLI to be authenticated (`codex login`).
Estimated cost: $30-50 in API credits. Wall-clock: ~10 hr.

## Compare + plot

Each variant's `output/{variant}/per_zone.csv` and
`output/{variant}_shock_east_west/per_zone.csv` contain the per-zone
sim/obs values for cross-variant analysis. After all five variants
have run, two aggregation scripts produce the paper's headline
artefacts directly from the per-variant CSVs:

```bash
# Cross-variant moments (paper Tables 2 and 5)
python examples/02_berlin_replication/build_moments_table.py

# Per-zone Δlog Q and Δlog w choropleths (paper Figures 3 and 4)
python examples/02_berlin_replication/plot_dlogQ_dlogw.py
```

Outputs land in `output/comparison/` as `comparison_moments_*.csv`,
`comparison_moments.md`, `figure_dlogQ.png`, and `figure_dlogw.png`.
`plot_dlogQ_dlogw.py` requires the `[plot]` install extra (geopandas
+ matplotlib).

Pre-rendered versions of the paper's headline figures and tables are
also bundled at `figures/comparison_moments.csv`, `figures/berlin_dlogQ.png`,
and `figures/berlin_dlogW.png` for offline reference. Custom analyses
beyond the bundled scripts can be built directly off the
`per_zone.csv` outputs using pandas/matplotlib.

## Numerical reproducibility

V1, V2, V3 are deterministic (closed-form or seeded-stochastic). V4
and V5 cache-replay paths (`--no-llm`) are bit-identical when run
against the bundled caches at `data/berlin/llm_cache_v4/` and
`data/berlin/llm_cache_v5/` respectively. Live LLM runs are
provider-dependent and additionally need:

1. Same LLM provider (`codex-cli` recommended)
2. Same seed (42)

Cross-variant numerical equivalence to the dev repo's outputs is
documented to within 1e-3 tolerance in the
{doc}`/reproducibility/berlin_v1_v5` page.

## Troubleshooting

**"Bundled Berlin data missing" error**: You ran `pip install` instead
of the ZIP download. The data files only ship inside the
anonymous.4open.science archive, not in the PyPI sdist. Re-fetch
the ZIP from anonymous.4open.science.

**LLM provider not configured**: For V4 and V5 live runs, verify
`codex --version` works (or your chosen provider's auth). Use `--no-llm`
on V5 to skip live LLM calls.

**Numerical divergence from paper**: Verify your seed is 42 (the paper
default) and you're using the bundled LLM cache for V5. Live LLM
runs at different seeds will not be bit-identical to the paper.

## Next steps

- {doc}`/api/index` — full API reference.
- {doc}`/reproducibility/berlin_v1_v5` — reproducibility tier definitions.
