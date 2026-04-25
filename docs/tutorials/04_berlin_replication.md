# Berlin V1-V5.4 replication

End-to-end reproduction of the paper's 5 model variants on the synthetic
96-zone Berlin scenario, baseline + East-West Express shock, at seed 42.

## Prerequisites

```bash
git clone https://github.com/MASE-eLab/agent-urban-planning.git
cd agent-urban-planning
pip install -e ".[llm,plot,berlin]"
```

The bundled Berlin Ortsteile NPZ files at `data/berlin/ortsteile/` are
required. They ship in the git repo (~15 MB; see `data/README.md` for
size + license attribution to Ahlfeldt et al. 2015) but NOT in the PyPI
sdist. **`pip install` alone is not enough — you need the git clone.**

## Reproducibility tiers

| Tier | Variant | Wall-clock | LLM credits? |
|---|---|---|---|
| Tier 3a | V1 (Baseline-softmax) | ~3 hr | No |
| Tier 3b | V2 (Baseline-ABM argmax) | ~3 hr | No |
| Tier 3c | V3 (Normal-ABM argmax) | ~3 hr | No |
| Tier 3d | V4-B (Hybrid-ABM) | ~3 hr | Yes (~$5) |
| Tier 4 | V5.4 (LLM-ABM, paper headline) | ~10 hr live / ~5 min cache-replay | Yes (~$30-50 live) / No (cache replay) |

A `--no-llm` mode on `run_v5_4_score_all.py` replays the bundled cached
LLM responses at `data/berlin/llm_cache_v5_4/` so Tier 4 can be reproduced
without any LLM credits.

## Run V1 (no LLM, simplest case)

```bash
python examples/03_berlin_replication/run_v1_softmax.py
```

Produces:
- `output/berlin_v1_softmax/per_zone.csv` (baseline)
- `output/berlin_v1_shock_east_west/per_zone.csv` (post-shock)

## Run V5.4 (paper headline, cache-replay mode)

```bash
python examples/03_berlin_replication/run_v5_4_score_all.py --no-llm
```

Replays cached LLM responses; ~5 min wall-clock. Produces:
- `output/berlin_v5_4_score_all/per_zone.csv`

## Run V5.4 (live LLM mode)

```bash
python examples/03_berlin_replication/run_v5_4_score_all.py --llm-provider codex-cli
```

Live mode requires the `codex` CLI to be authenticated (`codex login`).
Estimated cost: $30-50 in API credits. Wall-clock: ~10 hr.

## Compare + plot

After all 5 variants complete:

```bash
python examples/03_berlin_replication/compare_and_plot.py
```

Produces:
- `output/comparison/comparison_moments.csv` — cross-version moments table
- `output/comparison/berlin_q_observed_and_log_change.png` — choropleth
- `output/comparison/berlin_w_observed_and_log_change.png` — choropleth
- `output/comparison/llm_abm_satisfaction.csv` — V5.4 self-rating sidebar

## Numerical reproducibility

V1, V2, V3 are deterministic (closed-form or seeded-stochastic). V4-B
and V5.4 reproducibility depends on:

1. Same LLM provider (`codex-cli` recommended for V5.4)
2. Same prompt cache (bundled at `data/berlin/llm_cache_v5_4/`)
3. Same seed (42)

Cross-variant numerical equivalence to the dev repo's outputs is
documented to within 1e-3 tolerance in the
{doc}`/reproducibility/berlin_v1_v5_4` page.

## Troubleshooting

**"Bundled Berlin data missing" error**: You ran `pip install` instead
of `git clone`. The data files are git-only. Re-clone the repo or
download the data separately.

**LLM provider not configured**: For V4-B and V5.4 live runs, verify
`codex --version` works (or your chosen provider's auth). Use `--no-llm`
on V5.4 to skip live LLM calls.

**Numerical divergence from paper**: Verify your seed is 42 (the paper
default) and you're using the bundled LLM cache for V5.4. Live LLM
runs at different seeds will not be bit-identical to the paper.

## Next steps

- {doc}`/api/index` — full API reference.
- {doc}`/reproducibility/berlin_v1_v5_4` — reproducibility tier definitions.
