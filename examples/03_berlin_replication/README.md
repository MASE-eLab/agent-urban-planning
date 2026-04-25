# Berlin V1-V5.4 replication

Reproduces the paper's 5 model variants (V1, V2, V3, V4-B, V5.4) on the
synthetic 96-zone Berlin scenario, baseline + East-West Express shock.

## Prerequisites

```bash
pip install -e ".[llm,plot,berlin]"
```

Plus (for V5.4 LLM-ABM only): an LLM provider available locally:
- `codex-cli` — OAuth-managed `codex` CLI (preferred for V5.4 reproduction)
- `claude-code` — `claude` CLI
- or set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` for direct SDK access

The bundled Berlin Ortsteile NPZ files at `data/berlin/ortsteile/` are
required for all V1-V5.4 runs. They ship in the git repo (see
`data/README.md` for size + license attribution) but NOT in the PyPI sdist.
Reproducing these results requires `git clone` rather than `pip install`.

## Reproducibility tiers

| Tier | Variant | Wall-clock | LLM credits? |
|---|---|---|---|
| Tier 3a | V1 (Baseline-softmax) | ~3 hr | No |
| Tier 3b | V2 (Baseline-ABM argmax) | ~3 hr | No |
| Tier 3c | V3 (Normal-ABM argmax) | ~3 hr | No |
| Tier 3d | V4-B (Hybrid-ABM) | ~3 hr | Yes (modest, ~$5) |
| Tier 4 | V5.4 (LLM-ABM, paper headline) | ~10 hr | Yes (significant, ~$30-50) |

A `--no-llm` mode on `run_v5_4_score_all.py` replays the bundled cached
LLM responses at `data/berlin/llm_cache_v5_4/` so Tier 4 can be reproduced
without any LLM credits — just bundled data.

## Scripts

- `run_v1_softmax.py` — V1 baseline + East-West shock
- `run_v2_argmax_frechet.py` — V2 baseline + shock
- `run_v3_argmax_normal.py` — V3 baseline + shock
- `run_v4b_hybrid.py` — V4-B baseline + shock
- `run_v5_4_score_all.py` — V5.4 baseline + shock (paper headline)

All scripts share the same structure:
1. Load the Berlin scenario YAML (`data/berlin/scenarios/berlin_2006_ortsteile.yaml`).
2. Load agent config YAML (`data/berlin/agents/berlin_ortsteile_richer_10k.yaml`).
3. Instantiate `aup.UtilityEngine` / `aup.HybridDecisionEngine` /
   `aup.LLMDecisionEngine` with the variant's kwargs.
4. Run baseline market clearing.
5. Apply East-West Express τ shock.
6. Run shock market clearing (warm-started from baseline).
7. Save per_zone.csv outputs to `output/{variant}/`.

After all variants complete, run the comparison + plotting:

```bash
python examples/03_berlin_replication/compare_and_plot.py
```

This produces:
- `output/comparison/comparison_moments.csv` — cross-version moments
- `output/comparison/berlin_q_observed_and_log_change.png` — choropleth
- `output/comparison/berlin_w_observed_and_log_change.png` — choropleth
- `output/comparison/llm_abm_satisfaction.csv` — V5.4 self-rating sidebar

## Notes on numerical reproducibility

These scripts are seeded at 42 by default. Numerical reproducibility is
deterministic for V1, V2, V3 (no LLM noise). V4-B and V5.4 reproducibility
relies on:
- Same LLM provider (`codex-cli` recommended)
- Same prompt cache (bundled at `data/berlin/llm_cache_v5_4/`)
- Same seed

Cross-version equivalence with the dev repo's outputs is documented to
within 1e-3 numerical tolerance per the spec
(`openspec/changes/extract-library-agent-urban-planning/specs/library-examples-and-reproducibility/spec.md`).
