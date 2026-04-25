# Berlin V1-V5 replication

Reproduces the paper's 5 model variants (V1, V2, V3, V4, V5) on the
synthetic 96-zone Berlin scenario, baseline + East-West Express shock.

## Prerequisites

```bash
pip install -e ".[llm,plot,berlin]"
```

Plus (for V5 LLM-ABM only): an LLM provider available locally:
- `codex-cli` — OAuth-managed `codex` CLI (preferred for V5 reproduction)
- `claude-code` — `claude` CLI
- or set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` for direct SDK access

The bundled Berlin Ortsteile NPZ files at `data/berlin/ortsteile/` are
required for all V1-V5 runs. They ship in the git repo (see
`data/README.md` for size + license attribution) but NOT in the PyPI sdist.
Reproducing these results requires `git clone` rather than `pip install`.

## Reproducibility tiers

| Tier | Variant | Wall-clock | LLM credits? |
|---|---|---|---|
| Tier 3a | V1 (Baseline-softmax) | ~3 hr | No |
| Tier 3b | V2 (Baseline-ABM argmax) | ~3 hr | No |
| Tier 3c | V3 (Normal-ABM argmax) | ~3 hr | No |
| Tier 3d | V4 (Hybrid-ABM) | ~3 hr | Yes (modest, ~$5) |
| Tier 4 | V5 (LLM-ABM, paper headline) | ~10 hr live / ~5 min cache-replay | Yes / No |

A `--no-llm` mode on `run_v5_score_all.py` replays the bundled cached
LLM responses at `data/berlin/llm_cache_v5/` (hosted as a GitHub
release asset; see `data/README.md`) so Tier 4 can be reproduced
without any LLM credits — just bundled data.

## Scripts

- `run_v1_softmax.py` — V1 baseline + East-West shock
- `run_v2_argmax_frechet.py` — V2 baseline + shock
- `run_v3_argmax_normal.py` — V3 baseline + shock
- `run_v4_hybrid.py` — V4 baseline + shock (LLM elicits per-type β/κ)
- `run_v5_score_all.py` — V5 baseline + shock (paper headline)

All scripts share the same structure (defined in `_common.py`):
1. Load the Berlin scenario YAML
   (`data/berlin/scenarios/berlin_2006_ortsteile.yaml`).
2. Load agent config YAML
   (`data/berlin/agents/berlin_ortsteile_richer_10k.yaml`).
3. Construct an engine factory — `aup.UtilityEngine` /
   `aup.HybridDecisionEngine` / `aup.LLMDecisionEngine` with the
   variant's kwargs.
4. Run baseline market clearing → `output/{variant}/per_zone.csv`.
5. Apply East-West Express τ shock to the travel-time matrix.
6. Re-run the market warm-started from baseline →
   `output/{variant}_shock_east_west/per_zone.csv`.

Each script accepts `--seed`, `--iters`, and (where applicable)
`--num-agents`, `--batch-size`, `--llm-provider`.

## Smoke testing

To verify the pipeline runs end-to-end at low cost, drop the agent
count and iteration cap:

```bash
python examples/03_berlin_replication/run_v1_softmax.py --iters 3
python examples/03_berlin_replication/run_v2_argmax_frechet.py \
    --iters 3 --num-agents 10000 --batch-size 5000
python examples/03_berlin_replication/run_v3_argmax_normal.py \
    --iters 3 --num-agents 10000 --batch-size 5000
python examples/03_berlin_replication/run_v4_hybrid.py \
    --iters 3 --llm-provider stub-uniform
python examples/03_berlin_replication/run_v5_score_all.py \
    --iters 2 --num-agents 1000 --batch-size 500 --cluster-k 5 \
    --llm-provider stub-score-all
```

Each smoke test takes < 1 minute and writes a 96-row `per_zone.csv`.

## Notes on numerical reproducibility

These scripts are seeded at 42 by default. Numerical reproducibility is
deterministic for V1, V2, V3 (no LLM noise). V4 and V5 reproducibility
relies on:
- Same LLM provider (`codex-cli` recommended)
- Same prompt cache (bundled at `data/berlin/llm_cache_v5/`)
- Same seed

Cross-version equivalence with the dev repo's outputs is documented to
within 1e-3 numerical tolerance per the spec
(`openspec/changes/extract-library-agent-urban-planning/specs/library-examples-and-reproducibility/spec.md`).
