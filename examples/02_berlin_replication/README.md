# Berlin V1-V5 replication

Reproduces the paper's 5 model variants (V1, V2, V3, V4, V5) on the
96-zone Berlin Ortsteile scenario, baseline + East-West Express shock.

## What you'll get

Each `run_v*.py` script produces two `per_zone.csv` files (96 rows each):

- `output/{variant}/per_zone.csv` — baseline equilibrium
- `output/{variant}_shock_east_west/per_zone.csv` — shock equilibrium

Columns: `zone_id, Q_sim, HR_sim, HM_sim, wage_sim, Q_obs, HR_obs,
HM_obs, wage_obs` — simulated vs pack-observed housing prices,
residence count, workplace count, wages.

Plus `seed.json`, `diagnostics.json` (and `shock_config.json` for
shock runs) per directory for reproducibility metadata.

After all five variants finish, two further scripts aggregate the
per-zone CSVs into the artefacts reported in the paper:

| Script | Produces | Paper artefact |
|---|---|---|
| `build_moments_table.py` | `output/comparison/comparison_moments_abridged.csv` + `comparison_moments_full.csv` + `comparison_moments.md` | Tables 2 (abridged) and 5 (full) |
| `plot_dlogQ_dlogw.py` | `output/comparison/figure_dlogQ.png` and `figure_dlogw.png` | Figures 3 and 4 |

```bash
# After V1-V5 have all been run:
python examples/02_berlin_replication/build_moments_table.py
python examples/02_berlin_replication/plot_dlogQ_dlogw.py
```

`plot_dlogQ_dlogw.py` requires `geopandas` and `matplotlib`; install
with `pip install -e ".[plot]"` if not already.

## Prerequisites

```bash
git clone https://anonymous.4open.science/r/agent-urban-planning-4B4D.git
cd agent-urban-planning
pip install -e ".[llm,plot,berlin]"
```

The bundled Berlin Ortsteile NPZ files at `data/berlin/ortsteile/`
are required for **all** V1-V5 runs. They ship in the git repo
(~21 MB total — see `data/README.md` for size + license attribution
to Ahlfeldt et al. 2015) but **not** in the PyPI sdist. Reproducing
these results requires `git clone`, not `pip install`.

For V4 and V5 you also need an LLM provider:

| Provider | Auth | Notes |
|---|---|---|
| `codex-cli` | OAuth via `codex` CLI | Recommended for V5 reproduction (paper's provider) |
| `claude-code` | OAuth via `claude` CLI | Free on Claude Pro plans |
| `anthropic` | `ANTHROPIC_API_KEY` | Direct Anthropic SDK |
| `openai` | `OPENAI_API_KEY` | Direct OpenAI SDK |
| `zai-coding` | `ZAI_API_KEY` | Anthropic-compatible Z.ai proxy |

## Reproducibility tiers

| Tier | Variant | Wall-clock | LLM credits? |
|---|---|---|---|
| Tier 3a | V1 (Baseline-softmax) | ~3 hr | No |
| Tier 3b | V2 (Baseline-ABM argmax, Fréchet) | ~3 hr | No |
| Tier 3c | V3 (Normal-ABM argmax) | ~3 hr | No |
| Tier 3d | V4 (Hybrid-ABM, LLM elicitation) | ~3 hr | Yes (~$5 with codex-cli) |
| **Tier 4 V5 — cache replay** | V5 (LLM-ABM, paper headline) | **~5–10 min** | **No** (uses bundled cache) |
| Tier 4 V5 — from scratch | V5 (LLM-ABM, paper headline) | ~10 hr live | Yes (~$30–50 with codex-cli) |

## Real runs (paper config — full reproduction)

All scripts default to the paper config: **seed=42, iters=50** —
just run them. Don't pass `--num-agents` or `--iters` unless you
explicitly want a non-paper config (e.g., for smoke testing).

### V1 — Baseline-softmax (Tier 3a, ~3 hr, no credits)

```bash
python examples/02_berlin_replication/run_v1_softmax.py
```

Closed-form Cobb-Douglas + Fréchet softmax. Deterministic — repeat
runs are bit-identical.

### V2 — Baseline-ABM argmax with Fréchet shocks (Tier 3b, ~3 hr)

```bash
python examples/02_berlin_replication/run_v2_argmax_frechet.py
```

ABM with Fréchet idiosyncratic shocks. Seeded; same seed → same outputs.

### V3 — Normal-ABM argmax with Gaussian shocks (Tier 3c, ~3 hr)

```bash
python examples/02_berlin_replication/run_v3_argmax_normal.py
```

Same as V2 but with Gaussian shocks (thinner tails).

### V4 — Hybrid-ABM (Tier 3d, ~3 hr, ~$5 in LLM credits)

V4 uses an LLM to elicit per-agent preference weights `(β, κ)` once
per agent type, then runs the closed-form mixed logit. The
elicitation is light: ~50 LLM calls total per provider.

```bash
# Recommended provider (paper's provider, OAuth-managed):
python examples/02_berlin_replication/run_v4_hybrid.py --llm-provider codex-cli

# Other providers:
python examples/02_berlin_replication/run_v4_hybrid.py --llm-provider claude-code
python examples/02_berlin_replication/run_v4_hybrid.py --llm-provider anthropic
python examples/02_berlin_replication/run_v4_hybrid.py --llm-provider openai
```

Elicited preferences cache to
`.cache/llm_preferences_berlin_v4/` (per-agent-type JSON files keyed
by demographic signature). Same provider + seed = same elicited
preferences.

There is no separate "with cache" / "from scratch" distinction for V4
because the elicitation cache is small (~50 entries) and rebuilds in
seconds. The expensive part of V4 is the closed-form mixed-logit
clearing, which always runs from scratch.

### V5 — LLM-ABM, paper headline

V5 has two paths because the LLM-call volume is heavy
(~80,000 cached prompt/response pairs across baseline + shock).

#### Path A — Cache replay (recommended for reviewers, ~5–10 min, no credits)

Use the bundled LLM-response cache hosted as a GitHub release asset:

```bash
# 1. Download + extract the cache (one-time, 15 MB compressed → 320 MB raw)
curl -L -o llm_cache_v5.tar.gz \
  https://huggingface.co/datasets/aup-anon-2026/AUP-V5-LLM-cache-Berlin-Ortsteile/resolve/main/llm_cache_v5.tar.gz
tar -xzf llm_cache_v5.tar.gz -C data/berlin/

# 2. Run V5 with cache replay
python examples/02_berlin_replication/run_v5_score_all.py --no-llm
```

`--no-llm` mode runs at the full paper config (`cluster_k=50,
num_agents=1_000_000, iters=50, seed=42`) and replays cached LLM
responses for every prompt. The cache is split by phase
(`data/berlin/llm_cache_v5/baseline/` + `.../shock/`) for correctness;
both subdirs must exist after extraction.

If your params (e.g., `--cluster-k 5`, `--num-agents 1000`) don't
match the paper config, `--no-llm` will hard-fail on the first
cache miss with a clear error pointing here. Use Path B or smoke
testing for off-paper experiments.

#### Path B — From scratch with live LLM (~10 hr, ~$30–50 credits)

Reproduces V5 with no bundled cache. The first run populates
`data/berlin/llm_cache_v5/{baseline,shock}/`; subsequent runs at the
same seed reuse what's there.

```bash
# Recommended provider (paper's provider):
python examples/02_berlin_replication/run_v5_score_all.py --llm-provider codex-cli

# Alternatives:
python examples/02_berlin_replication/run_v5_score_all.py --llm-provider claude-code

# Concurrency tuning (default 15 for codex-cli; raise if you have headroom):
python examples/02_berlin_replication/run_v5_score_all.py \
    --llm-provider codex-cli --llm-concurrency 30
```

This path is what the paper authors ran to populate the cache that
Path A replays. Reproducing exactly requires the same provider +
model versions; live LLM runs at different providers will produce
qualitatively similar but not bit-identical numbers.

## Cross-variant comparison

After each variant has produced its `output/{variant}/per_zone.csv`
+ `output/{variant}_shock_east_west/per_zone.csv`, the per-zone
log-change moments are easy to recover with pandas:

```python
import numpy as np, pandas as pd
variants = ["berlin_v1_softmax", "berlin_v2_argmax_frechet",
            "berlin_v3_argmax_normal", "berlin_v4_hybrid",
            "berlin_v5_score_all"]
for v in variants:
    base = pd.read_csv(f"output/{v}/per_zone.csv")
    shock = pd.read_csv(f"output/{v}_shock_east_west/per_zone.csv")
    dlogQ = np.log(shock.Q_sim / base.Q_sim)
    print(f"{v}: μ={dlogQ.mean():+.4f} σ={dlogQ.std():.4f}")
```

The paper's headline moments + interpretation are in the main README
and at `figures/comparison_moments.csv`.

## Smoke testing (development / install verification)

To verify the pipeline runs end-to-end at low cost (each takes
< 1 minute), drop the agent count and iteration cap:

```bash
python examples/02_berlin_replication/run_v1_softmax.py --iters 3
python examples/02_berlin_replication/run_v2_argmax_frechet.py \
    --iters 3 --num-agents 10000 --batch-size 5000
python examples/02_berlin_replication/run_v3_argmax_normal.py \
    --iters 3 --num-agents 10000 --batch-size 5000
python examples/02_berlin_replication/run_v4_hybrid.py \
    --iters 3 --llm-provider stub-uniform
python examples/02_berlin_replication/run_v5_score_all.py \
    --iters 2 --num-agents 1000 --batch-size 500 --cluster-k 5 \
    --llm-provider stub-score-all
```

`stub-uniform` (V4) and `stub-score-all` (V5) are bundled stub LLM
clients that return fixed dummy responses — they exercise the full
plumbing without any network access or credits. Smoke-test outputs
are not paper-faithful; use the real-run instructions above for
reproducibility.

## Numerical reproducibility

| Variant | Determinism | Same seed = same outputs? |
|---|---|---|
| V1 | Closed-form, deterministic | exact bit-equality |
| V2/V3 | Seeded ABM | exact bit-equality |
| V4 | Seeded ABM + cached LLM elicitation | exact bit-equality given same provider |
| V5 (cache replay) | Cache-deterministic | exact bit-equality |
| V5 (live LLM) | Provider + temperature dependent | qualitative match only |

V5 live runs at temperature=0 with the same provider give numerically
similar (not bit-identical) outputs across runs because providers
periodically update model versions. Use Path A (cache replay) for
exact reproducibility.

## Pipeline structure

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

## See also

- `docs/reproducibility/berlin_v1_v5.md` — full reproducibility tier
  ladder (rendered at <https://anonymous.4open.science/r/agent-urban-planning-4B4D>)
- `docs/reproducibility/shock_analysis.md` — East-West Express
  shock methodology + Route-C min-of-paths formula
- `data/README.md` — bundled-data inventory + V5 cache release-asset
  download instructions
