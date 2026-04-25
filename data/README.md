# Bundled data

Data files in this directory ship with the **git repository** but are
**excluded from the PyPI source distribution** (via `MANIFEST.in`). This keeps
the PyPI install lightweight while still letting `git clone` users reproduce
all paper results out of the box.

`pip install agent-urban-planning` alone will NOT include these files —
you need a `git clone` for paper reproducibility.

## What's bundled (in this repo)

| Path | Purpose | Size | Bundled in git? | In PyPI sdist? |
|---|---|---|---|---|
| `singapore/` | Singapore real-data tiny scenario YAMLs | < 1 MB | yes | yes |
| `berlin/ortsteile/` | 96-zone Ortsteile NPZ files (paper main resolution) | 328 KB | yes | no |
| `berlin/bezirke/` | 12-Bezirk NPZ files (smoke-test resolution) | 20 KB | yes | no |
| `berlin/blocks/fund_2006.npz` | Block-level fundamentals for L-override | 840 KB | yes | no |
| `berlin/ortsteile_joint/` | Joint residence-workplace 2011 micro-data | 8.9 MB | yes | no |
| `berlin/crosswalks/` | Block ↔ Bezirk/Ortsteile CSVs + aggregation rules | 1.4 MB | yes | yes |
| `berlin/params/` | Calibration parameters (α, β, ε etc.) | < 4 KB | yes | yes |
| `berlin/public_demographics/` | Public demographic distributions | 80 KB | yes | yes |
| `berlin/scenarios/` | Berlin scenario YAML (96-zone Ortsteile, 2006) | < 4 KB | yes | yes |
| `berlin/agents/` | Berlin agent demographics (10k richer profiles) | 4.1 MB | yes | yes |
| `berlin/shocks/` | East-West Express τ shock spec | < 1 KB | yes | yes |
| `shapefiles/` | Berlin block + green + water shapefiles (5 files × 3 layers) | 10 MB | yes | no |

Total bundled: ~21 MB.

## What's NOT bundled — and how to regenerate

### Block-level NPZ files (`berlin/blocks/`)

Block-level travel-time matrices are **2 GB+** (e.g., `tt_2006_public.npz`
is 700 MB) — too large for git. Skip if you're working at the
Ortsteile resolution (paper default); regenerate from the original
Ahlfeldt et al. 2015 replication package via the dev-repo importer:

```bash
# In the dev repo (multi_agent_simulator)
python scripts/import_berlin_memory_pack.py
```

This reads the upstream `.h5` / `.mat` files and writes block-level
NPZs to `data/berlin/blocks/`.

### V5 LLM cache (`berlin/llm_cache_v5/`)

The bundled cache from the paper's V5 score-all-96 baseline + shock
runs is **~320 MB** — too large for git. Three options:

1. **Live LLM run**: configure a provider (`codex-cli` recommended)
   and rerun:

   ```bash
   python examples/03_berlin_replication/run_v5_score_all.py \
       --llm-provider codex-cli
   ```

   Wall-clock ~10 hours, requires LLM credits (~$30-50 for codex-cli).
   The first run populates `data/berlin/llm_cache_v5/`; subsequent
   runs at the same seed reuse the cache and finish in minutes.

2. **External release artifact (TBD)**: we plan to attach the cache
   bundle as a GitHub release asset for v0.1.0; check the
   [Releases page](https://github.com/MASE-eLab/agent-urban-planning/releases).
   Once available, download to `data/berlin/llm_cache_v5/`:

   ```bash
   mkdir -p data/berlin/llm_cache_v5
   curl -L https://github.com/MASE-eLab/agent-urban-planning/releases/download/v0.1.0/llm_cache_v5.tar.gz \
     | tar -xz -C data/berlin/llm_cache_v5/
   python examples/03_berlin_replication/run_v5_score_all.py --no-llm
   ```

3. **Smoke test only**: run with the bundled stub LLM client
   (returns uniform-score responses; not paper-faithful but exercises
   the full pipeline):

   ```bash
   python examples/03_berlin_replication/run_v5_score_all.py \
       --llm-provider stub-score-all --num-agents 1000 --iters 3
   ```

## Sources + attribution

### Berlin Ortsteile / Bezirke / blocks NPZ files

Derived from the Ahlfeldt et al. (2015) "Economics of Density" replication
package via this repo's importer (`scripts/import_berlin_memory_pack.py`
in the dev repo). The NPZ files are aggregated 96- and 12-zone summaries
computed from the original block-level data; no raw third-party data
is redistributed beyond what the upstream package permits.

Citation:

> Ahlfeldt, G. M., Redding, S. J., Sturm, D. M., Wolf, N. (2015). The
> economics of density: Evidence from the Berlin Wall. *Econometrica*,
> 83(6), 2127-2189. [DOI:10.3982/ECTA10876](https://doi.org/10.3982/ECTA10876)

The Econometric Society publishes a
[supplemental materials page](https://www.econometricsociety.org/publications/econometrica/2015/11/01/economics-density-evidence-berlin-wall)
with the original replication package (MATLAB code + block-level data).

### Berlin shapefiles (`shapefiles/`)

`Berlin4matlab.{shp,shx,dbf,prj,cpg}` (12,309 block polygons),
`BerlinGreen.*`, `BerlinWater.*` are sourced from the Ahlfeldt et al.
(2015) public replication package. They are redistributed here under
the original package's terms (academic / research use). If you use
these files in derivative work, please cite the Ahlfeldt et al. (2015)
paper above.

### Berlin Senate administrative shapefiles (used to build crosswalks)

The block-to-Bezirk and block-to-Ortsteil crosswalks are built from:

- **Bezirksgrenzen** (12 boroughs):
  <https://daten.berlin.de/datensaetze/geometrien-der-bezirke>
- **Ortsteile** (97 localities):
  <https://daten.berlin.de/datensaetze/geometrien-der-ortsteile>

## Reproducibility

For full reproducibility instructions, see
[`docs/reproducibility/berlin_v1_v5.md`](../docs/reproducibility/berlin_v1_v5.md)
and the example scripts at `examples/03_berlin_replication/`.
