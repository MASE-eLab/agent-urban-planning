# V5 LLM cache

This directory hosts the bundled prompt-response cache for the V5 (LLM-decide)
decision engine on the 96-zone Berlin Ortsteile benchmark instance. Reproducing
the V5 headline numbers in the paper without LLM credits (Tier 4) requires the
cache contents to be present here.

## What lives in git

- `croissant.json` — Croissant 1.0 JSON-LD metadata describing the cache
  (referenced from Appendix F of the paper).
- `README.md` — this file.

## What lives outside git

The cache itself is **not tracked in git** (320 MB across roughly 81,000 small
JSON files). It is distributed as a single compressed tarball
(`llm_cache_v5.tar.gz`, ~15 MB compressed) and must be downloaded and extracted
into this directory before running the V5 cache-replay reproduction.

## Extracting the cache

After cloning the repository:

```bash
# From the repository root:
curl -L -o llm_cache_v5.tar.gz https://huggingface.co/datasets/aup-anon-2026/AUP-V5-LLM-cache-Berlin-Ortsteile/resolve/main/llm_cache_v5.tar.gz
tar -xzf llm_cache_v5.tar.gz -C data/berlin/
rm llm_cache_v5.tar.gz
```

The extracted directory will contain two top-level subdirectories:

```
data/berlin/llm_cache_v5/
├── README.md            (this file)
├── croissant.json       (Croissant metadata)
├── baseline/            (~54,000 JSON files, ~214 MB)
└── shock/               (~27,000 JSON files, ~106 MB)
```

Each cache entry is a JSON file keyed by
`<persona-cluster-id>_<iteration>_<stage>.json` and holds the prompt sent to
the foundation model plus the parsed response (action assignment + 0–100
satisfaction score).

## Running V5 with the cache

```bash
python examples/02_berlin_replication/run_v5_score_all.py --no-llm
```

The `--no-llm` flag fails fast on a cache miss rather than falling through to a
live LLM call, so this command consumes no LLM credits. Wall-clock is roughly
five to ten minutes on commodity hardware.

See `examples/02_berlin_replication/README.md` for the full Tier-1-to-Tier-4
reproduction ladder and details on running V1–V4 (which do not require this
cache).
