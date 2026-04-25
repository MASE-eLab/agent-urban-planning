# Shock analysis methodology

## The East-West Express shock

A hypothetical 4-station rapid-transit line cross-cutting Berlin from the
eastern outer suburb (Marzahn) through inner-east (Lichtenberg), the CBD
(Mitte), and terminating in the inner-west (Charlottenburg).

Configuration in {file}`data/berlin/shocks/east_west_express.yaml`:

```yaml
name: east_west_express
description: Hypothetical cross-city express, 4 stations, 5 min between adjacent.
intra_station_min: 5.0
stations:
  - ortsteile_name: Marzahn
    role: outer-east-terminus
  - ortsteile_name: Lichtenberg
    role: inner-east-hub
  - ortsteile_name: Mitte
    role: cbd
  - ortsteile_name: Charlottenburg
    role: inner-west-terminus
```

End-to-end travel time: 3 × 5 = 15 min (roughly halves current east-west
public-transit commute of 30-45 min).

## Route-C min-of-paths

The shock modifies the τ matrix using a Route-C "min over baseline +
via-rail" rule:

```
τ_shock[i, j] = min(
    τ_baseline[i, j],                                       # original route
    min over all (p, q) of (
        τ_baseline[i, station_p]                            # i to nearest station
        + |p - q| × intra_station_min                       # rail leg
        + τ_baseline[station_q, j]                          # furthest station to j
    )
)
```

In words: travelers use the rail line if and only if it makes their
journey faster. OD pairs that don't benefit are unchanged.

Implementation at {func}`agent_urban_planning.research.berlin.railway_shock.apply_railway_shock`.

## Diagnostic: how many OD pairs are affected?

After applying the shock, the tool reports:

```
[shock] τ shock: 164/9216 pairs reduced (mean 5.93 min, max 24.78 min)
```

Only ~1.8% of pairs are affected — a focused, station-local intervention
rather than a city-wide commute overhaul. This focal nature is why
spatial patterns of ΔQ, Δw, ΔHR, ΔHM differ qualitatively across
V1-V5.4: the structural family captures localized accessibility
capitalization at the 4 stations, while V5.4 picks up city-wide
gradient flattening + agglomeration mechanisms.

## Comparing variants' shock responses

Cross-variant moments table (built by
{file}`scripts/build_comparison_moments_table.py` in the dev repo):

```
                       μ ΔQ      σ ΔQ      ΔY%       Δ⟨U⟩  (V1 ruler)
V1 Baseline-softmax    +0.0004  +0.0070   +0.0007    -0.0032
V2 Baseline-ABM argmax +0.0004  +0.0071   -0.0000    -0.0031
V3 Normal-ABM argmax   +0.0004  +0.0042   +0.0132    -0.0027
V4-B Hybrid-ABM        +0.0004  +0.0068   +0.0002    -0.0031
V5.4 LLM-ABM           +0.0016  +0.0083   +0.0299    -0.0056
```

The structural family clusters tightly (consensus); V5.4 stands out on:
- 4× larger μ ΔQ
- 1.2× larger σ ΔQ (more spread)
- 14× larger Δ Q̄ (mean rent rise)
- ~2× larger welfare drop on the V1 ruler
- Only positive aggregate productivity change (ΔY% = +3% vs ~0% for structural)

See the paper's §6 (cross-variant comparison) for the full discussion.

## Reproducing the moments table

After all 5 variants complete (Tier 3 + Tier 4):

```bash
python scripts/build_comparison_moments_table.py
```

Outputs `output/comparison/comparison_moments.csv` and a markdown
companion suitable for paste into the paper.

## See also

- {doc}`berlin_v1_v5_4` — full reproducibility tier ladder
- {doc}`/concepts/decision_engines` — the V1-V5.4 conceptual differences
- {doc}`/tutorials/04_berlin_replication` — task-oriented walkthrough
