#!/usr/bin/env python3
"""Render Berlin Δlog Q and Δlog w choropleths from V1-V5 per_zone.csv outputs.

Reproduces paper Figures 3 and 4: a 1x5 grid (one panel per engine V1 through
V5) showing the log-change in floor price (Δlog Q, Figure 3) and wages
(Δlog w, Figure 4) under the East-West Express transit shock.

Outputs in ``output/comparison/``:
  figure_dlogQ.png   -- paper Figure 3
  figure_dlogw.png   -- paper Figure 4

Usage:
  python examples/02_berlin_replication/plot_dlogQ_dlogw.py

Prerequisite: each ``run_v{1,2,3,4,5}_*.py`` must have run, producing the
per-zone CSVs under ``output/`` for both baseline and shock conditions. Also
requires ``geopandas`` and ``matplotlib``; install with
``pip install -e ".[plot]"``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "berlin"
SHAPEFILE_DIR = REPO_ROOT / "data" / "shapefiles"
OUTPUT_BASE = REPO_ROOT / "output"

VARIANT_ORDER = ("V1", "V2", "V3", "V4", "V5")
VARIANT_DISPLAY_LABELS = {
    "V1": "V1 softmax",
    "V2": "V2 ABM",
    "V3": "V3 Normal",
    "V4": "V4 Hybrid",
    "V5": "V5 LLM",
}

DEFAULT_OUTPUT_SUBDIRS: dict[str, str] = {
    "V1": "berlin_v1_softmax",
    "V2": "berlin_v2_argmax_frechet",
    "V3": "berlin_v3_argmax_normal",
    "V4": "berlin_v4_hybrid",
    "V5": "berlin_v5_score_all",
}

VAR_TO_SIM_COL = {"Q": "Q_sim", "w": "wage_sim"}
VAR_TO_LABEL = {"Q": "Q", "w": "w"}

# Visual reorder for drawing the rail line so segments don't self-cross.
RAILWAY_VISUAL_SEQUENCE: tuple[str, ...] = (
    "Marzahn", "Mitte", "Lichtenberg", "Charlottenburg",
)


# ----- I/O helpers ------------------------------------------------------

def _load_per_zone_col(path: Path, col: str) -> dict[str, float]:
    """Returns {zone_id: float} for the requested column."""
    out: dict[str, float] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                out[r["zone_id"]] = float(r[col])
            except (KeyError, TypeError, ValueError):
                out[r["zone_id"]] = float("nan")
    return out


def _read_crosswalk(path: Path) -> dict[int, str]:
    """Returns {block_int: aggregate_id} skipping comment lines."""
    import io
    with path.open() as f:
        lines = [l for l in f if not l.startswith("#")]
    rows = list(csv.DictReader(io.StringIO("".join(lines))))
    return {
        int(r["block_id"].replace("block_", "")): r["aggregate_id"]
        for r in rows
    }


def _load_zone_name_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            out[r["synthetic_id"]] = r["ortsteile_name"]
    return out


def _dissolve_blocks_to_zones(block_shp: Path, crosswalk: Path):
    """Loads block shapefile, attaches zone id, dissolves to 96 zones.

    Returns (zone_gdf, zone_centroid_dict).
    """
    import geopandas as gpd
    block_gdf = gpd.read_file(block_shp)
    block_gdf["block_int"] = block_gdf["SCHLUESSEL"].astype(int)
    cw = _read_crosswalk(crosswalk)
    block_gdf["z96"] = block_gdf["block_int"].map(cw)
    n_unmatched = int(block_gdf["z96"].isna().sum())
    if n_unmatched > 0:
        raise RuntimeError(
            f"{n_unmatched} blocks have no zone assignment in crosswalk"
        )
    zone_gdf = block_gdf.dissolve(by="z96", as_index=False)[["z96", "geometry"]]
    centroids = {row.z96: row.geometry.centroid for _, row in zone_gdf.iterrows()}
    return zone_gdf, centroids


def _compute_log_change(
    baseline: dict[str, float], shock: dict[str, float],
) -> dict[str, float]:
    """{zone_id: log(shock/baseline)} (NaN where either input is non-positive)."""
    out: dict[str, float] = {}
    for z in baseline:
        b = baseline.get(z, float("nan"))
        s = shock.get(z, float("nan"))
        if not (np.isfinite(b) and np.isfinite(s) and b > 0 and s > 0):
            out[z] = float("nan")
        else:
            out[z] = float(np.log(s / b))
    return out


def _pick_clip_limit(
    values_per_variant: dict[str, dict[str, float]], percentile: float,
) -> float:
    """Returns the chosen percentile of stacked |log change| across variants."""
    stacked = []
    for vmap in values_per_variant.values():
        for v in vmap.values():
            if np.isfinite(v):
                stacked.append(abs(float(v)))
    if not stacked:
        return 1e-6
    return float(np.percentile(np.array(stacked), percentile))


def _load_shock_stations(yaml_path: Path) -> list[str]:
    import yaml
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)
    return [s["ortsteile_name"] for s in cfg["stations"]]


def _station_centroids(
    station_names: list[str], zone_name_map: dict[str, str],
    zone_centroids: dict,
) -> tuple[list[float], list[float], list[str]]:
    name_to_z = {n: z for z, n in zone_name_map.items()}
    xs, ys, picked = [], [], []
    for n in station_names:
        if n in name_to_z:
            z = name_to_z[n]
            if z in zone_centroids:
                c = zone_centroids[z]
                xs.append(c.x)
                ys.append(c.y)
                picked.append(n)
    return xs, ys, picked


def _reorder_for_visual(
    xs: list[float], ys: list[float], labels: list[str],
) -> tuple[list[float], list[float], list[str]]:
    visual_idx = {n: i for i, n in enumerate(RAILWAY_VISUAL_SEQUENCE)}
    fallback = len(RAILWAY_VISUAL_SEQUENCE)
    indexed = sorted(zip(xs, ys, labels),
                     key=lambda t: visual_idx.get(t[2], fallback))
    if not indexed:
        return [], [], []
    xs2, ys2, labels2 = zip(*indexed)
    return list(xs2), list(ys2), list(labels2)


# ----- Plotting ---------------------------------------------------------

def _draw_panel(
    ax, zone_gdf, values_by_z96: dict[str, float],
    cmap: str, vmin: float, vmax: float, title: str,
    railway_xs: list[float] | None = None,
    railway_ys: list[float] | None = None,
    railway_labels: list[str] | None = None,
):
    g = zone_gdf.copy()
    g["value"] = g["z96"].map(values_by_z96)
    g.plot(
        column="value", cmap=cmap, vmin=vmin, vmax=vmax,
        ax=ax, legend=False,
        missing_kwds={"color": "lightgray"},
        edgecolor="#666666", linewidth=0.15,
    )
    if railway_xs and len(railway_xs) >= 2:
        ax.plot(railway_xs, railway_ys, "k-", linewidth=2.4, alpha=0.85,
                zorder=10, solid_capstyle="round")
        ax.scatter(railway_xs, railway_ys, s=50, c="black", marker="o",
                   zorder=11, edgecolors="white", linewidths=1.2)
        if railway_labels:
            for x, y, label in zip(railway_xs, railway_ys, railway_labels):
                ax.annotate(label, (x, y), xytext=(4, 4),
                            textcoords="offset points",
                            fontsize=6, fontweight="bold", color="black",
                            bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                      ec="black", alpha=0.85, linewidth=0.4),
                            zorder=12)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def plot_log_change_grid(
    var: str, zone_gdf,
    log_change_by_variant: dict[str, dict[str, float]],
    railway_xs: list[float], railway_ys: list[float], railway_labels: list[str],
    clip_percentile: float, output_path: Path,
):
    """Render a 1x5 grid (V1-V5) for one variable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    var_label = VAR_TO_LABEL[var]
    diff_lim = _pick_clip_limit(log_change_by_variant, clip_percentile)
    if diff_lim <= 0:
        diff_lim = 1e-6

    fig, axes = plt.subplots(
        1, 5, figsize=(20, 5.5),
        gridspec_kw={"wspace": 0.02},
    )
    for col, variant in enumerate(VARIANT_ORDER):
        values = log_change_by_variant.get(variant, {})
        display = VARIANT_DISPLAY_LABELS[variant]
        _draw_panel(
            axes[col], zone_gdf, values,
            cmap="RdBu_r", vmin=-diff_lim, vmax=+diff_lim,
            title=f"{display}",
            railway_xs=railway_xs, railway_ys=railway_ys,
            railway_labels=railway_labels,
        )

    div_sm = ScalarMappable(cmap="RdBu_r", norm=Normalize(-diff_lim, +diff_lim))
    div_sm.set_array([])
    cb = fig.colorbar(
        div_sm, ax=list(axes), fraction=0.018, pad=0.01,
        shrink=0.75, orientation="vertical",
    )
    cb.set_label(
        f"log(shock {var_label} / baseline {var_label})  "
        f"[clipped at {clip_percentile:.0f}-th pct = +/- {diff_lim:.4f}]",
        fontsize=9,
    )
    cb.ax.tick_params(labelsize=8)
    fig.suptitle(
        f"Per-zone Delta log {var_label} under East-West Express shock",
        fontsize=12, y=0.99,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----- Main -------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_ORDER))
    parser.add_argument("--shapefile", type=Path,
                        default=SHAPEFILE_DIR / "Berlin4matlab.shp")
    parser.add_argument("--crosswalk", type=Path,
                        default=DATA_DIR / "crosswalks" / "blocks_to_ortsteile.csv")
    parser.add_argument("--zone-names", type=Path,
                        default=DATA_DIR / "ortsteile" / "zone_names.csv")
    parser.add_argument("--shock-yaml", type=Path,
                        default=DATA_DIR / "shocks" / "east_west_express.yaml")
    parser.add_argument("--output-dir", type=Path,
                        default=OUTPUT_BASE / "comparison")
    parser.add_argument("--clip-percentile", type=float, default=95.0)
    args = parser.parse_args()

    # Verify inputs.
    missing = []
    paths: dict[str, dict[str, Path]] = {}
    for v in args.variants:
        sd = DEFAULT_OUTPUT_SUBDIRS[v]
        b = OUTPUT_BASE / sd / "per_zone.csv"
        s = OUTPUT_BASE / f"{sd}_shock_east_west" / "per_zone.csv"
        paths[v] = {"baseline": b, "shock": s}
        for kind, p in paths[v].items():
            if not p.exists():
                missing.append(f"  {v} {kind}: {p}")
    if missing:
        print("ERROR: missing per_zone.csv files. Run the relevant "
              "run_v*.py scripts first.", file=sys.stderr)
        for m in missing:
            print(m, file=sys.stderr)
        return 4
    for p in (args.shapefile, args.crosswalk, args.zone_names, args.shock_yaml):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 4

    # Geometry: load + dissolve blocks to zones.
    print(f"[plot] loading shapefile + dissolving to 96 zones...", flush=True)
    zone_gdf, zone_centroids = _dissolve_blocks_to_zones(
        args.shapefile, args.crosswalk,
    )

    # Railway overlay.
    zone_name_map = _load_zone_name_map(args.zone_names)
    station_names = _load_shock_stations(args.shock_yaml)
    rail_xs, rail_ys, rail_labels = _station_centroids(
        station_names, zone_name_map, zone_centroids,
    )
    rail_xs, rail_ys, rail_labels = _reorder_for_visual(
        rail_xs, rail_ys, rail_labels,
    )

    # Per-variant per-zone log changes for Q and w.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for var in ("Q", "w"):
        col = VAR_TO_SIM_COL[var]
        log_change_by_variant: dict[str, dict[str, float]] = {}
        for v in args.variants:
            print(f"[plot] {var} {v}", flush=True)
            base = _load_per_zone_col(paths[v]["baseline"], col)
            shock = _load_per_zone_col(paths[v]["shock"], col)
            log_change_by_variant[v] = _compute_log_change(base, shock)
        out = args.output_dir / f"figure_dlog{var}.png"
        plot_log_change_grid(
            var, zone_gdf, log_change_by_variant,
            rail_xs, rail_ys, rail_labels,
            args.clip_percentile, out,
        )
        print(f"[plot] {var} -> {out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
