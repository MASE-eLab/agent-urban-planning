#!/usr/bin/env python3
"""Build the cross-variant moments table from V1-V5 per_zone.csv outputs.

Aggregates the per-zone CSVs produced by the five ``run_v*.py`` scripts into
the cross-variant moments table reported as Table 2 (abridged) and Table 5
(full) in the paper.

Outputs in ``output/comparison/``:
  comparison_moments_abridged.csv  -- paper Table 2 (7 columns)
  comparison_moments_full.csv      -- paper Table 5 (10 columns)
  comparison_moments.md            -- GFM table for the README

Usage:
  python examples/02_berlin_replication/build_moments_table.py

Prerequisite: each ``run_v{1,2,3,4,5}_*.py`` must have already run, producing
the per-zone CSVs under ``output/`` for both baseline and shock conditions.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "berlin"
OUTPUT_BASE = REPO_ROOT / "output"

# Internal variant IDs and their display labels.
VARIANT_ORDER = ("V1", "V2", "V3", "V4", "V5")
VARIANT_DISPLAY_LABELS = {
    "V1": "V1 softmax",
    "V2": "V2 ABM",
    "V3": "V3 Normal",
    "V4": "V4 Hybrid",
    "V5": "V5 LLM",
}

# Per-variant output directories. Each ``run_v*.py`` writes to
# ``output/{subdir}/per_zone.csv`` (baseline) and
# ``output/{subdir}_shock_east_west/per_zone.csv`` (shock).
DEFAULT_OUTPUT_SUBDIRS: dict[str, str] = {
    "V1": "berlin_v1_softmax",
    "V2": "berlin_v2_argmax_frechet",
    "V3": "berlin_v3_argmax_normal",
    "V4": "berlin_v4_hybrid",
    "V5": "berlin_v5_score_all",
}

# Ahlfeldt 2015 calibrated elasticities (Berlin 2006 instance).
ALPHA = 0.80         # firm Cobb-Douglas labour share
BETA = 0.75          # household Cobb-Douglas housing share
KAPPA_EPS = 0.0987   # commute decay × Frechet shape (kappa × epsilon)


# ----- I/O helpers -------------------------------------------------------

def _load_per_zone_csv(path: Path) -> dict[str, dict[str, float]]:
    """Returns {zone_id: {col_name: float}} for all numeric columns."""
    out: dict[str, dict[str, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            zid = row["zone_id"]
            out[zid] = {}
            for k, v in row.items():
                if k == "zone_id":
                    continue
                try:
                    out[zid][k] = float(v)
                except (TypeError, ValueError):
                    out[zid][k] = float("nan")
    return out


def _load_zone_attributes(scenario_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Returns (A_by_zone, B_by_zone) from the scenario YAML.

    Zones are keyed by synthetic id (e.g. ``z96_NNN``) in the YAML.
    """
    import yaml
    with scenario_path.open() as f:
        cfg = yaml.safe_load(f)
    A: dict[str, float] = {}
    B: dict[str, float] = {}
    zones = cfg["zones"]
    if isinstance(zones, dict):
        for synth_id, attrs in zones.items():
            A[synth_id] = float(attrs["productivity_A"])
            B[synth_id] = float(attrs["amenity_B"])
    else:
        for z in zones:
            A[z["name"]] = float(z["productivity_A"])
            B[z["name"]] = float(z["amenity_B"])
    return A, B


def _load_zone_name_map(path: Path) -> dict[str, str]:
    """Returns {synthetic_id: ortsteile_name} from the zone-names CSV."""
    out: dict[str, str] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["synthetic_id"]] = row["ortsteile_name"]
    return out


def _load_tau(npz_path: Path) -> tuple[np.ndarray, list[str]]:
    """Loads (tau, zone_index) from a tt npz."""
    with np.load(npz_path, allow_pickle=False) as npz:
        tau = npz["tt"].astype(np.float64, copy=False)
        idx = [str(x) for x in npz["index"].tolist()]
    return tau, idx


def _apply_shock(
    tau: np.ndarray, zone_index: list[str],
    shock_yaml: Path, zone_name_map: dict[str, str],
) -> np.ndarray:
    """Apply the East-West Express tau shock via the bundled helper."""
    from agent_urban_planning.research.berlin.railway_shock import apply_railway_shock
    import yaml
    with shock_yaml.open() as f:
        cfg = yaml.safe_load(f)
    name_to_z = {n: z for z, n in zone_name_map.items()}
    station_zones = [name_to_z[s["ortsteile_name"]] for s in cfg["stations"]]
    intra_min = float(cfg.get("intra_station_min", 5.0))
    return apply_railway_shock(tau, zone_index, station_zones, intra_min)


# ----- Block 1: distribution-shape moments ------------------------------

def _compute_distribution_moments(
    base: dict[str, dict[str, float]], shock: dict[str, dict[str, float]],
) -> dict[str, float]:
    """{mu_dlogQ, sigma_dlogQ, mu_dlogw, sigma_dlogw, p95_abs_dlogQ}."""
    dQs, dws = [], []
    for z, br in base.items():
        sr = shock.get(z, {})
        Qb, Qs = br.get("Q_sim"), sr.get("Q_sim")
        wb, ws = br.get("wage_sim"), sr.get("wage_sim")
        if Qb and Qs and Qb > 0 and Qs > 0:
            dQs.append(math.log(Qs / Qb))
        if wb and ws and wb > 0 and ws > 0:
            dws.append(math.log(ws / wb))
    dQ = np.asarray(dQs, dtype=np.float64)
    dw = np.asarray(dws, dtype=np.float64)
    return {
        "mu_dlogQ":      float(dQ.mean()) if dQ.size else float("nan"),
        "sigma_dlogQ":   float(dQ.std())  if dQ.size else float("nan"),
        "mu_dlogw":      float(dw.mean()) if dw.size else float("nan"),
        "sigma_dlogw":   float(dw.std())  if dw.size else float("nan"),
        "p95_abs_dlogQ": float(np.percentile(np.abs(dQ), 95)) if dQ.size else float("nan"),
    }


# ----- Block 2: aggregate magnitudes ------------------------------------

def _compute_aggregates(
    base: dict[str, dict[str, float]], shock: dict[str, dict[str, float]],
    A_by_zone: dict[str, float], alpha: float = ALPHA,
) -> dict[str, float]:
    """{delta_Y_pct, delta_W_bar, delta_Q_bar}.

    Y      = sum_j A_j * HM_j^alpha  (Cobb-Douglas with exogenous A_j)
    W_bar  = HM-weighted mean wage
    Q_bar  = HR-weighted mean floor price
    """
    def _Y(per_zone):
        Y = 0.0
        for z, row in per_zone.items():
            HM = row.get("HM_sim", 0.0)
            A = A_by_zone.get(z, 1.0)
            if HM > 0:
                Y += A * (HM ** alpha)
        return Y

    def _Wbar(per_zone):
        num, den = 0.0, 0.0
        for z, row in per_zone.items():
            HM = row.get("HM_sim", 0.0)
            w = row.get("wage_sim", float("nan"))
            if HM > 0 and np.isfinite(w):
                num += HM * w
                den += HM
        return num / den if den > 0 else float("nan")

    def _Qbar(per_zone):
        num, den = 0.0, 0.0
        for z, row in per_zone.items():
            HR = row.get("HR_sim", 0.0)
            Q = row.get("Q_sim", float("nan"))
            if HR > 0 and np.isfinite(Q):
                num += HR * Q
                den += HR
        return num / den if den > 0 else float("nan")

    Y_b, Y_s = _Y(base), _Y(shock)
    Wb_b, Wb_s = _Wbar(base), _Wbar(shock)
    Qb_b, Qb_s = _Qbar(base), _Qbar(shock)
    return {
        "delta_Y_pct": (Y_s - Y_b) / Y_b if Y_b > 0 else float("nan"),
        "delta_W_bar": math.log(Wb_s / Wb_b) if Wb_b > 0 and Wb_s > 0 else float("nan"),
        "delta_Q_bar": math.log(Qb_s / Qb_b) if Qb_b > 0 and Qb_s > 0 else float("nan"),
    }


# ----- Block 3: welfare under V1 ruler (Cobb-Douglas + Frechet) --------

def _compute_welfare_v1_ruler(
    per_zone: dict[str, dict[str, float]],
    B_by_zone: dict[str, float], tau: np.ndarray, zone_index: list[str],
    beta: float = BETA, kappa_eps: float = KAPPA_EPS,
) -> float:
    """Population-weighted average per-agent utility under the V1 ruler.

    P_ij ~= HR_i * HM_j * exp(-kappa*eps * tau_ij), normalised to sum 1.
    U_ij  = log B_i + (1-beta) log w_j - (1-beta) log Q_i - kappa*eps * tau_ij.
    <U>   = sum_{i,j} P_ij * U_ij.
    """
    N = len(zone_index)
    Q = np.zeros(N); w = np.zeros(N); B = np.zeros(N); HR = np.zeros(N); HM = np.zeros(N)
    for i, z in enumerate(zone_index):
        row = per_zone.get(z, {})
        Q[i] = row.get("Q_sim", float("nan"))
        w[i] = row.get("wage_sim", float("nan"))
        HR[i] = row.get("HR_sim", 0.0)
        HM[i] = row.get("HM_sim", 0.0)
        B[i] = B_by_zone.get(z, 1.0)
    valid_i = (Q > 0) & np.isfinite(Q) & (HR > 0)
    valid_j = (w > 0) & np.isfinite(w) & (HM > 0)
    if not valid_i.any() or not valid_j.any():
        return float("nan")

    weight = (HR[:, None] * HM[None, :]) * np.exp(-kappa_eps * tau)
    weight[~valid_i, :] = 0.0
    weight[:, ~valid_j] = 0.0
    total = weight.sum()
    if total <= 0:
        return float("nan")
    P = weight / total

    log_B = np.log(np.where(B > 0, B, 1e-12))
    log_w = np.log(np.where(w > 0, w, 1e-12))
    log_Q = np.log(np.where(Q > 0, Q, 1e-12))
    U = (log_B[:, None] + (1.0 - beta) * log_w[None, :]
         - (1.0 - beta) * log_Q[:, None] - kappa_eps * tau)
    return float((P * U).sum())


# ----- Output rendering -------------------------------------------------

ABRIDGED_COLS = (
    "mu_dlogQ", "sigma_dlogQ", "p95_abs_dlogQ",
    "delta_Q_bar", "delta_Y_pct", "mu_dlogw", "delta_U_v1",
)

FULL_COLS = (
    "mu_dlogQ", "sigma_dlogQ", "mu_dlogw", "sigma_dlogw", "p95_abs_dlogQ",
    "delta_Y_pct", "delta_W_bar", "delta_Q_bar",
    "U_baseline", "delta_U_v1",
)


def _write_csv(rows: dict[str, dict[str, float]], cols: tuple[str, ...], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant"] + list(cols))
        for v in VARIANT_ORDER:
            r = rows[v]
            label = VARIANT_DISPLAY_LABELS[v]
            w.writerow([label] + [f"{r.get(c, float('nan')):.6f}" for c in cols])


def _write_md(rows: dict[str, dict[str, float]], out_path: Path):
    lines = [
        "# Berlin East-West Express — cross-variant moments\n",
        "Distribution-shape moments are across the 96 Ortsteile. "
        "`delta_U_v1` applies the V1 (Baseline-softmax) Cobb-Douglas + Frechet welfare "
        "formula uniformly to each variant's output state.\n",
    ]
    header = ["Variant"] + list(FULL_COLS)
    aligns = [":---"] + ["---:"] * len(FULL_COLS)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(aligns) + " |")
    for v in VARIANT_ORDER:
        r = rows[v]
        label = VARIANT_DISPLAY_LABELS[v]
        cells = [f"{r.get(c, float('nan')):+.4f}" for c in FULL_COLS]
        if v == "V5":
            label = f"**{label}**"
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join([label] + cells) + " |")
    out_path.write_text("\n".join(lines) + "\n")


# ----- Main -------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_ORDER),
                        help="Internal variant IDs (default: all five).")
    parser.add_argument("--scenario", type=Path,
                        default=DATA_DIR / "scenarios" / "berlin_2006_ortsteile.yaml")
    parser.add_argument("--zone-names", type=Path,
                        default=DATA_DIR / "ortsteile" / "zone_names.csv")
    parser.add_argument("--tau-npz", type=Path,
                        default=DATA_DIR / "ortsteile" / "tt_2006_actual.npz")
    parser.add_argument("--shock-yaml", type=Path,
                        default=DATA_DIR / "shocks" / "east_west_express.yaml")
    parser.add_argument("--output-dir", type=Path,
                        default=OUTPUT_BASE / "comparison")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Verify per-variant CSVs are on disk.
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

    # Load shared inputs.
    print(f"[moments] loading scenario + zone names + tau", flush=True)
    A_by_zone, B_by_zone = _load_zone_attributes(args.scenario)
    zone_name_map = _load_zone_name_map(args.zone_names)
    tau_baseline, zone_index = _load_tau(args.tau_npz)
    if len(zone_index) != 96:
        print(f"WARN: tau matrix has {len(zone_index)} zones, expected 96",
              file=sys.stderr)
    tau_shock = _apply_shock(tau_baseline, zone_index, args.shock_yaml, zone_name_map)

    # Compute per-variant rows.
    rows: dict[str, dict[str, float]] = {}
    for v in args.variants:
        print(f"[moments] {v} ({VARIANT_DISPLAY_LABELS[v]})", flush=True)
        base = _load_per_zone_csv(paths[v]["baseline"])
        shock = _load_per_zone_csv(paths[v]["shock"])

        moments = _compute_distribution_moments(base, shock)
        aggregates = _compute_aggregates(base, shock, A_by_zone)
        U_b = _compute_welfare_v1_ruler(base, B_by_zone, tau_baseline, zone_index)
        U_s = _compute_welfare_v1_ruler(shock, B_by_zone, tau_shock, zone_index)
        rows[v] = {
            **moments, **aggregates,
            "U_baseline": U_b,
            "delta_U_v1": U_s - U_b,
        }

    # Write outputs.
    abridged_path = args.output_dir / "comparison_moments_abridged.csv"
    full_path = args.output_dir / "comparison_moments_full.csv"
    md_path = args.output_dir / "comparison_moments.md"
    _write_csv(rows, ABRIDGED_COLS, abridged_path)
    _write_csv(rows, FULL_COLS, full_path)
    _write_md(rows, md_path)
    print(f"[moments] abridged (paper Table 2) -> {abridged_path}", flush=True)
    print(f"[moments] full (paper Table 5)     -> {full_path}", flush=True)
    print(f"[moments] markdown for README      -> {md_path}", flush=True)

    print()
    print(md_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
