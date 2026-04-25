"""Route-C graph-topology travel-time shock for a new railway line.

Given a pre-computed N×N τ matrix and a list of K station zones, compute
the shocked τ matrix by taking the min over direct and via-railway paths:

  new_τ[i, j] = min(
    old_τ[i, j],
    min over (p, q) of: old_τ[i, a_p] + T_rail(p, q) + old_τ[a_q, j]
  )

where ``T_rail(p, q) = |p - q| × intra_station_min`` is the cumulative
rail travel time between stations p and q (stations indexed along the
line).

At N=96, K=4 this completes in well under a second — fully vectorized
via numpy broadcasting.

Preserves symmetry iff input τ is symmetric. Idempotent on already-shocked
input.

Usage::

    from agent_urban_planning.research.berlin.railway_shock import (
        apply_railway_shock,
    )

    shocked_tt = apply_railway_shock(
        tt=env.transport_matrix,
        zone_index=env.transport_matrix_index,
        stations=["z96_012", "z96_035", "z96_000", "z96_076"],
        intra_station_min=5.0,
    )
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def apply_railway_shock(
    tt: np.ndarray,
    zone_index: Sequence[str],
    stations: Sequence[str],
    intra_station_min: float,
) -> np.ndarray:
    """Apply a Route-C graph-topology τ shock for a new railway line.

    Args:
        tt: shape (N, N) baseline τ matrix (units: minutes).
        zone_index: length-N list of zone IDs in tt's index order.
        stations: K station zone IDs (must be a subset of zone_index),
            ordered along the railway line.
        intra_station_min: travel time (minutes) between consecutive
            stations. ``T_rail(p, q) = |p - q| × intra_station_min``.

    Returns:
        shape (N, N) shocked τ matrix. Equal to ``tt`` for OD pairs not
        benefiting from the railway; strictly less than ``tt`` for pairs
        where via-railway is faster.

    Raises:
        ValueError: if any station is not in ``zone_index``, or if
            ``stations`` has fewer than 2 entries.
    """
    tt = np.asarray(tt, dtype=np.float64)
    if tt.ndim != 2 or tt.shape[0] != tt.shape[1]:
        raise ValueError(f"tt must be square 2D; got shape {tt.shape}")
    N = tt.shape[0]
    if len(zone_index) != N:
        raise ValueError(
            f"zone_index length ({len(zone_index)}) must match tt.shape[0] ({N})"
        )
    if len(stations) < 2:
        raise ValueError(
            f"stations must have at least 2 entries; got {len(stations)}"
        )

    name_to_idx = {z: i for i, z in enumerate(zone_index)}
    missing = [s for s in stations if s not in name_to_idx]
    if missing:
        raise ValueError(
            f"stations not in zone_index: {missing}"
        )
    station_idx = np.array([name_to_idx[s] for s in stations], dtype=np.int64)
    k = len(station_idx)
    intra = float(intra_station_min)

    # T_rail[p, q] = |p - q| * intra_station_min, a (k, k) matrix.
    p_range = np.arange(k, dtype=np.float64)
    T_rail = np.abs(p_range[:, None] - p_range[None, :]) * intra

    # Via-station extracts.
    tt_to_stations = tt[:, station_idx]       # shape (N, k)
    tt_from_stations = tt[station_idx, :]     # shape (k, N)

    # a[i, q] = min_p (tt_to_stations[i, p] + T_rail[p, q]).
    a = (tt_to_stations[:, :, None] + T_rail[None, :, :]).min(axis=1)
    # via_min[i, j] = min_q (a[i, q] + tt_from_stations[q, j]).
    via_min = (a[:, :, None] + tt_from_stations[None, :, :]).min(axis=1)

    shocked = np.minimum(tt, via_min)
    return shocked


def apply_railway_shock_with_diagnostics(
    tt: np.ndarray,
    zone_index: Sequence[str],
    stations: Sequence[str],
    intra_station_min: float,
) -> tuple[np.ndarray, dict]:
    """Apply shock + return ``(shocked_matrix, diagnostic metadata)``."""
    shocked = apply_railway_shock(tt, zone_index, stations, intra_station_min)
    diff = np.asarray(tt, dtype=np.float64) - shocked
    diff = np.maximum(diff, 0.0)
    reduced = diff > 1e-9
    n_pairs_reduced = int(reduced.sum())
    if n_pairs_reduced > 0:
        mean_reduction = float(diff[reduced].mean())
        max_reduction = float(diff[reduced].max())
    else:
        mean_reduction = 0.0
        max_reduction = 0.0
    diagnostics = {
        "n_pairs_reduced": n_pairs_reduced,
        "n_pairs_total": int(tt.size),
        "fraction_reduced": n_pairs_reduced / float(tt.size) if tt.size else 0.0,
        "mean_reduction_min": mean_reduction,
        "max_reduction_min": max_reduction,
        "min_tau_shocked": float(shocked.min()),
        "min_tau_baseline": float(np.asarray(tt).min()),
        "n_stations": len(stations),
        "intra_station_min": float(intra_station_min),
    }
    return shocked, diagnostics
