"""Warm-start helpers for policy-shock runs.

Read baseline equilibrium ``(Q, w)`` from a variant's ``per_zone.csv``
and inject those values into a fresh scenario so the shocked run's
tâtonnement starts at the baseline equilibrium rather than at scenario
defaults.

Typical usage::

    Q_dict, w_dict = load_baseline_prices(
        "output/berlin_v2_argmax_frechet/per_zone.csv",
        zone_index=["z96_000", "z96_001", ...],
    )
    scenario = inject_initial_prices(baseline_scenario, Q_dict, w_dict)
"""
from __future__ import annotations

import copy
import csv
import logging
import math
from pathlib import Path
from typing import Sequence


logger = logging.getLogger(__name__)


_MIN_POSITIVE_PRICE = 1e-2  # floor for zero-marginal baseline zones
_REQUIRED_COLS = ("zone_id", "Q_sim", "wage_sim")


def load_baseline_prices(
    per_zone_csv_path: str | Path,
    zone_index: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Read baseline ``Q`` and ``w`` per zone from a variant's ``per_zone.csv``.

    Args:
        per_zone_csv_path: path to baseline ``per_zone.csv``.
        zone_index: expected zone IDs, length-N. Mismatch raises
            ``ValueError``.

    Returns:
        ``(Q_dict, w_dict)``: two dicts keyed by zone_id with float values.
        Values ≤ 0 are floored at ``_MIN_POSITIVE_PRICE`` (=1e-2) and a
        warning is logged naming the affected zones.

    Raises:
        FileNotFoundError: path doesn't exist.
        ValueError: missing required columns, or zone-index mismatch.
    """
    path = Path(per_zone_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline per_zone.csv not found: {path}")

    rows: list[dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing_cols = [c for c in _REQUIRED_COLS if c not in fieldnames]
        if missing_cols:
            raise ValueError(
                f"{path}: missing required columns {missing_cols}; "
                f"have {fieldnames}"
            )
        for row in reader:
            rows.append(row)

    csv_zone_ids = {r["zone_id"] for r in rows}
    expected = set(zone_index)
    missing_in_csv = expected - csv_zone_ids
    extra_in_csv = csv_zone_ids - expected
    if missing_in_csv:
        raise ValueError(
            f"{path}: missing zones from baseline CSV "
            f"(first 5): {sorted(missing_in_csv)[:5]}"
        )
    if extra_in_csv:
        logger.warning(
            "Baseline CSV has zones not in zone_index (first 5): %s",
            sorted(extra_in_csv)[:5],
        )

    Q_dict: dict[str, float] = {}
    w_dict: dict[str, float] = {}
    floored_zones: list[tuple[str, str, float]] = []
    for r in rows:
        zid = r["zone_id"]
        if zid not in expected:
            continue
        Q_val = _parse_float(r["Q_sim"], f"{zid}.Q_sim")
        w_val = _parse_float(r["wage_sim"], f"{zid}.wage_sim")
        if Q_val <= 0 or not math.isfinite(Q_val):
            floored_zones.append((zid, "Q_sim", Q_val))
            Q_val = _MIN_POSITIVE_PRICE
        if w_val <= 0 or not math.isfinite(w_val):
            floored_zones.append((zid, "wage_sim", w_val))
            w_val = _MIN_POSITIVE_PRICE
        Q_dict[zid] = Q_val
        w_dict[zid] = w_val

    if floored_zones:
        sample = floored_zones[:5]
        logger.warning(
            "Floored %d non-positive baseline values to %.3f; first 5: %s",
            len(floored_zones), _MIN_POSITIVE_PRICE, sample,
        )

    return Q_dict, w_dict


def _parse_float(text: str, context: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        logger.warning("Could not parse %s=%r as float; using 0.0", context, text)
        return 0.0


def inject_initial_prices(scenario, Q_dict: dict[str, float], w_dict: dict[str, float]):
    """Return a deep-copied scenario with zone-level ``floor_price`` and
    ``wage_observed`` overridden by the baseline values.

    The scenario object has ``zones`` — a list of zone objects with
    attributes like ``floor_price_observed`` and ``wage_observed``. We
    deep-copy the scenario (including its zones) and overwrite those
    fields for each zone present in ``Q_dict`` / ``w_dict``. Zones NOT
    in the dict are left untouched.

    Args:
        scenario: Scenario-like object with ``.zones`` attribute.
        Q_dict: ``{zone_id → Q init}``
        w_dict: ``{zone_id → w init}``

    Returns:
        A modified copy of the scenario. The input object is NOT mutated.
    """
    new_scenario = copy.deepcopy(scenario)

    zones_attr = getattr(new_scenario, "zones", None)
    if zones_attr is None:
        raise ValueError("scenario has no .zones attribute")

    n_updated = 0
    for zone in zones_attr:
        zid = getattr(zone, "name", None)
        if zid is None or zid not in Q_dict:
            continue
        if hasattr(zone, "floor_price_observed"):
            zone.floor_price_observed = float(Q_dict[zid])
        if hasattr(zone, "wage_observed"):
            zone.wage_observed = float(w_dict[zid])
        n_updated += 1

    logger.info("Warm-start: injected baseline Q, w into %d zones", n_updated)
    return new_scenario
