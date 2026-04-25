"""Loader for railway-shock configuration YAMLs.

Resolves station ``ortsteile_name`` entries to synthetic zone IDs via
``data/berlin/ortsteile/zone_names.csv``.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class RailwayShockConfig:
    name: str
    description: str
    intra_station_min: float
    stations: list[str]               # real Ortsteile names, in line order
    station_synthetic_ids: list[str]  # resolved synthetic zone IDs
    station_roles: list[str]          # "outer-terminus" / "cbd" / etc.


def load_zone_name_map(zone_names_csv: str | Path) -> dict[str, str]:
    """Return ``{ortsteile_name → synthetic_id}``.

    If an Ortsteile appears on multiple synthetic rows (shouldn't with
    our current data), the first one wins.
    """
    mapping: dict[str, str] = {}
    with Path(zone_names_csv).open() as f:
        for row in csv.DictReader(f):
            name = row["ortsteile_name"]
            if name not in mapping:
                mapping[name] = row["synthetic_id"]
    return mapping


def load_shock_config(
    shock_yaml: str | Path,
    zone_names_csv: str | Path,
) -> RailwayShockConfig:
    """Load a shock-config YAML and resolve station names to synthetic IDs."""
    with Path(shock_yaml).open() as f:
        raw = yaml.safe_load(f)

    name = str(raw.get("name", "unnamed"))
    description = str(raw.get("description", "")).strip()
    intra = float(raw["intra_station_min"])
    station_entries = raw["stations"]
    station_names = [s["ortsteile_name"] for s in station_entries]
    station_roles = [s.get("role", "") for s in station_entries]

    name_to_synth = load_zone_name_map(zone_names_csv)
    missing = [n for n in station_names if n not in name_to_synth]
    if missing:
        raise ValueError(
            f"Shock config {name!r}: stations not found in zone_names.csv: "
            f"{missing}"
        )
    synth_ids = [name_to_synth[n] for n in station_names]

    return RailwayShockConfig(
        name=name,
        description=description,
        intra_station_min=intra,
        stations=station_names,
        station_synthetic_ids=synth_ids,
        station_roles=station_roles,
    )
