"""CityFetcher interface and standardized data types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ZoneData:
    name: str
    housing_supply: int
    housing_median_price: float  # monthly rent equivalent
    lat: float
    lon: float
    job_density: float = 0.0
    private_supply: int = 0
    private_median_price: float = 0.0
    job_count: int = 0


@dataclass
class TransportData:
    from_zone: str
    to_zone: str
    mode: str
    time_minutes: float
    cost: float


@dataclass
class ZoneDemographics:
    """Census-derived demographic distribution for one planning area.

    Histograms are lists of (label, count) tuples taken directly from
    Census 2020 tables. The ``population`` field is the total resident
    population used for per-capita calculations.
    """

    income_brackets: list[tuple[str, int]] = field(default_factory=list)
    household_size_hist: list[tuple[int, int]] = field(default_factory=list)
    age_brackets: list[tuple[str, int]] = field(default_factory=list)
    population: int = 0


@dataclass
class DemographicsData:
    income_mean: float
    income_sigma: float
    age_mean: float
    age_std: float
    age_min: int = 22
    age_max: int = 80
    household_size_values: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    household_size_weights: list[float] = field(default_factory=lambda: [0.15, 0.20, 0.25, 0.30, 0.10])
    children_pct: float = 0.40
    elderly_pct: float = 0.20
    car_owner_pct: float = 0.30
    job_location_zones: list[str] = field(default_factory=list)
    job_location_weights: list[float] = field(default_factory=list)
    zone_distributions: Optional[dict[str, ZoneDemographics]] = None


@dataclass
class FacilityData:
    zone: str
    type: str  # hospital, polyclinic, primary_school, secondary_school, mall, hawker_centre, etc.
    name: str
    capacity: int = 0


# ------------------------------------------------------------------
# Town / planning-area name normalization
# ------------------------------------------------------------------

# HDB town names that don't map 1:1 to URA planning areas.
_HDB_TOWN_EXCEPTIONS: dict[str, str] = {
    "KALLANG/WHAMPOA": "kallang",  # primary zone; see _HDB_TOWN_SPLITS for novena share
    "BUKIT BATOK EAST": "bukit_batok",
    "CENTRAL": "central_area",
    "MARINE PARADE": "marine_parade",
}

# HDB towns that span multiple URA planning areas. When a town's unit
# count is loaded, it must be SPLIT across zones by these shares.
#
# KALLANG/WHAMPOA: the HDB town covers both Kallang PA and parts of
# Novena PA (Whampoa estate, Moulmein View). Without this split,
# Novena gets 0 HDB supply which is incorrect.
# Share estimate: ~80% Kallang proper, ~20% Whampoa/Novena area.
# Source: HDB block distribution across subzones.
HDB_TOWN_SPLITS: dict[str, dict[str, float]] = {
    "KALLANG/WHAMPOA": {"kallang": 0.80, "novena": 0.20},
}

# Census 2020 has 55 planning areas; our simulator uses 27 residential
# ones. This table maps Census areas that don't have a direct 1:1
# match to the nearest simulator zone. Non-residential areas (Tuas,
# Lim Chu Kang, etc.) map to None and are skipped.
_CENSUS_AREA_TO_ZONE: dict[str, Optional[str]] = {
    "DOWNTOWN CORE": "central_area",
    "OUTRAM": "central_area",
    "RIVER VALLEY": "central_area",
    "SINGAPORE RIVER": "central_area",
    "MUSEUM": "central_area",
    "ORCHARD": "central_area",
    "MARINA EAST": "central_area",
    "MARINA SOUTH": "central_area",
    "STRAITS VIEW": "central_area",
    "TANGLIN": "bukit_timah",
    "NEWTON": "novena",
    "ROCHOR": "kallang",
    "PAYA LEBAR": "geylang",
    "BOON LAY": "jurong_west",
    "PIONEER": "jurong_west",
    "TENGAH": "bukit_panjang",
    # Non-residential / military / industrial — skip
    "TUAS": None,
    "LIM CHU KANG": None,
    "MANDAI": None,
    "SELETAR": None,
    "SIMPANG": None,
    "SUNGEI KADUT": None,
    "WESTERN WATER CATCHMENT": None,
    "CENTRAL WATER CATCHMENT": None,
    "CHANGI BAY": None,
    "NORTH-EASTERN ISLANDS": None,
    "SOUTHERN ISLANDS": None,
    "WESTERN ISLANDS": None,
}


def normalize_town_name(name: str) -> str:
    """Convert an HDB town name or Census planning-area name to the
    simulator's zone-name format (lowercase, underscores).

    Handles known exceptions (e.g. "KALLANG/WHAMPOA" → "kallang").
    Returns the zone name, or None for non-residential Census areas.
    """
    upper = name.strip().upper()
    if upper in _HDB_TOWN_EXCEPTIONS:
        return _HDB_TOWN_EXCEPTIONS[upper]
    if upper in _CENSUS_AREA_TO_ZONE:
        return _CENSUS_AREA_TO_ZONE[upper]
    return name.strip().lower().replace(" ", "_")


# URA region → planning area mapping for private housing data.
# Every one of the 27 simulator planning areas appears in exactly one region.
URA_REGION_TO_PLANNING_AREAS: dict[str, set[str]] = {
    "CCR": {
        "central_area", "bukit_timah", "novena",
    },
    "RCR": {
        "queenstown", "bukit_merah", "geylang", "kallang",
        "marine_parade", "toa_payoh", "bishan", "serangoon", "clementi",
    },
    "OCR": {
        "ang_mo_kio", "bedok", "bukit_batok", "bukit_panjang",
        "choa_chu_kang", "hougang", "jurong_east", "jurong_west",
        "pasir_ris", "punggol", "sembawang", "sengkang",
        "tampines", "woodlands", "yishun",
    },
}

# Inverse lookup: planning area → region code
PLANNING_AREA_TO_URA_REGION: dict[str, str] = {
    pa: region
    for region, areas in URA_REGION_TO_PLANNING_AREAS.items()
    for pa in areas
}


class CityFetcher(ABC):
    """Abstract base class for city-specific data fetchers."""

    @abstractmethod
    def fetch_zones(self) -> list[ZoneData]:
        """Fetch zone definitions with housing data."""
        ...

    @abstractmethod
    def fetch_transport(self) -> list[TransportData]:
        """Fetch transport routes between zones."""
        ...

    @abstractmethod
    def fetch_demographics(self) -> DemographicsData:
        """Fetch demographic distribution parameters."""
        ...

    @abstractmethod
    def fetch_facilities(self) -> list[FacilityData]:
        """Fetch facility inventory per zone."""
        ...


# Fetcher registry
_REGISTRY: dict[str, type[CityFetcher]] = {}


def register_fetcher(name: str, cls: type[CityFetcher]):
    """Register a city fetcher class."""
    _REGISTRY[name] = cls


def get_fetcher(name: str) -> type[CityFetcher]:
    """Look up a fetcher class by city name."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys()) or "(none)"
        raise KeyError(f"No fetcher registered for city '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_fetchers() -> list[str]:
    """List registered city names."""
    return list(_REGISTRY.keys())
