"""Spatial environment model with zones and transportation network."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from agent_urban_planning.data.loaders import (
    AhlfeldtParams,
    FacilityConfig,
    PolicyConfig,
    ScenarioConfig,
    TransportRouteConfig,
)


@dataclass
class Facility:
    type: str
    capacity: int
    quality: float


@dataclass
class Zone:
    name: str
    housing_supply: int
    housing_base_price: float
    amenity_score: float
    facilities: list[Facility] = field(default_factory=list)
    job_density: float = 0.0
    private_supply: int = 0
    private_base_price: float = 0.0
    # Added in berlin-replication-abm: Ahlfeldt-style fundamentals.
    commercial_floor_area: float = 0.0
    residential_floor_area: float = 0.0
    productivity_A: float = 0.0
    amenity_B: float = 0.0
    wage_observed: float = 0.0
    floor_price_observed: float = 0.0
    # Added in endogenous-agglomeration: raw (pre-agglomeration) fundamentals
    # from Ahlfeldt 2015. When endogenous_agglomeration == True, the market
    # recomputes productivity_A and amenity_B each iteration from these.
    productivity_fundamental_a: float = 0.0
    amenity_fundamental_b: float = 0.0
    # Added in endogenous-land-use: total floor supply per zone (m²).
    # When endogenous_land_use == True, commercial/residential split becomes
    # endogenous and a single unified price P_i clears combined demand vs
    # this total. Defaults to 0.0 — synthesized as
    # ``commercial_floor_area + residential_floor_area`` at construction.
    total_floor_area: float = 0.0

    def has_facility_type(self, facility_type: str) -> bool:
        return any(f.type == facility_type for f in self.facilities)

    def get_facilities_by_type(self, facility_type: str) -> list[Facility]:
        return [f for f in self.facilities if f.type == facility_type]


@dataclass
class TransportRoute:
    from_zone: str
    to_zone: str
    mode: str
    time_minutes: float
    cost_dollars: float


class TransportNetwork:
    """Transportation network between zones."""

    def __init__(self, routes: list[TransportRoute]):
        # Store routes indexed by (from, to) for fast lookup
        self._routes: dict[tuple[str, str], list[TransportRoute]] = {}
        for route in routes:
            key = (route.from_zone, route.to_zone)
            self._routes.setdefault(key, []).append(route)

    def get_routes(self, from_zone: str, to_zone: str) -> list[TransportRoute]:
        """Get all transport routes between two zones."""
        if from_zone == to_zone:
            return [TransportRoute(from_zone, to_zone, "walk", 0.0, 0.0)]
        return self._routes.get((from_zone, to_zone), [])

    def get_best_route(self, from_zone: str, to_zone: str) -> Optional[TransportRoute]:
        """Get the fastest route between two zones."""
        routes = self.get_routes(from_zone, to_zone)
        if not routes:
            return None
        return min(routes, key=lambda r: r.time_minutes)

    def update_route(
        self,
        from_zone: str,
        to_zone: str,
        mode: str,
        time_minutes: float,
        cost_dollars: float,
    ):
        """Add or update a route. If a route with the same mode exists, update it."""
        key = (from_zone, to_zone)
        routes = self._routes.setdefault(key, [])
        for i, r in enumerate(routes):
            if r.mode == mode:
                routes[i] = TransportRoute(from_zone, to_zone, mode, time_minutes, cost_dollars)
                return
        routes.append(TransportRoute(from_zone, to_zone, mode, time_minutes, cost_dollars))

    @property
    def all_routes(self) -> list[TransportRoute]:
        return [r for routes in self._routes.values() for r in routes]


class Environment:
    """Spatial environment holding zones and transportation network.

    Optionally carries an Ahlfeldt-style dense travel-time matrix keyed by
    zone name when the scenario is a Berlin replication. The edge-list
    ``TransportNetwork`` remains the primary interface; ``transport_matrix``
    is a zero-copy alternative used by ``AhlfeldtUtilityEngine`` for
    vectorized distance lookups.
    """

    def __init__(
        self,
        zones: list[Zone],
        transport: TransportNetwork,
        ahlfeldt_params: Optional[AhlfeldtParams] = None,
        transport_matrix: Optional[np.ndarray] = None,
        transport_matrix_index: Optional[list[str]] = None,
    ):
        self.zones = {z.name: z for z in zones}
        self.transport = transport
        # Berlin extensions. Always None / empty for Singapore scenarios.
        self.ahlfeldt_params = ahlfeldt_params
        self.transport_matrix = transport_matrix
        # zone-name list in the row/col order of transport_matrix
        self.transport_matrix_index = list(transport_matrix_index or [])
        # Built on demand for O(1) index lookups
        self._matrix_index_map: dict[str, int] = {
            name: i for i, name in enumerate(self.transport_matrix_index)
        }

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> "Environment":
        zones = [
            Zone(
                name=zc.name,
                housing_supply=zc.housing_supply,
                housing_base_price=zc.housing_base_price,
                amenity_score=zc.amenity_score,
                facilities=[
                    Facility(type=f.type, capacity=f.capacity, quality=f.quality)
                    for f in zc.facilities
                ],
                job_density=zc.job_density,
                private_supply=getattr(zc, "private_supply", 0),
                private_base_price=getattr(zc, "private_base_price", 0.0),
                commercial_floor_area=getattr(zc, "commercial_floor_area", 0.0),
                residential_floor_area=getattr(zc, "residential_floor_area", 0.0),
                productivity_A=getattr(zc, "productivity_A", 0.0),
                amenity_B=getattr(zc, "amenity_B", 0.0),
                wage_observed=getattr(zc, "wage_observed", 0.0),
                floor_price_observed=getattr(zc, "floor_price_observed", 0.0),
                productivity_fundamental_a=getattr(zc, "productivity_fundamental_a", 0.0),
                amenity_fundamental_b=getattr(zc, "amenity_fundamental_b", 0.0),
                # Synthesize total_floor_area when YAML omits it (backward
                # compat): sum of commercial + residential splits from pre-
                # endogenous-land-use scenarios.
                total_floor_area=(
                    float(getattr(zc, "total_floor_area", 0.0))
                    or float(getattr(zc, "commercial_floor_area", 0.0))
                    + float(getattr(zc, "residential_floor_area", 0.0))
                ),
            )
            for zc in config.zones
        ]
        routes = [
            TransportRoute(
                from_zone=r.from_zone,
                to_zone=r.to_zone,
                mode=r.mode,
                time_minutes=r.time_minutes,
                cost_dollars=r.cost_dollars,
            )
            for r in config.transport
        ]

        # Optionally load an Ahlfeldt-style travel-time matrix from NPZ.
        tt_matrix = None
        tt_index = None
        if config.transport_matrix_path:
            tt_path = Path(config.transport_matrix_path)
            if not tt_path.is_absolute():
                # Resolve relative paths against the project root (cwd)
                tt_path = Path.cwd() / tt_path
            with np.load(tt_path, allow_pickle=False) as npz:
                tt_matrix = npz["tt"].astype(np.float64, copy=False)
                if "index" in npz.files:
                    tt_index = [str(x) for x in npz["index"].tolist()]
                else:
                    # Default: assume matrix rows/cols align with zone order.
                    tt_index = [z.name for z in zones]
            if tt_matrix.shape != (len(zones), len(zones)):
                raise ValueError(
                    f"Transport matrix shape {tt_matrix.shape} does not match "
                    f"{len(zones)} zones in scenario '{config.name}'"
                )

        return cls(
            zones,
            TransportNetwork(routes),
            ahlfeldt_params=config.ahlfeldt_params,
            transport_matrix=tt_matrix,
            transport_matrix_index=tt_index,
        )

    def travel_time(self, origin: str, destination: str) -> float:
        """Return the travel time between two zones in minutes.

        Uses the dense ``transport_matrix`` when available (Berlin scenarios);
        falls back to the edge-list ``TransportNetwork`` otherwise.
        Returns ``float('inf')`` if no route exists.
        """
        if self.transport_matrix is not None and origin in self._matrix_index_map and destination in self._matrix_index_map:
            i = self._matrix_index_map[origin]
            j = self._matrix_index_map[destination]
            return float(self.transport_matrix[i, j])
        route = self.transport.get_best_route(origin, destination)
        return float(route.time_minutes) if route is not None else float("inf")

    def get_zone(self, name: str) -> Zone:
        if name not in self.zones:
            raise KeyError(f"Zone '{name}' not found")
        return self.zones[name]

    @property
    def zone_names(self) -> list[str]:
        return list(self.zones.keys())

    def apply_policy(self, policy: PolicyConfig) -> "Environment":
        """Apply a policy and return a new Environment with the changes.

        Does not mutate the original environment.
        """
        new_env = copy.deepcopy(self)

        # Apply facility investments
        for fi in policy.facility_investments:
            zone = new_env.get_zone(fi.zone)
            zone.facilities.append(Facility(
                type=fi.type,
                capacity=fi.capacity,
                quality=fi.quality,
            ))

        # Apply transit investments
        for ti in policy.transit_investments:
            from_zone, to_zone = ti.route
            new_env.transport.update_route(
                from_zone, to_zone, ti.mode,
                ti.new_time_minutes, ti.new_cost_dollars,
            )
            # Also update reverse direction
            new_env.transport.update_route(
                to_zone, from_zone, ti.mode,
                ti.new_time_minutes, ti.new_cost_dollars,
            )

        return new_env
