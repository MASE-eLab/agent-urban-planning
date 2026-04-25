"""Price-bucketed LLM call cache.

Caches LLM responses keyed on (entity_id, price_bucket_tuple) so that late
market-clearing iterations — where prices barely move — hit the cache and
skip redundant API calls.

Bucketing rounds each zone price to the nearest `price_bucket_pct` of its
*reference* (base) price, producing a stable hashable key tuple even when
exact prices drift slightly between iterations.

The default cache is in-memory and per-run by design: it is created at the
start of HousingMarket.clear() and discarded at the end, avoiding cross-run
correctness pitfalls. Long-running comparison scripts may optionally use the
disk-backed variant to survive crashes inside a single run.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Iterable, Optional

from agent_urban_planning.decisions.base import LocationChoice, ZoneChoice


def make_price_bucket_key(
    prices: dict[str, float],
    base_prices: dict[str, float],
    bucket_pct: float = 0.05,
) -> tuple:
    """Round each zone price to the nearest `bucket_pct` of its base price.

    Returns a hashable tuple of (zone, bucket_index) pairs sorted by zone.

    Args:
        prices: Current zone -> price mapping.
        base_prices: Reference zone -> price mapping (used as the bucket
            reference scale).
        bucket_pct: Bucket size as a fraction of base price (default 5%).

    Example:
        prices={a: 3520, b: 1810}, base={a: 3500, b: 1800}, bucket=0.05
        bucket_a = round(3520 / (3500 * 0.05)) = round(20.11) = 20
        bucket_b = round(1810 / (1800 * 0.05)) = round(20.11) = 20
        result = (("a", 20), ("b", 20))
    """
    if bucket_pct <= 0:
        raise ValueError(f"bucket_pct must be > 0, got {bucket_pct}")

    keys = sorted(prices.keys())
    bucketed = []
    for zone in keys:
        price = prices[zone]
        base = base_prices.get(zone, price)
        if base <= 0:
            base = 1.0
        bucket_size = base * bucket_pct
        bucket_idx = round(price / bucket_size) if bucket_size > 0 else 0
        bucketed.append((zone, bucket_idx))
    return tuple(bucketed)


class LLMCallCache:
    """In-memory cache for LLM responses keyed on (entity_id, price_bucket).

    The cache lives for one market clearing pass. Construct it at the start
    of HousingMarket.clear(), use it during iterations, and discard it when
    clearing finishes.

    Args:
        base_prices: Reference price mapping for bucketing. Typically the
            zone base prices from the environment.
        price_bucket_pct: Bucket size as a fraction of base price (default 0.05).
        enabled: Set to False to disable caching entirely (every get() returns
            None and `hits` stays 0). Useful for benchmarking.
    """

    def __init__(
        self,
        base_prices: dict[str, float],
        price_bucket_pct: float = 0.05,
        enabled: bool = True,
    ):
        self.base_prices = dict(base_prices)
        self.price_bucket_pct = price_bucket_pct
        self.enabled = enabled
        self._store: dict[tuple, Any] = {}
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------

    def make_key(self, entity_id: int, prices: dict[str, float]) -> tuple:
        """Construct a cache key for one entity (agent or archetype) at given prices."""
        bucket = make_price_bucket_key(prices, self.base_prices, self.price_bucket_pct)
        return (entity_id, bucket)

    def get(self, entity_id: int, prices: dict[str, float]) -> Optional[Any]:
        if not self.enabled:
            self.misses += 1
            return None
        key = self.make_key(entity_id, prices)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def put(self, entity_id: int, prices: dict[str, float], value: Any):
        if not self.enabled:
            return
        key = self.make_key(entity_id, prices)
        self._store[key] = value

    def clear(self):
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def flush(self) -> None:
        """Persist any pending state if the cache implementation supports it."""
        return None

    def close(self) -> None:
        """Release any resources held by the cache."""
        self.flush()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def total_lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.hits / self.total_lookups

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, item) -> bool:
        return item in self._store


class DiskBackedLLMCallCache(LLMCallCache):
    """Run-scoped cache persisted to disk.

    The cache file is rewritten atomically on ``flush()``. Callers are
    expected to scope each file to a single simulation run and policy so
    cached LLM decisions are never reused across unrelated experiments.
    """

    def __init__(
        self,
        base_prices: dict[str, float],
        path: str,
        price_bucket_pct: float = 0.05,
        enabled: bool = True,
    ):
        super().__init__(
            base_prices=base_prices,
            price_bucket_pct=price_bucket_pct,
            enabled=enabled,
        )
        self.path = path
        self._dirty = False
        self._load_from_disk()

    def put(self, entity_id: int, prices: dict[str, float], value: Any):
        if not self.enabled:
            return
        super().put(entity_id, prices, value)
        self._dirty = True

    def clear(self):
        super().clear()
        self._dirty = True
        self.flush()

    def flush(self) -> None:
        if not self.enabled or not self._dirty:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "version": 1,
            "base_prices": self.base_prices,
            "price_bucket_pct": self.price_bucket_pct,
            "entries": [
                {
                    "key": self._serialize_key(key),
                    "value": self._serialize_value(value),
                }
                for key, value in self._store.items()
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=os.path.dirname(self.path) or ".",
            suffix=".tmp",
        ) as tmp:
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp_path = tmp.name
        os.replace(tmp_path, self.path)
        self._dirty = False

    def _load_from_disk(self) -> None:
        if not self.enabled or not os.path.exists(self.path):
            return
        with open(self.path) as f:
            payload = json.load(f)
        entries = payload.get("entries", [])
        self._store = {}
        for item in entries:
            key = self._deserialize_key(item["key"])
            value = self._deserialize_value(item["value"])
            self._store[key] = value

    @staticmethod
    def _serialize_key(key: tuple) -> dict[str, Any]:
        entity_id, bucket = key
        return {
            "entity_id": int(entity_id),
            "bucket": [[zone, int(idx)] for zone, idx in bucket],
        }

    @staticmethod
    def _deserialize_key(payload: dict[str, Any]) -> tuple:
        return (
            int(payload["entity_id"]),
            tuple((str(zone), int(idx)) for zone, idx in payload.get("bucket", [])),
        )

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        # ZoneChoice is a subclass of LocationChoice, so both serialize here.
        if isinstance(value, LocationChoice):
            return {
                "__type__": "LocationChoice",
                "residence": value.residence,
                "workplace": value.workplace,
                "utility": value.utility,
                "zone_utilities": value.zone_utilities,
            }
        return {"__type__": "raw", "value": value}

    @staticmethod
    def _deserialize_value(payload: Any) -> Any:
        if isinstance(payload, dict):
            ptype = payload.get("__type__")
            # New unified type. We reconstruct as ZoneChoice (a subclass
            # of LocationChoice) so that legacy isinstance(x, ZoneChoice)
            # checks continue to pass while the data model carries the
            # full residence/workplace split.
            if ptype == "LocationChoice":
                return ZoneChoice(
                    residence=str(payload["residence"]),
                    workplace=str(payload["workplace"]),
                    utility=float(payload["utility"]),
                    zone_utilities={
                        str(zone): float(value)
                        for zone, value in payload.get("zone_utilities", {}).items()
                    },
                )
            # Legacy payloads from older cache files: reconstruct via ZoneChoice
            # (which sets workplace = zone_name for Singapore semantics).
            if ptype == "ZoneChoice":
                return ZoneChoice(
                    zone_name=payload["zone_name"],
                    utility=float(payload["utility"]),
                    zone_utilities={
                        str(zone): float(value)
                        for zone, value in payload.get("zone_utilities", {}).items()
                    },
                )
            if ptype == "raw":
                return payload.get("value")
        return payload
