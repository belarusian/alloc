"""Disk-backed cache for alloc data fetches.

Provides a :class:`DiskCache` class and decorator helpers that automatically
read TTL values from :data:`alloc.config.settings.settings`.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from alloc.config.settings import get_settings

logger = logging.getLogger(__name__)


class DiskCache:
    """Simple JSON-file disk cache with TTL support.

    Each cache entry is stored as a single JSON file containing the value,
    timestamp, and TTL.  Expired entries are treated as cache misses.
    """

    def __init__(self, cache_dir: Path | str, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)


    def get_ttl(self, cache_type: str) -> int:
        """Return TTL in seconds for *cache_type* via settings."""
        return get_settings().get_cache_ttl(cache_type)

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(func_name: str, args: tuple, kwargs: tuple) -> str:
        """Create a deterministic cache key from call signature."""
        raw = f"{func_name}|{args}|{kwargs}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get(self, key: str, ttl_seconds: int) -> Any | None:
        """Return cached value if it exists and has not expired.

        Returns ``None`` on miss or expiry.
        """
        if not self.enabled:
            return None

        entry_path = self.cache_dir / f"{key}.json"
        if not entry_path.exists():
            return None

        try:
            data = json.loads(entry_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read cache entry %s: %s", key, exc)
            return None

        age = time.time() - data["_ts"]
        if age > ttl_seconds:
            logger.debug(
                "Cache miss (expired) for key %s (age %.0fs > ttl %ds)",
                key, age, ttl_seconds,
            )
            return None

        logger.debug("Cache hit for key %s", key)
        return data["value"]

    def put(self, key: str, value: Any) -> None:
        """Write *value* to the cache."""
        if not self.enabled:
            return
        entry = {"value": value, "_ts": time.time()}
        entry_path = self.cache_dir / f"{key}.json"
        try:
            entry_path.write_text(json.dumps(entry))
        except OSError as exc:
            logger.warning("Failed to write cache entry %s: %s", key, exc)

    def invalidate(self, key: str) -> bool:
        """Remove a single cache entry.  Returns ``True`` if it existed."""
        entry_path = self.cache_dir / f"{key}.json"
        if entry_path.exists():
            entry_path.unlink()
            return True
        return False

    def clear(self, prefix: str = "") -> int:
        """Remove all entries whose key starts with *prefix*.

        Returns the number of files removed.
        """
        if not self.enabled:
            return 0
        count = 0
        for entry_path in self.cache_dir.glob("*.json"):
            if entry_path.stem.startswith(prefix):
                entry_path.unlink()
                count += 1
        return count


# ----------------------------------------------------------------------
# Decorator factory
# ----------------------------------------------------------------------

def cached(cache_type: str) -> Callable:
    """Decorator that caches a function's return value on disk.

    Reads TTL from :func:`get_settings().get_cache_ttl` for *cache_type*.

    **__cache_valid__ protocol**
    If the decorated function returns a ``dict`` containing
    ``{"__cache_valid__": False}``, the result is **not** written to
    cache and the ``__cache_valid__`` key is stripped from the returned
    dict before passing it back to the caller.
    """

    def decorator(func: Callable) -> Callable:
        cache = DiskCache(
            cache_dir=get_settings().cache_dir,
            enabled=get_settings().cache_enabled,
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = cache._make_key(func.__name__, args, tuple(sorted(kwargs.items())))
            ttl = get_settings().get_cache_ttl(cache_type)

            # Try cache first
            cached_value = cache.get(key, ttl)
            if cached_value is not None:
                return cached_value

            # Call the real function
            result = func(*args, **kwargs)

            # __cache_valid__ protocol
            if isinstance(result, dict) and result.get("__cache_valid__") is False:
                # Strip the sentinel and return cleaned dict — do NOT cache
                cleaned = {k: v for k, v in result.items() if k != "__cache_valid__"}
                return cleaned

            # Cache the result
            cache.put(key, result)
            return result

        return wrapper

    return decorator


# ----------------------------------------------------------------------
# Convenience decorators (pre-wired to settings TTLs)
# ----------------------------------------------------------------------

cache_latest_prices = functools.partial(cached, "latest_prices")
cache_historical = functools.partial(cached, "historical_data")
cache_ticker_info = functools.partial(cached, "ticker_details")
