"""Polygon API client wrapper with caching.

Thin wrapper around ``polygon.StocksClient`` that applies disk caching
to select methods while proxying all other attributes through
``__getattr__`` to the underlying client.

Key design decisions vs. a bare client:

1. **Dependency injection** — the cache instance is injected at
   construction time rather than created inside decorators.
2. **Logging** — all diagnostics go through ``logging``, never ``print``.
3. **Configurable cache types** — each wrapped method declares its
   cache type string, which is looked up in settings for TTL.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from polygon import StocksClient

from alloc.lib.cache import DiskCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache-type registry
# ---------------------------------------------------------------------------

# Maps method names to the cache-type string used by settings.get_cache_ttl().
# Add entries here to enable caching for additional Polygon methods.
CACHE_MAP: dict[str, str] = {
    "get_aggs": "historical_data",
    "get_ticker_details": "ticker_details",
}


# ---------------------------------------------------------------------------
# Decorator factory (uses injected cache)
# ---------------------------------------------------------------------------

def _cached_method(cache: DiskCache, cache_type: str) -> Callable:
    """Return a decorator that caches a method's result via *cache*.

    Parameters
    ----------
    cache : DiskCache
        The shared cache instance.
    cache_type : str
        Key for TTL lookup in settings.

    Returns
    -------
    Callable
        A decorator that wraps the target method.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = cache._make_key(func.__name__, args, tuple(sorted(kwargs.items())))
            ttl = cache.get_ttl(cache_type)

            cached_value = cache.get(key, ttl)
            if cached_value is not None:
                logger.debug("Cache hit for %s", func.__name__)
                return cached_value

            logger.debug("Cache miss for %s — calling upstream", func.__name__)
            result = func(*args, **kwargs)

            # __cache_valid__ protocol
            if isinstance(result, dict) and result.get("__cache_valid__") is False:
                cleaned = {k: v for k, v in result.items() if k != "__cache_valid__"}
                return cleaned

            cache.put(key, result)
            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class PolygonClient:
    """Cached wrapper around ``polygon.StocksClient``.

    Parameters
    ----------
    api_key : str
        Polygon.io API key.
    cache : DiskCache
        Shared cache instance (dependency-injected).
    cache_map : dict[str, str] | None
        Optional override of the default method-to-cache-type mapping.

    Examples
    --------
    >>> from alloc.lib.cache import DiskCache
    >>> cache = DiskCache("./cache", enabled=True)
    >>> client = PolygonClient(api_key="abc123", cache=cache)
    >>> bars = client.get_aggs("AAPL", 1, "day", "2024-01-01", "2024-01-31")
    """

    def __init__(
        self,
        api_key: str,
        cache: DiskCache,
        cache_map: dict[str, str] | None = None,
    ) -> None:
        self._client = StocksClient(api_key)
        self._cache = cache
        self._cache_map = cache_map or CACHE_MAP

        # Apply caching decorators to registered methods
        self._apply_caching()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_caching(self) -> None:
        """Wrap methods listed in ``_cache_map`` with caching decorators."""
        for method_name, cache_type in self._cache_map.items():
            raw = getattr(self._client, method_name, None)
            if raw is None:
                logger.warning(
                    "Method %r not found on StocksClient — skipping cache wrap",
                    method_name,
                )
                continue
            wrapped = _cached_method(self._cache, cache_type)(raw)
            setattr(self, method_name, wrapped)
            logger.info("Cached method %s (type=%s)", method_name, cache_type)

    # ------------------------------------------------------------------
    # Proxy
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying StocksClient.

        This allows uncached methods and properties to be called
        transparently.
        """
        return getattr(self._client, name)
