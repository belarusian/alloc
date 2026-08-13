"""Configuration management for alloc.

Reads environment variables and provides typed settings with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path


def _parse_bool(value: str | None, default: bool = True) -> bool:
    """Parse a boolean from an environment variable string.

    Accepts: true/1/t/yes (case-insensitive) → True, everything else → default.
    """
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "t", "yes")


class Settings:
    """Centralised application settings.

    Parameters
    ----------
    polygon_api_key : str
        Required Polygon.io API key.
    cache_enabled : bool
        Whether disk caching is active.
    cache_dir : Path
        Directory used for cached data.
    cache_expiry : dict[str, int]
        TTL in seconds per cache type.
    """

    def __init__(
        self,
        polygon_api_key: str | None = None,
        cache_enabled: bool = True,
        cache_dir: Path | str | None = None,
        cache_expiry: dict[str, int] | None = None,
    ) -> None:
        # --- API key (required) ---
        self.polygon_api_key: str = (
            polygon_api_key
            or os.environ.get("POLYGON_API_KEY", "")
        )
        if not self.polygon_api_key:
            raise EnvironmentError(
                "POLYGON_API_KEY environment variable is required but not set."
            )

        # --- Cache enabled ---
        raw_enabled = os.environ.get("CACHE_ENABLED")
        self.cache_enabled: bool = _parse_bool(raw_enabled, cache_enabled)

        # --- Cache directory ---
        raw_dir = os.environ.get("CACHE_DIR")
        self.cache_dir: Path = Path(
            cache_dir if cache_dir is not None else (raw_dir or "./cache")
        )

        # --- Cache expiry (TTL in seconds) ---
        self.cache_expiry: dict[str, int] = cache_expiry or {
            "latest_prices": 900,       # 15 minutes
            "historical_data": 86400,   # 24 hours
            "ticker_details": 604800,   # 7 days
        }

    def get_cache_ttl(self, cache_type: str) -> int:
        """Return TTL in seconds for *cache_type*.

        Raises ``KeyError`` if the type is unknown.
        """
        return self.cache_expiry[cache_type]


# Module-level singleton — created on first import.
settings = Settings()
