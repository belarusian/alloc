"""Integration tests: cache decorator wired to settings."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch


def _make_fake_settings(tmp_path: Path, **overrides):
    """Build a fake Settings instance for patching get_settings()."""
    defaults = {
        "cache_dir": tmp_path,
        "cache_enabled": True,
        "cache_expiry": {
            "latest_prices": 900,
            "historical_data": 86400,
            "ticker_details": 604800,
        },
    }
    defaults.update(overrides)

    class FakeSettings:
        def __init__(self) -> None:
            self.cache_dir = defaults["cache_dir"]
            self.cache_enabled = defaults["cache_enabled"]
            self.cache_expiry = defaults["cache_expiry"]

        def get_cache_ttl(self, cache_type: str) -> int:
            return self.cache_expiry[cache_type]

    return FakeSettings()


# 1. Cache decorator uses TTL from settings
def test_cache_uses_settings_ttl(tmp_path: Path) -> None:
    fake = _make_fake_settings(tmp_path)
    import importlib
    import alloc.lib.cache as cache_mod

    with patch("alloc.config.settings.get_settings", return_value=fake):
        importlib.reload(cache_mod)

        call_count = 0

        @cache_mod.cached("latest_prices")
        def fn() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        assert fn() == 1
        assert fn() == 1  # cached
        assert call_count == 1

        # Expire the entry
        for f in tmp_path.glob("*.json"):
            data = json.loads(f.read_text())
            data["_ts"] = time.time() - 9999
            f.write_text(json.dumps(data))

        assert fn() == 2  # re-called after expiry


# 2. Cache respects settings.cache_enabled = False
def test_cache_respects_enabled_flag(tmp_path: Path) -> None:
    fake = _make_fake_settings(tmp_path, cache_enabled=False)
    import importlib
    import alloc.lib.cache as cache_mod

    with patch("alloc.config.settings.get_settings", return_value=fake):
        importlib.reload(cache_mod)

        call_count = 0

        @cache_mod.cached("latest_prices")
        def fn() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        assert fn() == 1
        assert fn() == 2  # not cached
        assert call_count == 2


# 3. Cache uses settings.cache_dir
def test_cache_uses_settings_dir(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom_cache"
    fake = _make_fake_settings(custom_dir)
    import importlib
    import alloc.lib.cache as cache_mod

    with patch("alloc.config.settings.get_settings", return_value=fake):
        importlib.reload(cache_mod)

        @cache_mod.cached("latest_prices")
        def fn() -> str:
            return "hello"

        fn()
        # Verify file landed in custom_dir
        files = list(custom_dir.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["value"] == "hello"


# 4. Full roundtrip: write → read → expire → re-read
def test_full_roundtrip(tmp_path: Path) -> None:
    fake = _make_fake_settings(tmp_path)
    import importlib
    import alloc.lib.cache as cache_mod

    with patch("alloc.config.settings.get_settings", return_value=fake):
        importlib.reload(cache_mod)

        call_count = 0

        @cache_mod.cached("historical_data")
        def fn(ticker: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"ticker": ticker, "call": call_count}

        # First call — cached
        r1 = fn("AAPL")
        assert r1 == {"ticker": "AAPL", "call": 1}

        # Second call — from cache (same result)
        r2 = fn("AAPL")
        assert r2 == {"ticker": "AAPL", "call": 1}
        assert call_count == 1

        # Expire
        for f in tmp_path.glob("*.json"):
            data = json.loads(f.read_text())
            data["_ts"] = time.time() - 99999
            f.write_text(json.dumps(data))

        # Third call — re-executed
        r3 = fn("AAPL")
        assert r3 == {"ticker": "AAPL", "call": 2}
        assert call_count == 2
