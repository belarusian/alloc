"""Tests for alloc.lib.cache — DiskCache and decorators."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from alloc.lib.cache import DiskCache

# =====================================================================
# DiskCache unit tests
# =====================================================================

class TestDiskCache:
    """Tests for the DiskCache class."""

    @pytest.fixture()
    def cache(self, tmp_path: Path) -> DiskCache:
        return DiskCache(cache_dir=tmp_path, enabled=True)

    @pytest.fixture()
    def disabled_cache(self, tmp_path: Path) -> DiskCache:
        return DiskCache(cache_dir=tmp_path, enabled=False)

    # 1. put + get roundtrip
    def test_put_get_roundtrip(self, cache: DiskCache) -> None:
        cache.put("key1", {"a": 1})
        result = cache.get("key1", ttl_seconds=3600)
        assert result == {"a": 1}

    # 2. get miss returns None
    def test_get_miss(self, cache: DiskCache) -> None:
        assert cache.get("no-such-key", ttl_seconds=3600) is None

    # 3. expired entry returns None
    def test_get_expired(self, cache: DiskCache) -> None:
        cache.put("exp", {"b": 2})
        entry_path = cache.cache_dir / "exp.json"
        data = json.loads(entry_path.read_text())
        data["_ts"] = time.time() - 99999
        entry_path.write_text(json.dumps(data))
        assert cache.get("exp", ttl_seconds=10) is None

    # 4. invalidate existing key
    def test_invalidate_existing(self, cache: DiskCache) -> None:
        cache.put("inv", 42)
        assert cache.invalidate("inv") is True
        assert cache.get("inv", 3600) is None

    # 5. invalidate non-existing key
    def test_invalidate_missing(self, cache: DiskCache) -> None:
        assert cache.invalidate("nope") is False

    # 6. clear with prefix
    def test_clear_prefix(self, cache: DiskCache) -> None:
        cache.put("abc1", 1)
        cache.put("abc2", 2)
        cache.put("xyz", 3)
        removed = cache.clear(prefix="abc")
        assert removed == 2
        assert cache.get("abc1", 3600) is None
        assert cache.get("xyz", 3600) is not None

    # 7. clear all
    def test_clear_all(self, cache: DiskCache) -> None:
        cache.put("a", 1)
        cache.put("b", 2)
        removed = cache.clear()
        assert removed == 2

    # 8. disabled cache: put does nothing
    def test_disabled_put(self, disabled_cache: DiskCache) -> None:
        disabled_cache.put("k", "v")
        assert not (disabled_cache.cache_dir / "k.json").exists()

    # 9. disabled cache: get returns None
    def test_disabled_get(self, disabled_cache: DiskCache) -> None:
        assert disabled_cache.get("k", 3600) is None

    # 10. _make_key is deterministic
    def test_make_key_deterministic(self) -> None:
        k1 = DiskCache._make_key("fn", ("a",), (("b", 1),))
        k2 = DiskCache._make_key("fn", ("a",), (("b", 1),))
        assert k1 == k2
        assert len(k1) == 64  # SHA-256 hex

    # 11. _make_key differs for different args
    def test_make_key_different(self) -> None:
        k1 = DiskCache._make_key("fn", ("a",), ())
        k2 = DiskCache._make_key("fn", ("b",), ())
        assert k1 != k2

    # 12. clear on disabled cache returns 0
    def test_disabled_clear(self, disabled_cache: DiskCache) -> None:
        assert disabled_cache.clear() == 0


# =====================================================================
# Helper: build a fake Settings
# =====================================================================

def _fake_settings(tmp_path: Path, cache_enabled: bool = True):
    """Build a FakeSettings instance."""
    _ttl_map = {
        "latest_prices": 900,
        "historical_data": 86400,
        "ticker_details": 604800,
    }

    class FakeSettings:
        def __init__(self) -> None:
            self.cache_dir = tmp_path
            self.cache_enabled = cache_enabled
            self.cache_expiry = _ttl_map

        def get_cache_ttl(self, cache_type: str) -> int:
            return self.cache_expiry[cache_type]

    return FakeSettings()


# =====================================================================
# Decorator tests — patch at source so reload picks up the mock
# =====================================================================

class TestCachedDecorator:
    """Tests for the @cached decorator."""

    # 13. decorator caches result
    def test_decorator_caches(self, tmp_path: Path) -> None:
        fake = _fake_settings(tmp_path)
        import importlib
        import alloc.lib.cache as cache_mod

        # Patch at the source module so reload re-imports the mock
        with patch("alloc.config.settings.get_settings", return_value=fake):
            importlib.reload(cache_mod)

            call_count = 0

            @cache_mod.cached("latest_prices")
            def slow_fn(x: int) -> int:
                nonlocal call_count
                call_count += 1
                return x * 2

            assert slow_fn(5) == 10
            assert slow_fn(5) == 10
            assert call_count == 1

    # 14. decorator respects __cache_valid__: False
    def test_decorator_cache_valid_false(self, tmp_path: Path) -> None:
        fake = _fake_settings(tmp_path)
        import importlib
        import alloc.lib.cache as cache_mod

        with patch("alloc.config.settings.get_settings", return_value=fake):
            importlib.reload(cache_mod)

            call_count = 0

            @cache_mod.cached("latest_prices")
            def bad_fn() -> dict:
                nonlocal call_count
                call_count += 1
                return {"data": 1, "__cache_valid__": False}

            r1 = bad_fn()
            r2 = bad_fn()
            assert r1 == {"data": 1}
            assert r2 == {"data": 1}
            assert "__cache_valid__" not in r1
            assert call_count == 2

    # 15. convenience decorator cache_latest_prices
    def test_cache_latest_prices_convenience(self, tmp_path: Path) -> None:
        fake = _fake_settings(tmp_path)
        import importlib
        import alloc.lib.cache as cache_mod

        with patch("alloc.config.settings.get_settings", return_value=fake):
            importlib.reload(cache_mod)

            call_count = 0

            @cache_mod.cache_latest_prices()
            def prices() -> dict:
                nonlocal call_count
                call_count += 1
                return {"AAPL": 150}

            assert prices() == {"AAPL": 150}
            assert prices() == {"AAPL": 150}
            assert call_count == 1

    # 16. decorator with kwargs
    def test_decorator_kwargs(self, tmp_path: Path) -> None:
        fake = _fake_settings(tmp_path)
        import importlib
        import alloc.lib.cache as cache_mod

        with patch("alloc.config.settings.get_settings", return_value=fake):
            importlib.reload(cache_mod)

            call_count = 0

            @cache_mod.cached("latest_prices")
            def fn(a: int, b: int = 0) -> int:
                nonlocal call_count
                call_count += 1
                return a + b

            assert fn(1, b=2) == 3
            assert fn(1, b=2) == 3
            assert fn(1, b=3) == 4
            assert call_count == 2

    # 17. disabled cache skips caching
    def test_decorator_disabled(self, tmp_path: Path) -> None:
        fake = _fake_settings(tmp_path, cache_enabled=False)
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
            assert fn() == 2
            assert call_count == 2

    # 18. cache expiry via mocked time
    def test_decorator_expiry(self, tmp_path: Path) -> None:
        fake = _fake_settings(tmp_path)
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

            # Expire the cache entry
            for f in tmp_path.glob("*.json"):
                data = json.loads(f.read_text())
                data["_ts"] = time.time() - 9999
                f.write_text(json.dumps(data))

            assert fn() == 2  # re-called after expiry
