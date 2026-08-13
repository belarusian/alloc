"""Tests for alloc.lib.client — PolygonClient wrapper."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from alloc.lib.cache import DiskCache
from alloc.lib.client import CACHE_MAP, PolygonClient, _cached_method


# =====================================================================
# Fixtures
# =====================================================================

def _mock_method(name: str, return_value) -> MagicMock:
    """Create a MagicMock with __name__ set so functools.wraps works."""
    m = MagicMock(return_value=return_value)
    m.__name__ = name
    return m


@pytest.fixture()
def mock_rest_client() -> MagicMock:
    """Return a MagicMock standing in for polygon.RESTClient."""
    client = MagicMock()
    client.get_aggs = _mock_method("get_aggs", {"results": [1, 2, 3]})
    client.get_ticker_details = _mock_method("get_ticker_details", {"ticker": "AAPL"})
    client.get_news = _mock_method("get_news", [{"title": "Breaking"}])
    return client


@pytest.fixture()
def cache(tmp_path: Path) -> DiskCache:
    """Real DiskCache backed by a temp directory."""
    return DiskCache(cache_dir=tmp_path, enabled=True)


@pytest.fixture()
def disabled_cache(tmp_path: Path) -> DiskCache:
    """DiskCache with caching disabled."""
    return DiskCache(cache_dir=tmp_path, enabled=False)


@pytest.fixture()
def client(mock_rest_client: MagicMock, cache: DiskCache) -> PolygonClient:
    """PolygonClient with mocked RESTClient and real cache."""
    with patch("alloc.lib.client.RESTClient", return_value=mock_rest_client):
        return PolygonClient(api_key="fake-key", cache=cache)


# =====================================================================
# Construction
# =====================================================================

class TestConstruction:
    """Tests for PolygonClient.__init__."""

    def test_creates_underlying_client(self, mock_rest_client: MagicMock) -> None:
        with patch("alloc.lib.client.RESTClient") as mock_cls:
            mock_cls.return_value = mock_rest_client
            PolygonClient(api_key="k", cache=MagicMock(spec=DiskCache))
            mock_cls.assert_called_once_with("k")

    def test_stores_cache_reference(
        self, mock_rest_client: MagicMock, cache: DiskCache
    ) -> None:
        with patch("alloc.lib.client.RESTClient", return_value=mock_rest_client):
            c = PolygonClient(api_key="k", cache=cache)
            assert c._cache is cache

    def test_uses_default_cache_map(
        self, mock_rest_client: MagicMock, cache: DiskCache
    ) -> None:
        with patch("alloc.lib.client.RESTClient", return_value=mock_rest_client):
            c = PolygonClient(api_key="k", cache=cache)
            assert c._cache_map == CACHE_MAP

    def test_accepts_custom_cache_map(
        self, mock_rest_client: MagicMock, cache: DiskCache
    ) -> None:
        custom = {"get_aggs": "latest_prices"}
        with patch("alloc.lib.client.RESTClient", return_value=mock_rest_client):
            c = PolygonClient(api_key="k", cache=cache, cache_map=custom)
            assert c._cache_map == custom


# =====================================================================
# Caching behaviour
# =====================================================================

class TestCaching:
    """Tests that cached methods actually cache."""

    def test_get_aggs_is_cached(self, client: PolygonClient) -> None:
        """Second call to get_aggs should hit cache, not upstream."""
        client.get_aggs("AAPL", 1, "day", "2024-01-01", "2024-01-31")
        client.get_aggs("AAPL", 1, "day", "2024-01-01", "2024-01-31")
        # Underlying client's get_aggs called only once
        assert client._client.get_aggs.call_count == 1

    def test_get_ticker_details_is_cached(self, client: PolygonClient) -> None:
        client.get_ticker_details("AAPL")
        client.get_ticker_details("AAPL")
        assert client._client.get_ticker_details.call_count == 1

    def test_different_args_miss_cache(self, client: PolygonClient) -> None:
        client.get_aggs("AAPL", 1, "day", "2024-01-01", "2024-01-31")
        client.get_aggs("MSFT", 1, "day", "2024-01-01", "2024-01-31")
        assert client._client.get_aggs.call_count == 2

    def test_disabled_cache_skips_caching(
        self, mock_rest_client: MagicMock, disabled_cache: DiskCache
    ) -> None:
        with patch("alloc.lib.client.RESTClient", return_value=mock_rest_client):
            c = PolygonClient(api_key="k", cache=disabled_cache)
            c.get_aggs("AAPL", 1, "day", "2024-01-01", "2024-01-31")
            c.get_aggs("AAPL", 1, "day", "2024-01-01", "2024-01-31")
            assert mock_rest_client.get_aggs.call_count == 2


# =====================================================================
# Proxy behaviour
# =====================================================================

class TestProxy:
    """Tests that uncached methods are proxied to RESTClient."""

    def test_uncached_method_is_proxied(self, client: PolygonClient) -> None:
        """get_news is not in CACHE_MAP, should go straight through."""
        result = client.get_news("AAPL")
        assert result == [{"title": "Breaking"}]
        assert client._client.get_news.call_count == 1

    def test_uncached_method_called_every_time(self, client: PolygonClient) -> None:
        client.get_news("AAPL")
        client.get_news("AAPL")
        assert client._client.get_news.call_count == 2

    def test_unknown_attribute_raises(self, cache: DiskCache) -> None:
        """__getattr__ should delegate; missing attrs raise AttributeError.

        We use a SimpleNamespace so that missing attributes actually raise
        AttributeError (unlike MagicMock which returns a new mock).
        """
        def _noop(*a, **k):
            return None

        bare_client = SimpleNamespace()
        _noop.__name__ = "get_aggs"
        bare_client.get_aggs = _noop
        _noop2 = _noop
        _noop2.__name__ = "get_ticker_details"
        bare_client.get_ticker_details = _noop2

        with patch("alloc.lib.client.RESTClient", return_value=bare_client):
            c = PolygonClient(api_key="k", cache=cache)
            with pytest.raises(AttributeError):
                _ = c.this_method_does_not_exist_at_all


# =====================================================================
# __cache_valid__ protocol
# =====================================================================

class TestCacheValidProtocol:
    """Tests for the __cache_valid__ sentinel in client wrapper."""

    def test_cache_valid_false_not_cached(
        self, mock_rest_client: MagicMock, cache: DiskCache
    ) -> None:
        """A result with __cache_valid__: False should not be cached."""
        mock_rest_client.get_ticker_details.return_value = {
            "ticker": "AAPL",
            "__cache_valid__": False,
        }
        with patch("alloc.lib.client.RESTClient", return_value=mock_rest_client):
            c = PolygonClient(api_key="k", cache=cache)
            r1 = c.get_ticker_details("AAPL")
            r2 = c.get_ticker_details("AAPL")
            # Called twice — never cached
            assert mock_rest_client.get_ticker_details.call_count == 2
            # Sentinel stripped from return value
            assert "__cache_valid__" not in r1
            assert r1 == {"ticker": "AAPL"}


# =====================================================================
# _apply_caching edge cases
# =====================================================================

class TestApplyCaching:
    """Tests for _apply_caching internal method."""

    def test_missing_method_on_restclient_is_skipped(
        self, mock_rest_client: MagicMock, cache: DiskCache
    ) -> None:
        """If a method in cache_map doesn't exist on RESTClient, skip it."""
        custom_map = {
            "get_aggs": "historical_data",
            "nonexistent_method_xyz": "latest_prices",
        }
        with patch("alloc.lib.client.RESTClient", return_value=mock_rest_client):
            c = PolygonClient(api_key="k", cache=cache, cache_map=custom_map)
            # get_aggs should be wrapped
            assert hasattr(c, "get_aggs")
            # nonexistent_method_xyz should not cause an error
            # and should fall through to __getattr__ (which will raise)


# =====================================================================
# CACHE_MAP module constant
# =====================================================================

class TestCacheMap:
    """Tests for the CACHE_MAP module constant."""

    def test_cache_map_has_expected_keys(self) -> None:
        assert "get_aggs" in CACHE_MAP
        assert "get_ticker_details" in CACHE_MAP

    def test_cache_map_values_are_known_types(self) -> None:
        for v in CACHE_MAP.values():
            assert v in ("historical_data", "ticker_details", "latest_prices")
