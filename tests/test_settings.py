"""Tests for alloc.config.settings."""

from __future__ import annotations

from pathlib import Path

import pytest


def _import_fresh():
    """Re-import the module so env vars are re-read."""
    import importlib

    import alloc.config.settings as mod
    importlib.reload(mod)
    return mod


# --- 1. Settings class instantiates with explicit args ---
def test_settings_explicit():
    from alloc.config.settings import Settings
    s = Settings(polygon_api_key="abc123")
    assert s.polygon_api_key == "abc123"
    assert s.cache_enabled is True
    assert s.cache_dir == Path("./cache")


# --- 2. Missing API key raises EnvironmentError ---
def test_settings_missing_api_key(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    from alloc.config.settings import Settings
    with pytest.raises(EnvironmentError, match="POLYGON_API_KEY"):
        Settings()


# --- 3. API key from env var ---
def test_settings_api_key_from_env(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "env-key")
    mod = _import_fresh()
    assert mod.settings.polygon_api_key == "env-key"


# --- 4. cache_enabled default True ---
def test_settings_cache_enabled_default():
    from alloc.config.settings import Settings
    s = Settings(polygon_api_key="k")
    assert s.cache_enabled is True


# --- 5. cache_enabled parsed from env (true) ---
def test_settings_cache_enabled_env_true(monkeypatch):
    monkeypatch.setenv("CACHE_ENABLED", "true")
    from alloc.config.settings import Settings
    s = Settings(polygon_api_key="k")
    assert s.cache_enabled is True


# --- 6. cache_enabled parsed from env (false) ---
def test_settings_cache_enabled_env_false(monkeypatch):
    monkeypatch.setenv("CACHE_ENABLED", "false")
    from alloc.config.settings import Settings
    s = Settings(polygon_api_key="k")
    assert s.cache_enabled is False


# --- 7. cache_enabled parsed from env (1) ---
def test_settings_cache_enabled_env_1(monkeypatch):
    monkeypatch.setenv("CACHE_ENABLED", "1")
    from alloc.config.settings import Settings
    s = Settings(polygon_api_key="k")
    assert s.cache_enabled is True


# --- 8. cache_enabled parsed from env (yes) ---
def test_settings_cache_enabled_env_yes(monkeypatch):
    monkeypatch.setenv("CACHE_ENABLED", "yes")
    from alloc.config.settings import Settings
    s = Settings(polygon_api_key="k")
    assert s.cache_enabled is True


# --- 9. get_cache_ttl returns correct values ---
def test_get_cache_ttl():
    from alloc.config.settings import Settings
    s = Settings(polygon_api_key="k")
    assert s.get_cache_ttl("latest_prices") == 900
    assert s.get_cache_ttl("historical_data") == 86400
    assert s.get_cache_ttl("ticker_details") == 604800


# --- 10. get_cache_ttl raises KeyError for unknown type ---
def test_get_cache_ttl_unknown():
    from alloc.config.settings import Settings
    s = Settings(polygon_api_key="k")
    with pytest.raises(KeyError):
        s.get_cache_ttl("nonexistent")
