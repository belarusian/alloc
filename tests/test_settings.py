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
    from alloc.config.settings import get_settings, reset_settings
    reset_settings()
    s = get_settings()
    assert s.polygon_api_key == "env-key"


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


# --- 11. get_settings returns singleton ---
def test_get_settings_singleton(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    from alloc.config.settings import get_settings, reset_settings
    reset_settings()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


# --- 12. get_settings raises when no API key ---
def test_get_settings_no_api_key(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    from alloc.config.settings import get_settings, reset_settings
    reset_settings()
    with pytest.raises(EnvironmentError, match="POLYGON_API_KEY"):
        get_settings()


# --- 13. reset_settings allows re-creation ---
def test_reset_settings(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "key-v1")
    from alloc.config.settings import get_settings, reset_settings
    reset_settings()
    s1 = get_settings()
    assert s1.polygon_api_key == "key-v1"

    monkeypatch.setenv("POLYGON_API_KEY", "key-v2")
    reset_settings()
    s2 = get_settings()
    assert s2.polygon_api_key == "key-v2"
    assert s1 is not s2


# --- 14. No module-level settings variable ---
def test_no_module_level_settings():
    import alloc.config.settings as mod
    assert not hasattr(mod, "settings") or mod.settings is None
