"""Shared pytest fixtures for the alloc test suite.

The disk cache is disabled for the entire suite so that tests are hermetic
and do not read stale entries from a shared ``./cache`` directory.  The cache
key incorporates the identity of injected mock clients, which is unstable
across garbage collection, so an enabled cache would cause cross-test
pollution (a test reading a price cached by an unrelated test).

Tests that specifically exercise caching behaviour (``test_cache.py`` and
``test_cache_settings_integration.py``) patch ``get_settings`` with their own
fake and are unaffected by this fixture.
"""

from __future__ import annotations

import pytest

import alloc.config.settings as _settings_mod
import alloc.lib.cache as _cache_mod

# Capture the real get_settings once at import time so it can be restored
# even after tests that reload alloc.lib.cache rebind its reference to a fake.
_real_get_settings = _settings_mod.get_settings


@pytest.fixture(autouse=True)
def _isolate_disk_cache():
    """Disable the disk cache and pin ``cache.get_settings`` to the real one.

    Some tests reload ``alloc.lib.cache`` inside a ``patch`` block, which
    rebinds ``cache.get_settings`` to a fake and leaks it into later tests.
    This fixture forces the real reference back before every test and disables
    caching on the settings singleton for the duration of the test, so the
    decorated data-pipeline functions never touch the shared cache directory.
    """
    # Pin the cache module's get_settings to the real function (undo any leak).
    _cache_mod.get_settings = _real_get_settings

    settings = _real_get_settings()
    original_enabled = settings.cache_enabled
    settings.cache_enabled = False
    try:
        yield
    finally:
        settings.cache_enabled = original_enabled
