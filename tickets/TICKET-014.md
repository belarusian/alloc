# TICKET-014: Build Safety — Settings Lazy Initialization

**Module:** `alloc/config/settings.py`
**Test:** `tests/test_settings.py`
**Priority:** High — build fails in CI if POLYGON_API_KEY is not set.

## Problem

The `Settings` singleton is instantiated at module import time:
```python
settings = Settings()
```

This raises `EnvironmentError` if `POLYGON_API_KEY` is not set, which happens in CI.

Current workaround: set dummy env var in GitHub Actions. Better: make Settings lazy-loaded.

## What to Implement

1. Remove module-level `settings = Settings()` instantiation
2. Create `get_settings()` function that lazily loads settings on first call
3. Keep backwards compatibility: allow `from alloc.config.settings import settings` to work via lazy proxy
4. Update all imports to use `get_settings()` instead of direct import

### Changes

**alloc/config/settings.py:**
- Remove module-level `settings = Settings()`
- Add `get_settings()` function with lazy singleton pattern
- Optionally add `_settings_instance` module variable
- Update docstrings

**Code locations to update:**
- `alloc/core.py` — uses Settings() directly, update to get_settings()
- Any other module that imports `settings` from alloc.config

### Tests

- `test_settings_lazy_load`: verify Settings is not instantiated until get_settings() called
- `test_settings_singleton`: verify get_settings() returns same instance
- `test_settings_missing_api_key`: verify EnvironmentError still raised on first use

### Dependencies

- None — changes settings module only

### Design Improvements

This eliminates environment-dependent initialization at import time, making the module safe to import in CI/testing environments without credentials.

---

## After This Ticket

Build will pass in CI without requiring POLYGON_API_KEY env var to be set at import time. The env var will only be required when settings are actually used (first access).
