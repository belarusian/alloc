# TICKET-002: alloc/config/settings.py — Configuration Loading

**Module:** `alloc/config/settings.py`
**Test:** `tests/test_settings.py`
**Priority:** High — required by cache and all API modules.

## What to Implement

A configuration module that loads environment variables, validates required settings, and exposes typed constants for the rest of the application.

### Class: `Settings`

A singleton-style configuration class with these attributes:

| Attribute | Type | Default | Source |
|---|---|---|---|
| `polygon_api_key` | `str` | *(required)* | `POLYGON_API_KEY` env var |
| `cache_enabled` | `bool` | `True` | `CACHE_ENABLED` env var |
| `cache_dir` | `Path` | `Path("./cache")` | `CACHE_DIR` env var |
| `cache_expiry` | `dict[str, int]` | see below | hardcoded defaults |

Default `cache_expiry` values (seconds):
- `latest_prices`: 900 (15 minutes)
- `historical_data`: 86400 (24 hours)
- `ticker_details`: 604800 (7 days)

### Behavior

1. On import, load `.env` file from project root using `python-dotenv`
2. Raise `EnvironmentError` with a clear message if `POLYGON_API_KEY` is missing
3. Parse boolean env vars robustly (`true`, `1`, `t`, `yes` all map to `True`)
4. Resolve `CACHE_DIR` relative to project root, not the module file location

### Module-level Instance

Expose a pre-instantiated `settings` object at module level so consumers do `from alloc.config import settings`.

### Function: `get_cache_ttl(cache_type: str) -> int`

Convenience function to look up TTL for a cache type, falling back to 24 hours for unknown types.

## Tests

**File:** `tests/test_settings.py`

| Test | Verifies |
|---|---|
| `test_api_key_required` | Missing `POLYGON_API_KEY` raises `EnvironmentError` |
| `test_api_key_loaded` | Valid key is accessible as a non-empty string |
| `test_cache_enabled_default` | Missing `CACHE_ENABLED` defaults to `True` |
| `test_cache_enabled_false` | `CACHE_ENABLED=0` parses to `False` |
| `test_cache_dir_default` | Missing `CACHE_DIR` defaults to `Path("./cache")` |
| `test_cache_dir_custom` | `CACHE_DIR=/tmp/test` resolves correctly |
| `test_cache_expiry_defaults` | All three cache types have correct default TTLs |
| `test_get_cache_ttl_known` | Known type returns correct TTL |
| `test_get_cache_ttl_unknown` | Unknown type returns 24-hour default |
| `test_settings_singleton` | Module-level `settings` is a `Settings` instance |

Use `monkeypatch` to set/unset env vars. Do not depend on the real `.env` file.

## Dependencies

None. This is a leaf module with no internal dependencies.

## Improvements Over Seed

1. **Class-based over module-level constants.** The seed exposes bare constants (`POLYGON_API_KEY`, `CACHE_DIR`, etc.) at module level. The alloc version encapsulates them in a `Settings` class, enabling testing, mocking, and multiple configurations.
2. **Path-based cache dir.** The seed builds `CACHE_DIR` relative to `__file__`, which breaks in packaged installations. The alloc version uses a project-root-relative path.
3. **No rate limit settings.** The seed includes `RATE_LIMIT_SLEEP` which is set to 0 (disabled). The alloc version omits this — rate limiting belongs in the API client, not global config.
