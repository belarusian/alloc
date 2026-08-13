# TICKET-004: Integration — Cache + Settings Wiring

**Scope:** Verify `alloc/lib/cache.py` correctly imports and uses `alloc/config/settings.py`.
**Test:** `tests/test_cache_settings_integration.py`
**Priority:** Medium — validates the dependency chain works end-to-end.

## What to Implement

A small integration test module that verifies the cache decorator respects the settings module's configuration.

### Test Cases

| Test | Verifies |
|---|---|
| `test_cache_uses_settings_ttl` | `@cached("latest_prices")` uses 900s TTL from settings |
| `test_cache_respects_enabled_flag` | When `settings.cache_enabled` is `False`, decorator bypasses cache |
| `test_cache_dir_from_settings` | Cache writes to `settings.cache_dir` |
| `test_full_decorator_roundtrip` | End-to-end: decorate a function, call twice, verify single execution on second call |

## Dependencies

- `TICKET-001`, `TICKET-002`, `TICKET-003` — all must be implemented first

## Implementation Notes

This is a test-only ticket. No source code changes. The validator should create the test file and confirm it passes against the implemented modules.
