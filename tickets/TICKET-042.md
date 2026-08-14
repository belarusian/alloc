# TICKET-042: Apply cache decorators to data pipeline functions per TICKET-007 spec

**Module:** `alloc/models/data.py`
**Priority:** Medium — missing disk caching for market data fetches

## Evidence

TICKET-007 specification requires:
- `get_multi_asset_data()` should be decorated with `@cache_historical` from `alloc.lib.cache`
- `fetch_latest_prices()` should be decorated with `@cache_latest_prices` from `alloc.lib.cache`
- `fetch_latest_prices()` should implement the `__cache_valid__` protocol: if any ticker returns an invalid/zero price, set `result["__cache_valid__"] = False`

Current implementation in `alloc/models/data.py`:
- `get_multi_asset_data()` (line ~140) has **no cache decorator** — every call hits the Polygon API
- `fetch_latest_prices()` (line ~290) has **no cache decorator** — every call hits the API
- Neither function imports from `alloc.lib.cache`
- `fetch_latest_prices()` does not implement `__cache_valid__` protocol

The cache decorators exist and are functional (`alloc/lib/cache.py` lines ~130-170), with `cache_historical` and `cache_latest_prices` exported as convenience decorators.

## Impact

- **Excessive API calls**: Without caching, every workflow run fetches all historical data from Polygon.io, incurring rate limit pressure and slower execution.
- **No cache invalidation on bad data**: Without `__cache_valid__`, stale/zero prices could be cached and served on subsequent runs.
- **Specification non-compliance**: TICKET-007 explicitly requires these decorators; their absence means the ticket is not fully complete.

## Suggestion

1. Add imports at top of `alloc/models/data.py`:
