# TICKET-003: alloc/lib/cache.py — File-Based Pickle Cache

**Module:** `alloc/lib/cache.py`
**Test:** `tests/test_cache.py`
**Priority:** High — required by all data-fetching modules.

## What to Implement

A disk-backed cache using pickle serialization, MD5-hash keys derived from function arguments, and per-type TTL expiration. Exposed as a decorator factory for transparent caching.

### Class: `DiskCache`

| Method | Signature | Purpose |
|---|---|---|
| `__init__` | `(cache_dir: Path, enabled: bool = True)` | Initialize with config |
| `_make_key` | `(func_name: str, args, kwargs) -> str` | MD5 hash of `func_name + sorted args/kwargs` |
| `_cache_path` | `(key: str) -> Path` | Return `cache_dir / f"{key}.pkl"` |
| `get` | `(key: str, ttl_seconds: int) -> Any \| None` | Load if exists and not expired |
| `put` | `(key: str, value: Any) -> None` | Serialize and write to disk |
| `invalidate` | `(key: str) -> bool` | Delete a single cache entry |
| `clear` | `(prefix: str \| None = None) -> int` | Delete matching entries, return count |

### Decorator Factory: `cached(cache_type: str = "historical_data")`

Returns a decorator that wraps a function with cache-aside logic:

1. If caching disabled, call through
2. Compute key from `func.__name__`, `args`, `kwargs`
3. Look up TTL from `settings.get_cache_ttl(cache_type)`
4. If cache hit and not expired, return cached value
5. On miss, call the function, cache the result, return it

### `__cache_valid__` Protocol

If a cached function returns a dict with `"__cache_valid__": False`, skip writing the cache but still return the cleaned result (without the metadata key) to the caller. This allows a function to signal "I ran but the result shouldn't be cached."

### Convenience Decorators

```python
cache_latest_prices = functools.partial(cached, cache_type="latest_prices")
cache_historical = functools.partial(cached, cache_type="historical_data")
cache_ticker_info = functools.partial(cached, cache_type="ticker_details")
```

## Tests

**File:** `tests/test_cache.py`

| Test | Verifies |
|---|---|
| `test_key_deterministic` | Same args produce same MD5 key |
| `test_key_unique_args` | Different args produce different keys |
| `test_key_unique_kwargs` | Different kwargs produce different keys |
| `test_key_kwargs_order_invariant` | `f(a=1, b=2)` and `f(b=2, a=1)` produce same key |
| `test_put_get_roundtrip` | Value written can be read back |
| `test_get_expired` | Entry older than TTL returns `None` |
| `test_get_missing` | Nonexistent key returns `None` |
| `test_invalidate_existing` | Deletes file, returns `True` |
| `test_invalidate_missing` | No error, returns `False` |
| `test_clear_all` | Removes all cache files |
| `test_clear_prefix` | Only removes files containing prefix |
| `test_decorator_hit` | Second call returns cached value without invoking function |
| `test_decorator_miss` | First call invokes function and caches result |
| `test_decorator_disabled` | Bypasses cache when `enabled=False` |
| `test_decorator_ttl_expiry` | Cached value expires after TTL |
| `test_cache_valid_false` | `__cache_valid__: False` skips write, returns cleaned dict |
| `test_cache_valid_true` | `__cache_valid__: True` writes normally |
| `test_convenience_decorators` | `cache_latest_prices` etc. apply correct TTL |
| `test_cache_dir_created` | Missing cache directory is created on first write |

Use `tmp_path` fixture for isolated cache directories. Mock `time.time()` to control expiration.

## Dependencies

- `TICKET-001` (package inits) — must be able to import `alloc.config.settings`
- `TICKET-002` (settings) — needs `settings.get_cache_ttl()` and `settings.cache_enabled`

## Improvements Over Seed

1. **Class-based over procedural.** The seed uses module-level functions (`ensure_cache_dir`, `get_cache_key`, `is_cache_valid`) with global imports. The alloc version encapsulates state in a `DiskCache` class, making it testable and composable.
2. **Explicit TTL parameter.** The seed hardcodes TTL lookup inside `is_cache_valid`. The alloc version passes TTL explicitly through `get()`, decoupling expiration policy from storage.
3. **`invalidate` and `clear` methods.** The seed only has `clear_cache(cache_type)` which does string matching on filenames. The alloc version has precise key-based invalidation and prefix-based clearing.
4. **No print statements.** The seed uses `print()` for cache hit/miss logging. The alloc version uses Python's `logging` module (or no output by default, letting callers log).
