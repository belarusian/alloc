# TICKET-007: alloc/models/data.py — Multi-frequency Market Data Pipeline

**Module:** alloc/models/data.py
**Test:** tests/test_data.py
**Priority:** High — RL agent cannot train or predict without market state vectors.

## What to Implement

A data pipeline module that fetches multi-frequency price data from Polygon.io, caches it via alloc.lib.cache, and constructs fixed-dimension state vectors for the RL agent. Read ~/Research/new-trader/trader/models/data.py for understanding — never copy verbatim.

### Function 1: get_multi_asset_data()

```python
def get_multi_asset_data(
    tickers: list[str],
    client: PolygonClient,
    end_date: datetime | None = None,
    hourly_days: int = 7,
    daily_days: int = 365,
    weekly_weeks: int = 52,
) -> dict[str, dict[str, list[float]]]:
```

**Behavior:**
1. Accept a list of ticker symbols and a PolygonClient instance (dependency-injected, not a module singleton).
2. For each ticker, fetch three frequency bands via client.get_aggs():
   - **Hourly**: last hourly_days days (default 7), multiplier=1, timespan="hour"
   - **Daily**: last daily_days days (default 365), multiplier=1, timespan="day"
   - **Weekly**: last weekly_weeks weeks (default 52), multiplier=1, timespan="week"
3. Extract .close prices from each bar, ensuring scalar float values (handle ensure_scalar_price-style normalization).
4. Return a nested dict: {ticker: {"hourly": [float, ...], "daily": [float, ...], "weekly": [float, ...]}}.
5. Decorate with @cache_historical from alloc.lib.cache for disk caching.
6. Use logging for diagnostics — never print().

**Key differences from seed (new-trader/trader/models/data.py):**
- **Dependency injection**: Accept client as parameter instead of using a module-level singleton. The seed creates polygon_client = CachedPolygonClient(POLYGON_API_KEY) at module level (line 20).
- **No rate-limit sleep**: The seed inserts time.sleep(RATE_LIMIT_SLEEP) between API calls (lines 82, 100, 118). The alloc PolygonClient handles caching at the method level; rate limiting belongs in the client layer, not the data layer.
- **No threading lock**: The seed uses a rate_limit_lock threading lock (line 22). Alloc design is single-threaded for data fetching.
- **Logging over print**: The seed uses print() throughout (lines 67, 69, 80, etc.). Alloc uses logging.getLogger(__name__).

### Function 2: build_state_vector()

```python
def build_state_vector(
    multi_freq_data: dict[str, dict[str, list[float]]],
    current_allocation: dict[str, float],
    tickers: list[str],
    n_hourly: int = 168,
    n_daily: int = 365,
    n_weekly: int = 52,
) -> np.ndarray:
```

**Behavior:**
1. For each ticker in tickers (ordered):
   - Take the **last n_hourly** hourly prices from multi_freq_data[ticker]["hourly"].
   - Take the **last n_daily** daily prices from multi_freq_data[ticker]["daily"].
   - Take the **last n_weekly** weekly prices from multi_freq_data[ticker]["weekly"].
   - **Normalize** each window by dividing by the most-recent price in that window (the last element). If the most-recent price is 0, use 1.0 to avoid division by zero.
   - **Pad** with 1.0 if the window has fewer elements than requested (insufficient history).
   - Concatenate the three normalized windows into one per-ticker vector of length n_hourly + n_daily + n_weekly.
2. Concatenate all per-ticker vectors into one flat array.
3. Append the **current allocation vector** (values from current_allocation for each ticker, in ticker order).
4. Return a single np.ndarray of shape (n_tickers * (n_hourly + n_daily + n_weekly) + n_tickers,).

**Determinism contract:** The output dimension is fixed regardless of how much history is available. Padding ensures the RL agent always receives the same input shape.

**Key differences from seed (new-trader/trader/models/data.py lines 150-210):**
- The seed build_state_vector() uses a module-level polygon_client to fetch data inline. The alloc version receives pre-fetched multi_freq_data as a parameter — separation of concerns.
- The seed normalizes by the **latest price across all frequencies** (line 178: latest_price = data[ticker]["hourly"][-1]). The alloc version normalizes **per-frequency window** by its own most-recent price, which is more robust when frequencies have different recency.
- The seed pads with 0.0 (line 185). The alloc version pads with 1.0 (normalized "no change" signal), which is more meaningful for the RL agent — a price ratio of 1.0 means "flat," whereas 0.0 means "crashed to zero."

### Function 3: get_latest_prices()

```python
def get_latest_prices(
    tickers: list[str],
    client: PolygonClient,
) -> dict[str, float]:
```

**Behavior:**
1. For each ticker, call client.get_last_trade(ticker.upper()).
2. Extract .price as a float. If no trade or no price attribute, return 0.0.
3. Return {ticker: float, ...}.
4. Decorate with @cache_latest_prices from alloc.lib.cache.
5. Implement the __cache_valid__ protocol: if any ticker returns an invalid/zero price, set result["__cache_valid__"] = False so the cache layer skips persistence.

**Key differences from seed (new-trader/trader/models/data.py lines 280-320):**
- **Dependency injection**: Accept client as parameter. The seed uses the module-level polygon_client singleton.
- **Logging over print**: The seed prints timestamps and prices (lines 285-295). Alloc logs at DEBUG level.
- **No rate-limit sleep**: Same rationale as get_multi_asset_data.

### Module Structure

```python
"""alloc.models.data — Multi-frequency market data pipeline.

Fetches hourly/daily/weekly price bars from Polygon.io via a cached
client, constructs fixed-dimension state vectors for the RL agent,
and provides latest-price fetching for real-time prediction mode.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from alloc.lib.cache import cache_historical, cache_latest_prices
from alloc.lib.client import PolygonClient

logger = logging.getLogger(__name__)

# ... functions ...
```

### Tests Required

**File:** tests/test_data.py

| Test | Verifies |
|------|----------|
| test_get_multi_asset_data_returns_correct_structure | Dict of ticker -> {hourly, daily, weekly} lists |
| test_get_multi_asset_data_caches | Second call with same args returns cached result |
| test_get_multi_asset_data_injects_client | Function accepts client parameter, no module singleton |
| test_build_state_vector_fixed_dimension | Output shape is deterministic regardless of history length |
| test_build_state_vector_normalization | Prices are normalized by most-recent price in each window |
| test_build_state_vector_padding | Short history is padded to full window length with 1.0 |
| test_build_state_vector_allocation_appended | Current allocation values appear at end of vector |
| test_build_state_vector_zero_price_handling | Division by zero avoided when latest price is 0 |
| test_get_latest_prices_returns_dict | Dict of ticker -> float |
| test_get_latest_prices_caches | Second call returns cached result |
| test_get_latest_prices_invalid_skip_cache | __cache_valid__: False prevents caching |
| test_get_latest_prices_missing_trade | Returns 0.0 for ticker with no trade data |

### Dependencies

- TICKET-001 through TICKET-006 — all prior infrastructure must exist
- alloc/lib/client.py — PolygonClient must be available
- alloc/lib/cache.py — cache_historical and cache_latest_prices decorators must be available

### Improvements Over Seed

1. **Dependency injection** — no module-level singletons; client is passed as a parameter to every function.
2. **Per-window normalization** — each frequency band normalizes by its own most-recent price, not a single cross-frequency price.
3. **Meaningful padding** — pad with 1.0 (flat signal) instead of 0.0 (crash signal).
4. **Logging over print** — all diagnostics go through logging.
5. **No embedded rate limiting** — rate limiting belongs in the client layer, not the data layer.
6. **Separation of concerns** — build_state_vector() receives pre-fetched data; it does not call the API itself.
