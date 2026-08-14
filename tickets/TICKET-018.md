# TICKET-018: Create alloc/lib/utils.py — PricePreprocessor class

**Module:** `alloc/lib/utils.py` (extends TICKET-017)
**Source:** `~/Research/new-trader/trader/lib/utils.py` (lines 70–200)
**Priority:** High — price preprocessing is needed by the simulation loop for efficient lookups

## Problem

The seed's `preprocess_price_data` function (130 lines) is the heavy lifter that takes raw price data per ticker, normalizes timestamps, identifies trailing contiguous windows (allowing 5-day gaps for weekends/holidays), finds common dates across all tickers, and builds a day-indexed lookup structure. This is needed by the simulation loop to efficiently look up prices for any day.

## What to Implement

Extend `alloc/lib/utils.py` with:

1. `@dataclass PriceIndex` — structured return with `days_available: int`, `prices: dict[int, dict[str, float]]`, `complete: bool`, `missing_tickers: list[str]`, `dates: list[datetime]`
2. `class PricePreprocessor` — configurable gap tolerance (default 5 days), method `preprocess(tickers: list[str], price_data: dict[str, dict]) -> PriceIndex`
3. Internal methods: `_normalize_timestamps`, `_find_trailing_window`, `_intersect_dates`, `_build_price_index`

## Design Improvements Over Seed

- Typed signatures with dataclass returns instead of dicts with string keys
- `datetime.fromtimestamp(ts, tz=timezone.utc)` instead of deprecated `datetime.utcfromtimestamp`
- Configurable gap tolerance via constructor parameter
- Separated internal methods for testability

## Dependencies

TICKET-017 (utility functions in same module)

## Verification

- `pytest tests/test_utils.py -xvs` — all tests pass including preprocessing tests
- `ruff check alloc/lib/utils.py` — clean
- `mypy alloc/lib/utils.py --ignore-missing-imports` — clean
