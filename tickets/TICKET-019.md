# TICKET-019: Create tests/test_utils.py — Utility functions and PricePreprocessor tests

**Module:** `tests/test_utils.py` (new)
**Priority:** High — tests for TICKET-017 and TICKET-018

## What to Implement

Create `tests/test_utils.py` with comprehensive tests:

### ensure_scalar_price tests
- numpy scalar → float
- numpy array → float via `.item()`
- Single-element list → float
- Single-element tuple → float
- Direct float passthrough

### format_allocation tests
- Basic formatting with default precision
- Custom precision
- Sorted by weight descending
- Percentage conversion
- Empty dict handling

### create_timestamp_string tests
- Correct format: `YYYYMMDD_HHMMSS`
- Returns string type
- 15 characters length

### safe_divide tests
- Normal division
- Zero denominator returns default
- Custom default value
- Negative numbers

### PricePreprocessor tests
- Basic preprocessing with complete data
- Missing tickers detection
- Trailing window identification
- 5-day gap tolerance
- Date intersection across tickers
- Price index lookup correctness
- Empty data handling
- Timestamp normalization

## Dependencies

TICKET-017, TICKET-018

## Verification

- `pytest tests/test_utils.py -xvs` — all tests pass
- `ruff check tests/test_utils.py` — clean
- `mypy tests/test_utils.py --ignore-missing-imports` — clean
