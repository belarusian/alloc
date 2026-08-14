# TICKET-012: alloc/core.py — Results Serialization and JSON Persistence

**Module:** `alloc/core.py` (serialization helpers)
**Test:** `tests/test_core.py`
**Priority:** Medium — results must be serializable for downstream analysis and model comparison.

## What to Implement

Serialization helpers and result-persistence logic that replace the ad-hoc JSON dumping scattered across the seed's `main()` and `run_portfolio_simulation()`. The seed has two distinct serialization paths: backtest results (lines 856–865) and prediction results (lines 934–960), each with its own numpy-to-list conversion logic. The alloc version consolidates this into reusable helpers.

### Function 1: serialize_results()
