# TICKET-011: alloc/lib/metrics.py — Decoupled Performance Metrics

**Module:** `alloc/lib/metrics.py` (new file)
**Test:** `tests/test_metrics.py`
**Priority:** Medium — metrics are currently embedded in the simulation function; decoupling enables reuse and testing.

## What to Implement

A standalone metrics calculation module that replaces the inline metrics logic scattered throughout the seed's `run_portfolio_simulation()` (lines 500–640 in `~/Research/new-trader/trader/core.py`). The seed calculates Sharpe ratio, max drawdown, volatility, win rate, buy-and-hold benchmark, and outperformance directly inside the simulation function. The alloc version extracts these into pure functions.

### Function 1: calculate_sharpe_ratio()
