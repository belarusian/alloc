# TICKET-011: alloc/lib/metrics.py — Decoupled Performance Metrics

**Module:** `alloc/lib/metrics.py` (new file)
**Test:** `tests/test_metrics.py`
**Priority:** Medium — metrics are currently embedded in the simulation function; decoupling enables reuse and testing.

## What to Implement

A standalone metrics calculation module that replaces the inline metrics logic scattered throughout the seed's `run_portfolio_simulation()` (lines 500–640 in `~/Research/new-trader/trader/core.py`). The seed calculates Sharpe ratio, max drawdown, volatility, win rate, buy-and-hold benchmark, and outperformance directly inside the simulation function. The alloc version extracts these into pure functions.

### Function 1: `calculate_sharpe_ratio(daily_returns: list[float], risk_free_rate: float = 0.0) -> float`

Annualized Sharpe ratio. Daily returns → mean/std → annualize by sqrt(252).

### Function 2: `calculate_max_drawdown(portfolio_values: list[float]) -> float`

Maximum peak-to-trough decline. Returns negative value.

### Function 3: `calculate_volatility(daily_returns: list[float]) -> float`

Annualized volatility (std of daily returns * sqrt(252))

### Function 4: `calculate_win_rate(daily_returns: list[float]) -> float`

Fraction of positive daily returns.

### Function 5: `calculate_buy_and_hold(returns: list[float], initial_value: float) -> float`

Buy-and-hold benchmark final value.

### Function 6: `calculate_outperformance(portfolio_values: list[float], benchmark_values: list[float]) -> float`

Percentage outperformance vs benchmark.

### Design Improvements Over Seed

1. **Pure functions** — seed calculates metrics inline with mutable state
2. **Testable** — each metric is independently testable
3. **Reusable** — metrics can be used in reports, dashboards, workflow

### Dependencies

- None

### Tests

**File:** `tests/test_metrics.py`

| Test | Verifies |
|---|---|
| `test_sharpe_ratio_positive` | Positive returns yield positive Sharpe |
| `test_sharpe_ratio_zero` | Zero volatility returns 0 |
| `test_max_drawdown_simple` | Correct max drawdown calculation |
| `test_volatility_scaling` | Annualization factor applied |
| `test_win_rate` | Correct fraction of positive returns |
