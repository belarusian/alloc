# TICKET-045: Wire Portfolio value-history into SimulationRunner; retire ad-hoc Sharpe/ROI

**Status:** OPEN
**Date:** 2025-08-16
**Cycle:** 36
**Priority:** Medium
**Issue:** https://github.com/belarusian/alloc/issues/111

## Summary

`alloc/models/portfolio.py` now tracks `portfolio_values` and exposes
`calculate_returns()` (daily / cumulative / annualized / Sharpe / max-drawdown)
and `calculate_portfolio_statistics()`. However, `SimulationRunner` in
`alloc/core.py` never calls `record_value()`, so the history stays at its
single seeded entry and the new methods are dead code in the live path.
Meanwhile `core.py` still computes Sharpe and ROI ad-hoc.

## Evidence

- `alloc/models/portfolio.py` — `Portfolio.__init__` seeds
  `self.portfolio_values = [float(initial_cash)]`; `record_value()` appends.
- `alloc/core.py` — `SimulationRunner.run()` builds `portfolio_values` as a
  local list (returned in `results["portfolio_values"]`) but never invokes
  `portfolio.record_value(...)`.
- `alloc/core.py` — the `_trainer` closure recomputes Sharpe inline:
  `daily_returns = np.diff(values) / np.maximum(values[:-1], 1e-8)` and
  `sharpe_ratio = mean/std * sqrt(252)`, and ROI as
  `(final_value - initial_value) / initial_value * 100`. These duplicate the
  logic now centralised in `Portfolio.calculate_returns()`.

## Impact

- The seed-parity capability is present but unused in the production loop.
- Two divergent Sharpe implementations (core.py ad-hoc vs. Portfolio method)
  risk drifting apart; core.py's version uses `np.diff/np.maximum` while the
  Portfolio method uses a guard on `values[i-1] > 0`.
- No max-drawdown or annualized-return is surfaced in results today.

## Suggestion (implementation plan)

1. In `SimulationRunner.run()`, after each day's trades execute, call
   `portfolio.record_value(prices)` so `portfolio.portfolio_values` mirrors the
   per-day valuation series.
2. Replace the ad-hoc Sharpe/ROI block in `_trainer` with a call to
   `portfolio.calculate_returns()`; map its keys onto the existing result
   fields (`sharpe_ratio`, `model_roi` from `cumulative_return * 100`).
3. Add `max_drawdown` and `annualized_return` to the returned results dict.
4. Keep the `np.maximum(values[:-1], 1e-8)` guard semantics or reconcile with
   the Portfolio method's `> 0` guard — pick one and document it.
5. Add a test in `tests/test_core.py` asserting `results["portfolio_values"]`
   length equals `trading_days + 1` and that `sharpe_ratio` matches
   `portfolio.calculate_returns()["sharpe_ratio"]`.

## Acceptance criteria

- `SimulationRunner` populates `portfolio.portfolio_values` via `record_value`.
- Sharpe/ROI in results are derived from `Portfolio.calculate_returns()`.
- `max_drawdown` and `annualized_return` present in results.
- Full test suite green.
