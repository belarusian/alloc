# TICKET-3: SimulationRunner.run() returns raw simulation dict — _trainer must compute derived metrics (sharpe_ratio, outperformance, model_roi, buyhold_roi)

## Evidence
- `alloc/core.py` `SimulationRunner.run()` returns: `final_value, initial_value, portfolio_values, daily_returns, rewards, allocation_history, dates, final_holdings, final_prices`
- `alloc/core.py` `_trainer` closure must compute: `sharpe_ratio, outperformance, model_roi, buyhold_roi` from these raw results
- `grep -n "sharpe_ratio\|outperformance\|model_roi\|buyhold_roi" alloc/core.py` — need to verify these are computed in `_trainer`

## Impact
- If `_trainer` does not compute these metrics, `TrainingTrial` fields will be `None`, causing `WorkflowRunner._combined_score()` to default to 0.
- Trial ranking becomes meaningless without proper metric computation.
- CLI output shows zeros for all metrics.

## Suggestion
- Verify `_trainer` closure computes all 4 derived metrics from `SimulationRunner.run()` output.
- If missing, add metric computation: Sharpe from `daily_returns`, outperformance vs buy-and-hold, ROI calculations.
- Add fallback defaults and NaN handling.

## Implementation Plan
1. Read full `_trainer` body to confirm metric computation exists
2. If missing, add `compute_sharpe(daily_returns)`, `compute_buyhold_roi(final_prices, initial_prices)`, etc.
3. Ensure `model_roi` and `buyhold_roi` are computed as percentages
4. Add validation: raise `ValueError` if metrics are NaN or infinite
5. Add unit test with synthetic returns asserting metric correctness
