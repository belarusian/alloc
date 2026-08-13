# TICKET-006: Portfolio Execution Engine & Reward Calculation

**Module:** `alloc/models/portfolio.py`
**Test:** `tests/test_portfolio.py`
**Priority:** High — core portfolio management for RL agent.

## What to Implement

A `Portfolio` class and a standalone `calculate_portfolio_reward()` function. Read `~/Research/new-trader/trader/models/portfolio.py` for understanding — never copy.

### Class: `Portfolio`

**Constructor:** `Portfolio(tickers: list[str], initial_cash: float = 100_000, transaction_cost: float = 0.001)`
- Tracks `shares_held` (dict ticker → float), `cash` (float), `transaction_cost`
- History arrays: `portfolio_values`, `allocations_history`, `transactions_history`

**Methods:**
- `get_portfolio_value(prices: dict[str, float]) -> float`: sum of (shares × price) + cash
- `get_allocation(prices: dict[str, float]) -> dict[str, float]`: percentage of portfolio per ticker + cash; handles zero-value edge case
- `execute_trades(target_allocation: dict[str, float], prices: dict[str, float]) -> dict`: core trade engine
  1. Calculate current vs target allocation for each ticker
  2. Separate into buys and sells (threshold: 1% of portfolio value)
  3. Execute sells first to raise cash
  4. Reserve cash for transaction costs
  5. Execute buys with available cash; if shortfall, scale buys proportionally
  6. Apply transaction costs (total traded value × cost factor)
  7. Return execution dict with trades, costs, final allocation, portfolio value after

### Function: `calculate_portfolio_reward()`

Signature: `calculate_portfolio_reward(current_allocation, previous_allocation, returns, risk_aversion=0.5, transaction_cost=0.001, diversification_weight=0.05, concentration_penalty=0.02) -> float`

Components:
1. **Portfolio return**: dot product of allocation (ex-cash) × asset returns
2. **Risk penalty**: `risk_aversion × std(returns × allocation)`
3. **Transaction penalty**: `sum(abs(current - previous)) × transaction_cost`
4. **Diversification bonus**: combined Shannon entropy (40%), normalized HHI (30%), participation ratio (30%), sqrt-transformed, scaled by log(num_assets)
5. **Concentration penalty**: quadratic penalty when max allocation exceeds threshold (min(0.4, ideal × 1.5))

### Tests Required

- Portfolio initialization and value calculation
- Allocation percentages from mixed positions
- Trade execution: sell-first ordering, shortfall scaling, transaction cost deduction
- Reward function: each component in isolation (return, risk, diversification, concentration)
- Edge cases: zero portfolio value, all-cash allocation, single-ticker concentration
