# TICKET-053: Live-rebalance entry point (load trained model → recommended orders)

**Status:** OPEN
**Cycle:** 43
**Priority:** High
**Depends on:** TICKET-052 (model persistence round-trip)
**Target module:** new `alloc/lib/rebalance.py` + `alloc/core.py` CLI wiring

## Summary

The seed's predict mode is a **true live-rebalance**: it loads a previously
trained actor-critic model, fetches the latest market prices, constructs the
current market state, calls the model's public `get_allocation` to produce a
recommended allocation, executes the trades against the current portfolio
(transaction costs + shortfall scaling), and emits recommended orders plus the
post-execution portfolio value.

alloc/ has **no equivalent**. Its `--predict` CLI flag (`alloc/core.py`
`main`) just runs a fresh `SimulationRunner.run()` and saves
`prediction_results.json`; it never loads a trained model, never fetches
latest prices to build a live state, and never emits recommended orders from a
trained model.

## Evidence

- `alloc/core.py` `main` (lines 635-745): `--predict` sets `mode_str`
  (line 655) but the code path is identical to backtest except it skips saving
  weights (lines 733-735 only run under `if args.backtest`). No model load, no
  latest-price fetch, no `get_allocation` on a live state.
- `alloc/models/data.py` already provides the pieces:
  - `get_multi_asset_data(tickers, client, ...)` (line 161)
  - `build_state_vector(multi_freq_data, current_allocation, tickers,
    n_hourly, n_daily, n_weekly)` (line 255)
  - `fetch_latest_prices(tickers, client)` (line 334)
- `alloc/models/portfolio.py`:
  - `Portfolio.__init__(tickers, initial_cash=100_000.0,
    transaction_cost=0.001)` (line 49) — note `shares_held` is zeroed at init
    (line 64), so seeding from live positions requires mutating
    `portfolio.shares_held` and `portfolio.cash` after construction.
  - `Portfolio.get_allocation(prices)` (line 72) — current weights incl. cash.
  - `Portfolio.execute_trades(target_allocation, prices)` (line 93) — rebalance
    execution with shortfall scaling and transaction costs; returns a dict with
    `scale_factor` and `total_transaction_costs`.
- `alloc/models/networks.py` `ActorCriticNetworks.get_allocation(state)`
  (line 488) is the public inference entry point (returns a length-
  `num_assets` vector, last element = cash).
- The only missing glue is: load model → fetch data → build state →
  `get_allocation` → seed a `Portfolio` from current positions →
  `execute_trades` → return recommended orders + post-execution value.

## Implementation plan

1. **New module `alloc/lib/rebalance.py`** with a
   `rebalance_portfolio(...)` function (and a small dataclass or dict result):
   - Inputs: `model_path`, `tickers`, `positions` (dict ticker→dollar value),
     `client`, `n_hourly=5, n_daily=5, n_weekly=5`, `transaction_cost=0.0`,
     `initial_value` (optional; derived from positions + cash if omitted).
   - Steps:
     a. `networks = ActorCriticNetworks.load_model(model_path)` (TICKET-052).
     b. `prices = fetch_latest_prices(tickers, client)`.
     c. `multi_freq = get_multi_asset_data(tickers, client)`.
     d. Build `current_allocation` (non-cash weights) from `positions` +
        `prices` (value / total, cash = remainder).
     e. `state = build_state_vector(multi_freq, current_allocation, tickers,
        n_hourly, n_daily, n_weekly)`.
     f. `allocation = networks.get_allocation(state)`.
     g. Seed a `Portfolio(tickers, initial_cash=initial_value,
        transaction_cost=...)`; then set `portfolio.shares_held[t] =
        positions[t] / prices[t]` and `portfolio.cash = initial_value -
        sum(positions.values())` (Portfolio zeroes shares at init, line 64).
     h. `target = {t: allocation[i] for i, t in enumerate(tickers)};
        target['cash'] = allocation[-1]`.
     i. `execution = portfolio.execute_trades(target, prices)`.
   - Return a dict with: `recommended_allocation` (ticker→weight incl. cash),
     `recommended_orders` (list of {ticker, action, shares, price, value}),
     `portfolio_value_before`, `portfolio_value_after`,
     `total_transaction_costs`, `scale_factor`.
2. **CLI wiring in `alloc/core.py`**: add a `--rebalance` mode (or a
   `--positions` + `--model-path` predict path) that calls
   `rebalance_portfolio` and logs the recommended orders, mirroring the seed's
   predict output. Keep the existing `--predict` (forward simulation) intact.
3. **Tests** (`tests/test_rebalance.py`): use a fake client (patch
   `fetch_latest_prices`/`get_multi_asset_data` at the module level via
   `patch.object` on the rebalance module's imported references, or inject a
   stub client) and a small saved model (via TICKET-052 `save_model`) to
   assert: allocation sums to 1.0, cash >= min_cash, orders are produced,
   post-execution value is finite, and transaction costs are applied.

## Verification

- `pytest tests/test_rebalance.py -x -q` — all pass.
- `pytest tests/ -x -q` — full suite green.
- `ruff check alloc/` — clean.
- `mypy alloc/ --ignore-missing-imports` — clean.

## Notes

- **Semantics reference:** read `~/Research/new-trader/trader/core.py`
  (predict mode, ~lines 863-935) and
  `~/Research/new-trader/utils/trader_tools/multi_rebalance_helper.py`
  (`rebalance_portfolio`, `positions_to_allocation`) for understanding only.
  Nothing is copied; the implementation targets alloc's existing public API
  (`get_allocation`, `build_state_vector`, `fetch_latest_prices`,
  `execute_trades`).
