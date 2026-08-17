# Architecture

How `alloc` is structured and how data flows through it.

## Operating model

`alloc` is a reinforcement-learning portfolio allocation engine. Each
training run produces a **short-lived model snapshot** tuned to current market
conditions. The workflow is cyclical: ingest fresh data -> spawn candidate
models -> rank by Sharpe / outperformance -> deploy -> repeat.

There are two distinct entry points:

1. **Single-run backtest/predict** — `alloc.core.main` (and
   `python -m alloc.core`). Runs one `SimulationRunner` over a historical
   window and serialises results.
2. **Multi-trial workflow** — `alloc.cli.main` (and `python -m alloc`).
   Orchestrates many training trials via `WorkflowRunner`, scoring and ranking
   candidates.

## Data flow (single-run path)

    PolygonClient (alloc.lib.client)
        |  (DiskCache, alloc.lib.cache)
        v
    alloc.models.data
        get_multi_asset_data()  -> hourly/daily/weekly closes
        build_state_vector()    -> fixed-dim state vector
        fetch_latest_prices()   -> latest trade prices
        |
        v
    alloc.core.SimulationRunner.run()
        |  per trading day:
        |    state = build_state_vector(...)
        |    action = networks.get_allocation(state)
        |    portfolio.execute_trades(action, prices)
        |    reward = composite (return, risk, cost, diversification, concentration)
        |    networks.update_critic / update_actor
        v
    results dict -> save_results() -> {path}/results.json | prediction_results.json
    (backtest only) actor/critic .h5 weights saved

## Key design decisions

- **Injected client, no singleton.** `alloc.models.data` and
  `SimulationRunner` receive the `PolygonClient` as an argument, keeping the
  data layer testable with a stub client.
- **Fixed-dimension state.** `build_state_vector` normalises each frequency
  window by its most-recent price (last element = 1.0) and pads short windows
  with zeros, so the state vector has a constant size regardless of history
  length. The current allocation is appended.
- **Cash as residual.** The actor emits `num_assets` sigmoid outputs; the
  `CashLambda` layer enforces a minimum cash fraction on the last element and
  re-normalises to sum 1.0.
- **Soft targets.** DDPG target networks are hard-copied at init and
  soft-updated each step with coefficient `tau`.
- **Realistic execution.** `Portfolio.execute_trades` sells first, then scales
  buys down proportionally if cash is insufficient (shortfall scaling), and
  applies transaction costs on total traded value.

## Model persistence (current state)

`alloc.core.main` (backtest mode) writes `actor_weights.h5` and
`critic_weights.h5` to the model directory. **There is no load path and no
config file** — a saved model cannot be re-instantiated. This is the parity
gap being closed by TICKET-052 (add `save_model`/`load_model` that persist
weights + config) and TICKET-053 (a live-rebalance entry point that loads a
trained model and emits recommended orders).

## Testing

- `tests/` mirrors the package: `test_actor_critic.py`, `test_data.py`,
  `test_portfolio.py`, `test_core.py`, `test_cli.py`, `test_workflow.py`,
  `test_cache.py`, `test_client.py`, `test_dashboard*.py`,
  `test_cycle_signals.py`, `test_replay_buffer.py`, `test_state_builder.py`,
  `test_utils.py`, `test_settings.py`, `test_packages.py`, plus integration
  tests (`test_ddpg_integration.py`, `test_cache_settings_integration.py`,
  `test_dashboard_integration.py`).
- `tests/conftest.py` provides shared fixtures.
