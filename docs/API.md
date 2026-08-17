# Public API Reference

Reference of the public interfaces exposed by `alloc`. Signatures are
verified against the source. Internal helpers (leading underscore) are
omitted unless noted.

## alloc.models.networks

### class ActorCriticNetworks

DDPG actor-critic pair for portfolio allocation.

    ActorCriticNetworks(
        input_dim: int,
        num_assets: int,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        dropout: float = 0.1,
        gamma: float = 0.99,
        tau: float = 0.005,
        min_cash_allocation: float = 0.0,
        buffer_capacity: int = 1_000_000,
        seed: int | None = None,
    )

Attributes: `input_dim`, `num_assets`, `gamma`, `tau`,
`min_cash_allocation`, `dropout`, `actor`, `critic`, `actor_target`,
`critic_target`, `actor_optimizer`, `critic_optimizer`, `replay_buffer`.

Methods:

- `get_allocation(state, add_noise=False, noise_scale=0.1) -> np.ndarray`
  — public inference entry point. Returns a length-`num_assets` allocation
  vector summing to 1.0, last element = cash (>= `min_cash_allocation`).
- `update_critic(...)` / `update_actor(...)` — DDPG training steps.
- `_soft_update_targets()` — soft-update target networks (internal).

> **Gap (TICKET-052):** no `save_model(directory)` / `load_model(directory)`
> round-trip yet. Only raw `.h5` weights are saved by `alloc.core.main`.

### class ReplayBuffer

Fixed-capacity experience replay. `add(...)`, `sample(n)`, `__len__`.

### class CashLayer / CashLambda

Keras layers enforcing a minimum cash fraction on the last allocation
element. `get_config()` returns `{"min_cash": ...}`.

## alloc.models.data

- `get_multi_asset_data(tickers, client, end_date=None, hourly_days=7,
  daily_days=365, weekly_weeks=52) -> dict[str, dict[str, list[float]]]`
  — fetch hourly/daily/weekly closes.
- `build_state_vector(multi_freq_data, current_allocation, tickers,
  n_hourly, n_daily, n_weekly) -> np.ndarray` — fixed-dim state vector
  (normalised price windows + current allocation appended).
- `fetch_latest_prices(tickers, client) -> dict[str, float]` — latest trade
  price per ticker (0.0 on error).

## alloc.models.portfolio

### class Portfolio

    Portfolio(tickers: list[str], initial_cash: float = 100_000.0,
              transaction_cost: float = 0.001)

Attributes: `tickers`, `cash`, `transaction_cost`, `shares_held`
(zeroed at init), `portfolio_values`.

Methods:

- `get_portfolio_value(prices) -> float`
- `get_allocation(prices) -> dict[str, float]` — current weights incl. `'cash'`.
- `execute_trades(target_allocation, prices) -> dict` — rebalance with
  shortfall scaling + transaction costs; returns metadata incl.
  `scale_factor`, `total_transaction_costs`.

## alloc.core

- `SimulationRunner(tickers, initial_value, networks, data_pipeline, client,
  transaction_cost, risk_aversion, gamma, tau, diversification_weight,
  concentration_penalty, min_cash, batch_size, verbose)` — `.run(trading_days)`.
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> None` — backtest/predict CLI entry point.
- `create_trainer(conservative=False) -> Callable` — trainer factory used by
  the workflow CLI.
- `save_results(results, path, mode="backtest")`,
  `load_results(path, mode="backtest")`, `serialize_results(results)`.

## alloc.cli

Multi-trial training workflow CLI.

- `build_parser() -> argparse.ArgumentParser`
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> int` — exit codes: 0 success, 1 user error, 2 workflow
  failure, 3 unexpected.

## alloc.utils.workflow

- `TrainingConfig` — dataclass of training hyperparameters.
- `TrainingTrial` / `WorkflowResult` — result containers.
- `WorkflowRunner(config, trainer)` — `.run() -> WorkflowResult`.

## alloc.lib

- `PolygonClient(api_key, cache)` — Polygon.io wrapper (cached).
- `DiskCache(cache_dir, enabled)` — disk cache with per-type TTL.
- `crawl_package(...)` (dashboard) / `generate_html(...)`, `publish(...)`
  (publish_dashboard).
- `alloc.lib.utils` — scalar coercion, allocation formatting, price-index
  helpers.

## Planned (TICKET-053)

- `alloc.lib.rebalance.rebalance_portfolio(model_path, tickers, positions,
  client, n_hourly=5, n_daily=5, n_weekly=5, transaction_cost=0.0,
  initial_value=None) -> dict` — load trained model, fetch latest prices,
  build live state, `get_allocation`, seed `Portfolio` from positions,
  `execute_trades`, return recommended orders + post-execution value.
