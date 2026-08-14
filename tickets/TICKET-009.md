# TICKET-009: alloc/core.py — SimulationRunner Class and Simulation Loop

**Module:** `alloc/core.py` (new file)
**Test:** `tests/test_core.py`
**Priority:** High — this is the central orchestrator; without it, the engine cannot run.

## What to Implement

A class-based simulation runner that replaces the procedural `run_portfolio_simulation()` function from the seed. The new design encapsulates simulation state, separates concerns, and uses logging instead of print statements.

### Class: SimulationRunner

**Constructor parameters:**
- `tickers: list[str]` — ticker symbols
- `initial_value: float` — starting capital
- `networks: ActorCriticNetworks` — the DDPG model
- `data_pipeline: Any` — data fetching module (alloc.models.data)
- `client: Any` — PolygonClient instance
- `transaction_cost: float = 0.001`
- `risk_aversion: float = 0.5`
- `gamma: float = 0.95`
- `tau: float = 0.01`
- `diversification_weight: float = 0.05`
- `concentration_penalty: float = 0.02`
- `min_cash: float = 0.05`
- `batch_size: int = 32`
- `verbose: bool = False`

**Method: run(trading_days: int) -> dict**
Per-day loop:
1. Fetch current prices for all tickers
2. Build state vector from multi-frequency data + current allocation
3. Actor proposes allocation (with noise during exploration phase: first half of days)
4. Portfolio executes trades (sell first, then buy, with shortfall scaling)
5. Calculate composite reward (return - risk_penalty - transaction_cost + diversification)
6. Build next state vector
7. Update networks via TD error (except last day unless continuous_learning)
8. Track: portfolio_values, daily_returns, rewards, allocation_history, dates

**Return value:** dict with keys:
- `final_value`, `initial_value`, `portfolio_values`, `daily_returns`
- `rewards`, `allocation_history`, `dates`
- `final_holdings`, `final_prices`

### Design Improvements Over Seed

1. **Class-based** — seed is a 500-line procedural function with 20+ parameters
2. **No print statements** — use logging at appropriate levels
3. **No matplotlib** — defer plotting to a separate module
4. **Dependency injection** — all dependencies passed to constructor
5. **Type hints** — all parameters and return values fully typed

### Dependencies

- `alloc.models.networks.ActorCriticNetworks`
- `alloc.models.portfolio.Portfolio`
- `alloc.models.portfolio.calculate_portfolio_reward`
- `alloc.models.data.get_multi_asset_data`
- `alloc.models.data.build_state_vector`
- `alloc.models.data.fetch_latest_prices`
