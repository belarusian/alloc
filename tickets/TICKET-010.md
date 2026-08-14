# TICKET-010: alloc/core.py — CLI Entry Point

**Module:** `alloc/core.py` (CLI portion)
**Test:** `tests/test_core.py`
**Priority:** High — CLI is the primary user interface.

## What to Implement

A CLI entry point (`main()`) and argument parser (`parse_args()`) that replaces the seed's procedural CLI.

### Function: parse_args() -> argparse.Namespace

**Required arguments:**
- `--backtest` / `--predict` (mutually exclusive, one required)
- `--tickers` (nargs='+', required) — ticker symbols

**Optional arguments (with defaults):**
- `--initial-value` (float, default=100000)
- `--trading-days` (int, default=242)
- `--model-path` (str, default='saved_models/portfolio_model')
- `--actor-lr` (float, default=0.0001)
- `--critic-lr` (float, default=0.0005)
- `--gamma` (float, default=0.95)
- `--tau` (float, default=0.01)
- `--risk-aversion` (float, default=0.5)
- `--transaction-cost` (float, default=0.001)
- `--diversification-weight` (float, default=0.05)
- `--concentration-penalty` (float, default=0.02)
- `--min-cash` (float, default=0.05)
- `--batch-size` (int, default=32)
- `--replay-capacity` (int, default=50000)
- `--verbose` (flag)

### Function: main()

1. Parse args
2. Load settings from config
3. Initialize PolygonClient with API key from settings
4. Initialize ActorCriticNetworks with args
5. Initialize SimulationRunner with args
6. Call runner.run()
7. Calculate and log metrics
8. Serialize results to JSON
9. Save model weights if backtest mode

### Design Improvements Over Seed

1. **Logging** instead of print
2. **No matplotlib** dependency
3. **Settings class** instead of module-level constants
4. **Type hints** on all functions
