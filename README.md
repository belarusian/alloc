# alloc

Reinforcement-learning portfolio allocation engine that trains actor-critic policy networks on multi-frequency market data and produces optimal asset allocations for a fixed basket of tickers.

## Operating Philosophy

- Every model is a short-lived snapshot optimized for current market conditions. Once it drifts, you retire it and train a new one.
- Overfitting to the latest regime is a feature — each snapshot captures what works now, not what generalizes forever.
- The workflow is cyclical: ingest fresh data → spawn candidates → pick the strongest metrics → deploy → repeat.

## Architecture

**Actor–Critic (DDPG-style)** policy network:

- **Actor** proposes allocation percentages across a fixed basket of assets + cash. Uses per-asset branches with varying widths to prevent symmetric learning collapse. A cash constraint layer guarantees allocations sum to 1.0 with a configurable minimum cash floor.
- **Critic** evaluates each proposed allocation by estimating expected future return via Q-value regression.
- Target networks are soft-updated for stable learning. Experience replay breaks temporal correlation.

**Multi-objective reward function:**

| Component | Purpose |
|---|---|
| Portfolio return | Weighted sum of asset returns |
| Risk penalty | Volatility-adjusted drawdown |
| Transaction cost | Penalizes unnecessary turnover |
| Diversification bonus | Shannon entropy + HHI — rewards spreading risk |
| Concentration penalty | Quadratic penalty on oversized positions |

**Multi-frequency state construction:**

Each tick's state combines 24 hourly returns, 10 daily returns, and 4 weekly returns per asset — giving the model short-term momentum, medium-term trend, and long-term direction. A `day_index` gate prevents lookahead bias during backtest.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure Polygon.io API key
cp .env.example .env
# Edit .env, set POLYGON_API_KEY

# Train a model on a basket of tickers
python -m alloc.core --backtest \
  --tickers AAPL,META,GOOG,NVDA \
  --trading-days 242 \
  --plot

# Get allocation recommendation from trained model
python -m alloc.core --predict \
  --tickers AAPL,META,GOOG,NVDA \
  --model-path results/my_model
```

## Repository Layout

- `alloc/` — Core Python package (networks, portfolio engine, data pipeline, configuration)
- `utils/` — CLI entry points and workflow tools
- `tests/` — Test suite
- `scripts/` — Bootstrap and health-check helpers
- `adr/` — Architecture decision records

## Testing

```bash
pytest tests/ -x -q
```

## License

MIT
