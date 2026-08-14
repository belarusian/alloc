# alloc

**Multi-asset portfolio management system using DDPG reinforcement learning.**

`alloc` trains Deep Deterministic Policy Gradient (DDPG) actor-critic networks on multi-frequency market data (hourly, daily, weekly) and produces optimal asset allocations for a fixed basket of tickers. Each training run yields a short-lived model snapshot tuned to current market conditions — when the regime shifts, you retrain.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Testing](#testing)
- [License](#license)
- [Contributing](#contributing)

---

## Project Overview

### Purpose

`alloc` is a reinforcement-learning portfolio allocation engine designed for systematic trading. It:

1. **Fetches** multi-frequency price data (hourly, daily, weekly) from Polygon.io.
2. **Builds** fixed-dimension state vectors from normalised price windows and current allocations.
3. **Trains** a DDPG actor-critic pair to learn an allocation policy that maximises a composite reward signal combining return, risk, transaction costs, diversification, and concentration.
4. **Simulates** portfolio rebalancing with realistic trade execution (shortfall scaling, transaction costs, cash constraints).
5. **Orchestrates** multi-trial training workflows that score and rank candidate models by Sharpe ratio and outperformance vs. buy-and-hold.

### Operating Philosophy

- **Every model is a short-lived snapshot** optimised for current market conditions. Once it drifts, retire it and train a new one.
- **Overfitting to the latest regime is a feature** — each snapshot captures what works now, not what generalises forever.
- **The workflow is cyclical**: ingest fresh data → spawn candidates → pick the strongest metrics → deploy → repeat.

### Key Modules

| Module | Responsibility |
|---|---|
| `alloc.lib.client` | Polygon.io API wrapper with disk caching |
| `alloc.lib.cache` | Disk-based cache with configurable TTL per data type |
| `alloc.models.data` | Multi-frequency data fetching and state vector construction |
| `alloc.models.networks` | DDPG actor-critic networks and replay buffer |
| `alloc.models.portfolio` | Portfolio tracking, trade execution, reward calculation |
| `alloc.core` | Simulation runner and results serialisation |
| `alloc.utils.workflow` | Multi-trial training orchestration and scoring |
| `alloc.cli` | Command-line interface for the training workflow |
| `alloc.config.settings` | Environment-driven configuration management |

---

## Installation

### Prerequisites

- Python ≥ 3.10
- A [Polygon.io](https://polygon.io) API key

### Steps
