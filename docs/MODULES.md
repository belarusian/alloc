# Module Catalog

Catalog of every module in the `alloc` package and its relationships.
Line counts and public symbols are verified against the source.

## Package layout

    alloc/
    ├── __init__.py            # package marker, __version__
    ├── __main__.py            # `python -m alloc` -> alloc.cli.main
    ├── cli.py                 # multi-trial training workflow CLI
    ├── core.py                # SimulationRunner + backtest/predict CLI + results I/O
    ├── models/
    │   ├── __init__.py
    │   ├── data.py            # multi-frequency data fetch + state vector build
    │   ├── networks.py        # DDPG actor-critic + replay buffer + CashLambda
    │   └── portfolio.py       # Portfolio tracking, trade execution, reward
    ├── lib/
    │   ├── __init__.py
    │   ├── cache.py           # DiskCache + TTL decorator helpers
    │   ├── client.py          # Polygon.io StocksClient wrapper (cached)
    │   ├── cycle_signals.py   # terminal tree-view of health signals
    │   ├── dashboard.py       # crawls package -> JSON health metadata
    │   ├── publish_dashboard.py  # JSON metadata -> HTML dashboard
    │   └── utils.py           # scalar coercion, formatting, price-index helpers
    └── utils/
        ├── __init__.py
        └── workflow.py        # TrainingConfig / WorkflowRunner multi-trial orchestration

## Module responsibilities

| Module | Responsibility | Key public symbols |
|---|---|---|
| `alloc.core` | DDPG simulation loop; backtest/predict CLI; results serialisation | `SimulationRunner`, `parse_args`, `main`, `create_trainer`, `save_results`, `load_results`, `serialize_results` |
| `alloc.cli` | Multi-trial training workflow CLI (argparse, typed converters, exit codes) | `build_parser`, `parse_args`, `main`, `EXIT_*` |
| `alloc.models.data` | Fetch hourly/daily/weekly prices; build fixed-dim state vectors | `get_multi_asset_data`, `build_state_vector`, `fetch_latest_prices` |
| `alloc.models.networks` | DDPG actor-critic pair, soft targets, replay buffer, cash constraint | `ActorCriticNetworks`, `ReplayBuffer`, `CashLayer`, `CashLambda`, `_calculate_cash` |
| `alloc.models.portfolio` | Holdings tracking, trade execution (shortfall scaling + costs), reward | `Portfolio` |
| `alloc.lib.client` | Polygon.io API wrapper with disk caching | `PolygonClient` |
| `alloc.lib.cache` | Disk-backed cache with per-type TTL | `DiskCache` (+ decorator helpers) |
| `alloc.lib.cycle_signals` | Terminal tree-view of dashboard health signals | (viewer entry points) |
| `alloc.lib.dashboard` | Crawl `alloc/` -> per-module JSON health metadata | `crawl_package` |
| `alloc.lib.publish_dashboard` | Render JSON metadata -> HTML dashboard | `generate_html`, `publish` |
| `alloc.lib.utils` | Scalar-price coercion, allocation formatting, timestamp/price-index helpers | (helper functions) |
| `alloc.utils.workflow` | Multi-trial training orchestration and scoring | `TrainingConfig`, `TrainingTrial`, `WorkflowResult`, `WorkflowRunner` |

## Dependency graph (intra-package)

    alloc.cli -> alloc.utils.workflow -> alloc.core.create_trainer
    alloc.__main__ -> alloc.cli
    alloc.core -> alloc.models.networks
               -> alloc.models.data
               -> alloc.models.portfolio
               -> alloc.lib.client -> alloc.lib.cache
    alloc.models.networks -> (tensorflow / keras only)
    alloc.models.data -> (client injected, no hard dep)
    alloc.lib.dashboard -> alloc.lib.publish_dashboard (optional, CLI-driven)

Notes:
- `alloc.core` is the central orchestrator for the single-run backtest/predict
  path; `alloc.cli` is the separate multi-trial workflow entry point.
- `alloc.models.data` receives the client as an injected argument (no
  module-level singleton), keeping it testable.
- `alloc.lib.dashboard` / `publish_dashboard` are optional, invoked only when
  the CLI `--publish-dashboard` flag is set.

## Known gaps (tracked as tickets)

- **TICKET-052** — `ActorCriticNetworks` has no `save_model`/`load_model`
  round-trip; only raw `.h5` weights are saved by `alloc.core.main`.
- **TICKET-053** — No live-rebalance entry point; `--predict` runs a fresh
  forward simulation rather than loading a trained model and emitting
  recommended orders.
