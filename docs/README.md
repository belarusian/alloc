# alloc — Documentation

Welcome. This directory documents the `alloc` package for newcomers landing
at the repo.

## Start here

1. **README.md** (repo root) — what `alloc` is, install, usage, philosophy.
2. **docs/MODULES.md** — catalog of every module and its relationships.
3. **docs/ARCHITECTURE.md** — how the system is structured and how data flows.
4. **docs/API.md** — reference of the public interfaces.

## Quick orientation

- **Two entry points:**
  - `python -m alloc.core --backtest --tickers AAPL,META` — single-run
    backtest/predict.
  - `python -m alloc --tickers AAPL,META --iterations 5` — multi-trial
    training workflow.
- **Core loop:** fetch multi-frequency data -> build a fixed-dim state vector
  -> DDPG actor emits an allocation -> portfolio executes trades -> composite
  reward -> update actor/critic.
- **Models are short-lived snapshots** tuned to the current regime; retrain
  when the regime shifts.

## Open work (parity gap)

The seed's predict mode is a true live-rebalance (load trained model ->
recommended orders). `alloc` does not yet have that. Tracked as:

- **TICKET-052** — model persistence round-trip
  (`ActorCriticNetworks.save_model` / `load_model`).
- **TICKET-053** — live-rebalance entry point
  (`alloc.lib.rebalance.rebalance_portfolio` + CLI wiring).

See `tickets/` for full detail.

## Conventions

- The client is **injected**, never a module-level singleton.
- Public symbols are documented in `docs/API.md`; internal helpers use a
  leading underscore.
- Every module has a module-level docstring; public functions are documented.
