# TICKET-027: Decouple `alloc.core.main` from `alloc.cli.main` — two distinct CLI modes

**Module:** `alloc/core.py`, `alloc/cli.py`
**Priority:** Medium — architectural clarity

## Evidence

`alloc/core.py` currently contains **two** CLI entry points:

1. **`alloc.core.parse_args` / `alloc.core.main`** (lines 476–640) — a single-run backtest/predict CLI that builds `SimulationRunner` directly. It has its own argparse parser with `--backtest`/`--predict` modes, `--actor-lr`, `--critic-lr`, `--gamma`, `--tau`, etc.

2. **`alloc.cli.main`** (new, TICKET-023) — a multi-trial training workflow CLI that builds `TrainingConfig` → `WorkflowRunner` → trainer.

These two CLIs overlap in arguments (`--tickers`, `--trading-days`, `--batch-size`, `--risk-aversion`, `--transaction-cost`, `--concentration-penalty`, `--min-cash`, `--verbose`) but serve different purposes:

| | `alloc.core.main` | `alloc.cli.main` |
|---|---|---|
| Purpose | Single simulation run | Multi-trial training + selection |
| Key args | `--backtest`/`--predict`, `--actor-lr`, `--gamma` | `--iterations`, `--update-iterations`, `--fresh-only` |
| Output | Backtest results JSON | Best trial + centroid recommendation |

## Impact

- Users confused about which entry point to use
- Argument duplication and potential default-value drift between the two parsers
- `alloc.core` mixes library code (`SimulationRunner`) with CLI code (`parse_args`/`main`)

## Suggestion

1. **Keep `alloc.cli.main`** as the primary multi-trial training entry point (`python -m alloc`)
2. **Rename `alloc.core.main`** to `alloc.core._cli_main` (private) and expose it via a subcommand or separate module `alloc/cli_single.py` for single-run backtest/predict
3. **Share common arguments** by extracting a `_add_common_args(parser)` helper that both parsers call
4. **Document the distinction** in `docs/CLI.md`:
   - `python -m alloc` → multi-trial training workflow
   - `python -m alloc.single` → single-run backtest/predict

## Verification

- `python -m alloc --help` → multi-trial workflow help
- `python -m alloc.single --help` → single-run backtest help
- No argument defaults differ between the two parsers for shared args
