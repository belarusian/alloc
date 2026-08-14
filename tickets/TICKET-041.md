# TICKET-041: Validate ticker-position consistency in CLI and TrainingConfig

**Module:** `alloc/cli.py`, `alloc/utils/workflow.py`
**Priority:** Medium — silent data mismatch risk

## Evidence

In `alloc/cli.py`, `build_config()` (line ~200) maps `args.ticker_list` and `args.positions` independently into `TrainingConfig`. There is no cross-validation that:
1. Every ticker in `--tickers` has a corresponding entry in `--positions-values`
2. Every key in `--positions-values` corresponds to a ticker in `--tickers`

In `alloc/utils/workflow.py`, `TrainingConfig.__post_init__` (line ~70) validates that tickers is non-empty and positions is non-empty with positive values, but does **not** validate that `set(tickers) == set(positions.keys())`.

Example of silent mismatch:
