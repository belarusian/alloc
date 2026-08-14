# TICKET-020: Create alloc/utils/workflow.py — TrainingConfig and TrainingTrial dataclasses

**Module:** `alloc/utils/workflow.py` (new)
**Priority:** High — foundational data structures for workflow orchestration

## Problem

The alloc package has no workflow orchestration layer. The seed's `multi_rebalance.py` (1666L) orchestrates multi-iteration training: creates N independent training candidates, evaluates by Sharpe/outperformance, picks best, generates allocation statistics. We need a clean, typed interface that separates orchestration from execution.

## What to Implement

Create `alloc/utils/workflow.py` with:

1. `@dataclass TrainingConfig` — all hyperparameters:
   - `tickers: list[str]` — ticker symbols
   - `positions: dict[str, float]` — current positions (dollar values)
   - `iterations: int` — number of independent training trials
   - `update_iterations: int` — max updates per trial (0 = fresh-only mode)
   - `trading_days: int` — simulation days
   - `batch_size: int` — training batch size
   - `min_allocation: float` — minimum allocation per asset
   - `concentration_penalty: float` — penalty for concentrated positions
   - `transaction_cost: float` — transaction cost factor
   - `risk_aversion: float` — risk aversion parameter
   - `min_cash_alloc: float` — minimum cash allocation
   - `target_sharpe: float` — target Sharpe ratio
   - `target_value: float` — target final portfolio value
   - `target_outperformance: float` — target outperformance percentage

2. `@dataclass TrainingTrial` — per-trial result:
   - `iteration: int` — trial number
   - `update: int` — update number within trial
   - `sharpe_ratio: float | None`
   - `outperformance: float | None`
   - `final_value: float | None`
   - `model_roi: float | None`
   - `buyhold_roi: float | None`
   - `allocation: list[float]` — final allocation vector
   - `model_path: str | None` — path to saved model
   - `results_path: str | None` — path to results JSON

3. `@dataclass WorkflowResult` — aggregate result:
   - `status: str` — success/error
   - `trials: list[TrainingTrial]` — all trial results
   - `best_trial: TrainingTrial` — best by combined score
   - `allocation_stats: dict` — mean/std/min/max per ticker
   - `concentration: dict` — max weight, Herfindahl index
   - `metrics_progression: list[dict]` — per-iteration best metrics

## Dependencies

None — pure data structures at this stage.

## Verification

- `pytest tests/test_workflow.py -xvs` — all tests pass
- `ruff check alloc/utils/` — clean
- `mypy alloc/utils/ --ignore-missing-imports` — clean
