# TICKET-021: Create alloc/utils/workflow.py — WorkflowRunner class

**Module:** `alloc/utils/workflow.py` (extends TICKET-020)
**Priority:** High — core orchestration engine

## Problem

The seed's `run_complete_workflow` (500+ lines) orchestrates the training loop: spawns N trials, tracks best metrics, computes allocation statistics, finds centroid recommendation. We need a clean `WorkflowRunner` class that implements this without subprocess calls or shell script dependencies.

## What to Implement

Extend `alloc/utils/workflow.py` with:

1. `class WorkflowRunner`:
   - `__init__(self, config: TrainingConfig, trainer: Callable[..., dict])` — inject training callable
   - `run(self) -> WorkflowResult` — execute full multi-trial workflow
   - `_run_trial(self, trial_num: int) -> TrainingTrial` — single trial execution
   - `_combined_score(self, trial: TrainingTrial) -> float` — 0.5×Sharpe + 0.5×(outperformance/10)
   - `_allocation_stats(self, trials: list[TrainingTrial]) -> dict` — mean/std/min/max per ticker
   - `_concentration_metrics(self, trials: list[TrainingTrial]) -> dict` — max weight, Herfindahl

2. Key design decisions:
   - Trainer is injected (dependency injection) — no hardcoded subprocess calls
   - Fresh-only mode: `update_iterations=0` means each trial trains from scratch
   - All trials run sequentially (no threading yet)
   - Results aggregated into `WorkflowResult`

## Dependencies

TICKET-020 (dataclasses)

## Verification

- `pytest tests/test_workflow.py -xvs` — all tests pass
- `ruff check alloc/utils/` — clean
- `mypy alloc/utils/ --ignore-missing-imports` — clean
