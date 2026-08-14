# TICKET-2: `recommended_trades` key is absent from trainer output — audit requirement unmet

## Evidence
- `grep -rn "recommended_trades" alloc/` returns **no results** (exit code 1)
- `alloc/core.py` `create_trainer()` docstring lists output keys: `sharpe_ratio, outperformance, final_value, model_roi, buyhold_roi, allocation, model_path, results_path, update`
- `alloc/utils/workflow.py` `TrainingTrial` dataclass has no `recommended_trades` field
- Audit requirement specifies trainer must return `recommended_trades: List[Dict]`

## Impact
- Downstream consumers expecting `recommended_trades` will get `KeyError`.
- `WorkflowRunner` result aggregation does not include trade recommendations.
- CLI output (`print_results`) does not render trade recommendations.

## Suggestion
- Add `recommended_trades` to `TrainingTrial` dataclass in `alloc/utils/workflow.py`
- Compute trade recommendations in `_trainer` closure from `allocation_history` or `final_holdings`
- Extract from `SimulationRunner.run()` results and pass through to trainer output
- Add to `WorkflowResult` metrics progression and CLI rendering

## Implementation Plan
1. Add `recommended_trades: list[dict] | None = None` to `TrainingTrial` in `alloc/utils/workflow.py`
2. In `_trainer` closure, derive `recommended_trades` from `allocation_history[-1]` or `final_holdings`
3. Update `WorkflowRunner._run_trial()` to extract `recommended_trades` from result dict
4. Update `print_results()` in `alloc/cli.py` to render trade recommendations
5. Add unit test asserting `recommended_trades` key presence and schema
