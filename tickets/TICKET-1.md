# TICKET-1: Trainer signature mismatch — audit expects positional (tickers, trading_days, model_path) but implementation uses **kwargs from TrainingConfig

## Evidence
- `alloc/core.py` line ~380: `create_trainer()` returns `_trainer(**kwargs)` accepting keyword args matching `TrainingConfig` fields (tickers, positions, update_iterations, trading_days, batch_size, etc.)
- `alloc/utils/workflow.py` line ~195: `WorkflowRunner._run_trial()` calls `self.trainer(**kwargs)` with kwargs built from `TrainingConfig`
- Audit requirement specified positional signature: `(tickers, trading_days, model_path)` — this does NOT match the actual implementation

## Impact
- The audit's expected signature is incorrect for the current codebase. The actual contract uses keyword dispatch via `TrainingConfig` fields.
- If downstream consumers expect positional `(tickers, trading_days, model_path)`, they will fail with `TypeError`.
- `model_path` is not a trainer input — it is an output key in the result dict.

## Suggestion
- Update audit documentation to reflect the actual `**kwargs` contract.
- If positional signature is desired, add an adapter layer or redefine `create_trainer()` to accept `(tickers, trading_days, model_path)` and internally construct `TrainingConfig`.
- Document the actual contract in `docs/ALLOC_INTEGRATION.md`.

## Implementation Plan
1. Document actual `**kwargs` contract in `docs/ALLOC_INTEGRATION.md` ✅ (done)
2. If positional signature is required, create `TrainerAdapter` wrapper
3. Add type annotation: `Callable[..., dict[str, Any]]` → explicit `Protocol`
4. Verify `WorkflowRunner` dispatch remains compatible
