# TICKET-4: No type safety between WorkflowRunner trainer dispatch and create_trainer output — missing Protocol/TypedDict

## Evidence
- `alloc/utils/workflow.py` line ~150: `trainer: Callable[..., dict]` — no type constraints on input kwargs or output dict keys
- `alloc/core.py` line ~380: `create_trainer() -> Callable[..., dict[str, Any]]` — return type is unstructured dict
- `alloc/utils/workflow.py` `_run_trial()` uses `.get()` with string keys — no compile-time validation
- No `Protocol`, `TypedDict`, or `dataclass` bridges the trainer contract

## Impact
- Typos in key names (e.g., `sharpe_ration` vs `sharpe_ratio`) silently produce `None` values.
- Missing keys are silently handled by `.get()` defaults, masking integration bugs.
- Refactoring trainer output breaks `TrainingTrial` construction without tooling warnings.

## Suggestion
- Define `TrainerOutput = TypedDict` with all required/optional keys
- Define `TrainerProtocol = Protocol` with `__call__(**kwargs) -> TrainerOutput`
- Annotate `WorkflowRunner.trainer: TrainerProtocol`
- Annotate `create_trainer() -> TrainerProtocol`

## Implementation Plan
1. Add `alloc/types.py` with `TrainerOutput` TypedDict and `TrainerProtocol`
2. Update `WorkflowRunner.__init__` to accept `trainer: TrainerProtocol`
3. Update `create_trainer()` return annotation to `TrainerProtocol`
4. Replace `.get()` calls with direct key access (validated by type checker)
5. Run `mypy alloc/` to verify type safety
