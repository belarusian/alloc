# TICKET-5: No end-to-end integration test for WorkflowRunner ↔ SimulationRunner bridge

## Evidence
- `ls tests/` — no `test_alloc_integration.py` or equivalent exists
- `alloc/cli.py` `main()` imports `create_trainer` at runtime with try/except — no test validates this path
- `WorkflowRunner.run()` iterates trials via `_run_trial()` — no test validates full multi-trial workflow
- `SimulationRunner.run()` requires live data from `PolygonClient` — no mock strategy exists

## Impact
- Silent integration failures in CI/CD.
- Regression risk when modifying trainer signature, metric computation, or simulation loop.
- No confidence that `create_trainer()` output matches `TrainingTrial` expectations.

## Suggestion
- Create `tests/test_alloc_integration.py` with mocked data provider and deterministic seeds.
- Test full pipeline: `create_trainer()` → `WorkflowRunner` → `WorkflowResult` schema validation.
- Pin random seeds and use fast/synthetic environment for CI.

## Implementation Plan
1. Add `tests/test_alloc_integration.py`
2. Mock `PolygonClient` and `data_module` with synthetic price data
3. Instantiate `create_trainer()` and verify return type is `Callable`
4. Run `WorkflowRunner` with minimal config and assert `WorkflowResult` schema
5. Assert `TrainingTrial` fields are populated (not all `None`)
6. Add to CI matrix with `pytest -m alloc_integration`
