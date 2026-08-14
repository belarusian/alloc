# TICKET-022: Create tests/test_workflow.py — Workflow orchestration tests

**Module:** `tests/test_workflow.py` (new)
**Priority:** High — tests for TICKET-020 and TICKET-021

## What to Implement

Create `tests/test_workflow.py` with comprehensive tests:

### TrainingConfig tests
- Default values
- Custom values
- Missing positions normalization

### TrainingTrial tests
- Construction with all fields
- Partial metrics (None values)
- Combined score calculation

### WorkflowResult tests
- Construction with trials
- Best trial selection
- Allocation stats aggregation

### WorkflowRunner tests
- Single trial execution
- Multi-trial execution
- Fresh-only mode (update_iterations=0)
- Best trial selection by combined score
- Allocation statistics (mean/std/min/max)
- Concentration metrics (max weight, Herfindahl)
- Centroid recommendation
- Trainer injection / mocking
- Error handling (failed trials)
- Target metrics early termination

## Dependencies

TICKET-020, TICKET-021

## Verification

- `pytest tests/test_workflow.py -xvs` — all tests pass
- `ruff check tests/test_workflow.py` — clean
- `mypy tests/test_workflow.py --ignore-missing-imports` — clean
