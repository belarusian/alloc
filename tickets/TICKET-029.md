# TICKET-029: Create tests/test_state_builder.py

**Module:** `tests/test_state_builder.py` (new)
**Priority:** High — tests for StateBuilder class

## What to Implement

Create `tests/test_state_builder.py` with:
- StateBuilder construction with default/custom windows
- build_state with complete data
- build_state with insufficient history (padding)
- normalize_window correctness
- pad_window correctness
- State shape verification (1, N)

## Dependencies

TICKET-028

## Verification

- `pytest tests/test_state_builder.py -xvs` — all tests pass
- `ruff check tests/test_state_builder.py` — clean
- `mypy tests/test_state_builder.py --ignore-missing-imports` — clean
