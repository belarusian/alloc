# TICKET-037: Add edge case validation to CLI and workflow

**Module:** `alloc/cli.py`, `alloc/utils/workflow.py`
**Priority:** Medium — improve robustness

## What to Implement

Add input validation:
1. Empty tickers list → raise ValueError
2. Zero/negative positions → raise ValueError
3. Invalid JSON positions → meaningful error message
4. Empty workflow result → graceful handling

## Verification

- pytest tests/ -x -q passes
- ruff check alloc/ passes
- mypy alloc/ --ignore-missing-imports passes
