# TICKET-028: Create StateBuilder class in alloc/models/data.py

**Module:** `alloc/models/data.py`
**Priority:** High — state construction is needed by the simulation loop

## Problem

The seed's state construction is a procedural function with 12 parameters. We need a `StateBuilder` class with configurable window sizes, cleaner API, no pandas dependency.

## What to Implement

Create `StateBuilder` class in `alloc/models/data.py`:
- `__init__(self, hourly_window: int = 168, daily_window: int = 365, weekly_window: int = 52)`
- `build_state(self, price_data: dict, allocation: list[float]) -> np.ndarray` — shape (1, N)
- `_normalize_window(self, prices: list[float]) -> list[float]` — divide by last price minus 1
- `_pad_window(self, prices: list[float], target_length: int) -> list[float]` — pad if insufficient history

## Dependencies

None

## Verification

- `pytest tests/test_data.py -xvs` — all tests pass
- `ruff check alloc/models/data.py` — clean
- `mypy alloc/models/data.py --ignore-missing-imports` — clean
