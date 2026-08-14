# TICKET-017: Create alloc/lib/utils.py — Lightweight Utility Functions

**Module:** `alloc/lib/utils.py` (new)
**Source:** `~/Research/new-trader/trader/lib/utils.py` (lines 1–70)
**Priority:** Medium — foundational utilities needed by preprocessing and display code

## Problem

The `alloc/` package has no utility module. Four lightweight functions from the original
trader implementation are needed:

| Function | Source Lines | Purpose |
|---|---|---|
| `ensure_scalar_price` | 10–22 | Convert numpy arrays, lists, tuples → scalar float |
| `format_allocation` | 24–42 | Format ticker:weight dict as human-readable string, sorted by weight desc |
| `create_timestamp_string` | 44–46 | `datetime.now()` formatted for file naming |
| `safe_divide` | 48–60 | Division with zero-denominator protection |

None of these exist in `alloc/` today. `alloc/models/data.py` has an inline `_normalise`
helper (line 112) that duplicates `safe_divide` logic but only for price normalization.

## What to Implement

Create `alloc/lib/utils.py` with typed signatures and modern Python:

1. `ensure_scalar_price(price: Any) -> float` — handles numpy `.item()`, single-element sequences, direct float
2. `format_allocation(allocation: dict[str, float], precision: int = 4) -> str` — sorted desc by weight, percentage format
3. `create_timestamp_string() -> str` — `datetime.now().strftime("%Y%m%d_%H%M%S")`
4. `safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float` — zero protection

## Dependencies

None. Pure utility functions, no external dependencies beyond stdlib.

## Verification

- `pytest tests/test_utils.py -xvs` — all tests pass
- `ruff check alloc/lib/utils.py` — clean
- `mypy alloc/lib/utils.py --ignore-missing-imports` — clean
