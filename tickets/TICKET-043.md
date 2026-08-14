# TICKET-043: Wire StateBuilder into training pipeline (replace legacy build_state_vector)

**Module:** `alloc/core.py`, `alloc/models/data.py`
**Priority:** Medium — architectural inconsistency

## Evidence

`alloc/models/data.py` defines two parallel implementations:

1. **`StateBuilder` class** (line ~17) — OOP design with configurable windows, `_normalize_window`, `_pad_window`, and `build_state()` method. Returns 2-D array of shape `(1, N)`.

2. **`build_state_vector()` function** (line ~250) — legacy standalone function. Returns 1-D array of shape `(N,)`.

The training pipeline in `alloc/core.py` uses **only the legacy function**:
- Line 667: `data_pipeline=data_module` — passes the entire module as the pipeline
- Line 272: `self.data_pipeline.build_state_vector(...)` — calls the legacy function
- Line 347: `self.data_pipeline.build_state_vector(...)` — calls the legacy function again

`StateBuilder` is **never instantiated or used** in `core.py`, `workflow.py`, or `cli.py`. It exists only in tests (`tests/test_state_builder.py`).

Additionally, the two implementations differ in behavior:
- `StateBuilder._normalize_window` subtracts 1.0 after dividing by last price → last element is `0.0`
- `build_state_vector()._normalise` divides by last price without subtracting → last element is `1.0`
- `StateBuilder.build_state` returns shape `(1, N)` (2-D)
- `build_state_vector` returns shape `(N,)` (1-D)

## Impact

- **Dead code**: `StateBuilder` is untested in production paths — bugs in it would go undetected.
- **Inconsistent normalization**: If someone switches to `StateBuilder`, the RL agent receives different input distributions (0-centered vs 1-centered), breaking trained models.
- **Shape mismatch**: `StateBuilder` returns `(1, N)` but the pipeline expects `(N,)`. The padding logic at line 281-288 of `core.py` masks this for the 1-D case but would behave differently with 2-D input.
- **Maintenance burden**: Two implementations of the same concept means fixes must be applied twice.

## Suggestion

**Option A (preferred): Retire legacy function, adopt StateBuilder**

1. Update `StateBuilder._normalize_window` to match the legacy normalization (divide by last, no subtract-1), OR update `build_state_vector` to match `StateBuilder` — pick one canonical behavior and document it.
2. Change `core.py` to instantiate `StateBuilder` instead of passing `data_module`:
