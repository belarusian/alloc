# TICKET-036: Add type annotations to core modules

**Module:** `alloc/core.py`, `alloc/lib/*.py`
**Priority:** Medium — improve type safety

## What to Implement

Run mypy in strict mode and add missing type hints:
1. `alloc/core.py` — SimulationRunner methods
2. `alloc/lib/cache.py` — DiskCache methods
3. `alloc/lib/client.py` — PolygonClient methods

## Verification

- mypy alloc/ --ignore-missing-imports passes
- All tests pass
