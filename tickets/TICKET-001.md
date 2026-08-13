# TICKET-001: Package __init__.py files

**Modules:**
- `alloc/__init__.py`
- `alloc/lib/__init__.py`
- `alloc/config/__init__.py`
- `alloc/models/__init__.py`

**Priority:** Blocking — nothing imports without these.

## What to Implement

Four `__init__.py` files to make `alloc`, `alloc.lib`, `alloc.config`, and `alloc.models` importable packages.

### `alloc/__init__.py`
- Project metadata: `__version__ = "0.1.0"`
- Optional: top-level re-exports of public APIs once modules exist
- Keep minimal for Cycle 1

### `alloc/lib/__init__.py`
- Empty or re-exports from `cache` once it exists

### `alloc/config/__init__.py`
- Empty or re-exports from `settings` once it exists

### `alloc/models/__init__.py`
- Empty — placeholder for data models (Cycle 5+)

## Tests

**File:** `tests/test_packages.py`

Verify each package is importable:
- `import alloc` succeeds, `alloc.__version__` is a non-empty string
- `import alloc.lib` succeeds
- `import alloc.config` succeeds
- `import alloc.models` succeeds

## Dependencies

None. This ticket has no dependencies and should be implemented first.

## Improvements Over Seed

The seed has no `__init__.py` files in subdirectories — imports are done with absolute paths from `trader/`. The alloc version uses proper Python packages with `__init__.py` files, enabling `from alloc.lib import cache` style imports.
