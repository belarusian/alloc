# TICKET-005: Gate Configuration — pyproject.toml or pytest.ini

**Files:** `pyproject.toml` (or `pytest.ini` + `ruff.toml`)
**Priority:** Medium — needed for the gate to run.

## What to Implement

Ensure the project has tool configuration so the gate commands work:

### `pyproject.toml`

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-x -q"

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I"]

[tool.mypy]
ignore_missing_imports = true
python_version = "3.10"
```

## Tests

No tests. Verify manually that the three gate commands succeed:
- `pytest tests/ -x -q`
- `ruff check alloc/`
- `mypy alloc/ --ignore-missing-imports`

## Dependencies

- `TICKET-001` — needs the package structure to exist for ruff/mypy to scan

## Implementation Notes

If `pyproject.toml` already exists, merge these settings into it. Do not overwrite existing content.
