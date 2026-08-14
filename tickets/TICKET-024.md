# TICKET-024: Create `tests/test_cli.py` — CLI entry point tests

**Module:** `tests/test_cli.py` (new)
**Priority:** High — no test coverage for the CLI entry point

## Evidence

`ls tests/test_cli.py` fails — no test file exists for `alloc/cli.py`. The CLI module has 550 lines with:
- 6 type converter functions (`_positive_int`, `_non_negative_int`, `_non_negative_float`, `_positive_float`, `_fraction`, `_json_string`)
- `build_parser()`, `parse_args()`, `build_config()`, `print_results()`, `main()`
- Exit code constants and argparse integration

None of these are tested.

## Impact

- Type converter edge cases (negative numbers, invalid JSON, out-of-range fractions) are untested
- `parse_args` post-processing (ticker splitting, fresh-only mode) is untested
- `build_config` mapping from Namespace → TrainingConfig is untested
- `main()` exit codes (0/1/2/3) are untested
- Regression risk: any refactor of CLI args silently breaks the interface

## Suggestion

Create `tests/test_cli.py` with:

### Type converter tests
- `_positive_int`: valid, zero, negative, non-numeric
- `_non_negative_int`: valid, negative, non-numeric
- `_non_negative_float`: valid, negative, non-numeric
- `_positive_float`: valid, zero, negative, non-numeric
- `_fraction`: valid [0,1], below 0, above 1, non-numeric
- `_json_string`: valid dict, non-dict JSON, invalid JSON, non-string keys, non-numeric values

### Parser tests
- `build_parser()` returns ArgumentParser
- `parse_args()` with minimal required args
- `parse_args()` with all args
- `parse_args()` ticker splitting and uppercasing
- `parse_args()` fresh-only sets update_iterations=0
- `parse_args()` empty tickers raises error

### Config tests
- `build_config()` maps all fields correctly
- `build_config()` uses ticker_list not raw tickers string

### Main tests
- `main()` returns 0 on success (mock trainer)
- `main()` returns 1 on bad args
- `main()` returns 2 on workflow failure
- `main()` returns 3 on unexpected exception

## Verification

- `pytest tests/test_cli.py -xvs` — all tests pass
- `ruff check tests/test_cli.py` — clean
- `mypy tests/test_cli.py --ignore-missing-imports` — clean
