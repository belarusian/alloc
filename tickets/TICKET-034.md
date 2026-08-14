# TICKET-034: Add test for `alloc.__main__` module entry point

## What's Wrong

`alloc/__main__.py` has **zero test coverage**. This module enables `python -m alloc` invocation and delegates to `alloc.cli.main()`. If the delegation breaks (import error, wrong function reference, exit code mismatch), the module entry point silently fails for users.

## Evidence

- `alloc/__main__.py` lines 1–17: delegates to `alloc.cli.main()`
- `grep -rn "__main__\|python -m alloc" tests/` → no matches
- `tests/test_cli.py` exists but only tests `alloc.cli` directly, not the `__main__` delegation path
- The `if __name__ == "__main__"` guard on line 16 is never exercised by any test

## Impact

- Users running `python -m alloc` could encounter uncaught import errors or wrong exit codes with no test safety net
- The `__main__` module is the documented entry point in `README.md` ("python -m alloc.core --backtest") — if it breaks, the quick-start guide is broken
- Minor risk but high visibility: this is the first thing a new user tries

## Suggestion

Add to `tests/test_cli.py` (or a new `tests/test_main.py`):
