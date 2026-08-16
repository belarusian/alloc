# TICKET-047: Issue #101 "alloc.__main__ zero coverage" is stale — consolidate tests

**Status:** OPEN
**Date:** 2025-08-16
**Cycle:** 36
**Priority:** Low
**Issue:** https://github.com/belarusian/alloc/issues/113

## Summary

Issue #101 claims `alloc/__main__.py` has zero test coverage. This is no
longer true: `__main__` is exercised by **two** test classes, and a third
was added this cycle. The issue should be closed and the duplicate coverage
consolidated.

## Evidence

- `tests/test_cli.py::TestMainModule` (line ~803) — 2 tests (import, has main).
- `tests/test_actor_critic.py::TestMainModule` (line ~505, tagged TICKET-034)
  — 7 tests (import, delegation, callable, exit code, invalid args, AST
  guard check, runpy invocation).
- `tests/test_portfolio.py::TestMainEntryPoint` (added this cycle) — 2 tests
  (delegation identity, exit-code propagation).
- All pass: `pytest tests/test_cli.py::TestMainModule
  tests/test_actor_critic.py::TestMainModule tests/test_portfolio.py::TestMainEntryPoint`
  → 11 passed.

## Impact

- Issue #101 is misleading; a newcomer would believe the entry point is
  untested and may add redundant tests.
- Three near-duplicate `TestMainModule`/`TestMainEntryPoint` classes scatter
  the same concerns across three files.

## Suggestion (implementation plan)

1. Close issue #101 with a note that coverage now exists (cite the three
   classes).
2. Consolidate the `__main__` tests into a single `tests/test_main.py`
   (or keep them in `test_cli.py` since `__main__` delegates to `cli.main`),
   removing the duplicates in `test_actor_critic.py` and
   `test_portfolio.py`.
3. Keep the strongest assertions: delegation identity
   (`alloc.__main__.main is alloc.cli.main`), exit-code propagation, and the
   runpy `python -m alloc --help` invocation.

## Acceptance criteria

- Issue #101 closed.
- A single canonical test location for `alloc.__main__`.
- No loss of the delegation / exit-code / runpy assertions.
