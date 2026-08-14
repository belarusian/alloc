# TICKET-038: Wire publish_dashboard into alloc CLI

**Module:** `alloc/cli.py`, `alloc/lib/publish_dashboard.py`
**Priority:** Medium — completes the dashboard pipeline
**Cycle:** 21

## What's Wrong

`alloc.lib.publish_dashboard` is implemented and tested but has no entry point
in the main CLI (`alloc/cli.py`). Users must invoke it via
`python -m alloc.lib.publish_dashboard` rather than through the unified
`alloc` command.

## Evidence

- `alloc/cli.py` — no subcommand for dashboard publishing
- `alloc/lib/publish_dashboard.py:310` — `main()` exists but is standalone
- `alloc/__main__.py` — delegates to `alloc.cli.main` only

## Impact

Users cannot generate HTML dashboards through the primary CLI interface.
The dashboard workflow is fragmented: JSON via `alloc`, HTML via a separate
module invocation.

## Suggestion

Add a `--publish-dashboard` flag (or subcommand) to `alloc/cli.py`:
