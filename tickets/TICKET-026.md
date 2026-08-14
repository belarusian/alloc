# TICKET-026: Create `alloc/__main__.py` — enable `python -m alloc` entry point

**Module:** `alloc/__main__.py` (new)
**Priority:** Medium — canonical CLI invocation path

## Evidence

`ls alloc/__main__.py` fails — no `__main__.py` exists. Currently the only way to invoke the CLI is:
