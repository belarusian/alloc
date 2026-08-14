# TICKET-039: Fix _sync_to_ghpages git workflow — uses invalid `git cp` command

**Module:** `alloc/lib/publish_dashboard.py`
**Priority:** High — sync feature is broken
**Cycle:** 21

## What's Wrong

The `_sync_to_ghpages()` function at line ~260 uses `git cp` which is **not a
valid git command**. The correct approach is to use Python's `shutil.copy2` or
`subprocess.run(["cp", ...])`.

## Evidence

`alloc/lib/publish_dashboard.py:268`:
