# TICKET-023: Create `alloc.core.create_trainer` — bridge from WorkflowRunner to SimulationRunner

**Module:** `alloc/core.py`
**Priority:** High — blocks `alloc/cli.py` from functioning end-to-end

## Evidence

`alloc/cli.py:478` imports `from alloc.core import create_trainer`, but `grep -n "create_trainer" alloc/core.py` returns nothing. The `WorkflowRunner._run_trial` method (line 247 of `alloc/utils/workflow.py`) calls `self.trainer(**kwargs)` with this signature:
