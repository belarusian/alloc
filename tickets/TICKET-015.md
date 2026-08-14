# TICKET-015: Project Health Dashboard — Cycle Signals

**Module:** `alloc/lib/dashboard.py`, `alloc/lib/cycle_signals.py`
**Test:** `tests/test_dashboard.py`
**Priority:** High — need visibility into project health, test coverage, code smells.

## What to Implement

Build a project health dashboard similar to autonomous-project's cycle_signals system. This provides observability into the alloc codebase state.

### Components

1. **Codebase Crawler** (`alloc/lib/dashboard.py`)
   - Crawl `alloc/` directory for all .py files
   - Extract: file path, lines of code, function count, class count, test count
   - Detect signals:
     - S1: no tests (module has 0 test functions)
     - S2: oversized module (>200 lines AND >15 functions)
     - S3: dead code candidates (modules with 0 incoming imports)
     - S4: error hotspots (modules with ruff/mypy errors)
   - Generate JSON metadata: `alloc/docs_dashboard_metadata.json`

2. **Cycle Signals** (`alloc/lib/cycle_signals.py`)
   - Read JSON metadata
   - Output tree view with signals
   - Format: tree, json, text
   - Similar to autonomous-project's cycle_signals

3. **Dashboard Publisher** (`alloc/lib/publish_dashboard.py`)
   - Generate HTML dashboard from JSON metadata
   - Option to sync to GitHub Pages
   - Validate HTML ↔ JSON sync

### Signals to Track

- S1: no tests
- S2: oversized module (>200 lines, >15 functions)
- S3: dead code (0 imports)
- S4: lint/type errors
- S5: coverage trend (compare to previous cycle)

### Output Format

Tree view:
```
alloc: 10m 5000L 250f [S1,S2]
  └ models: 5m 2500L 120f [S2]
  └ lib: 3m 1500L 80f [S1]
  └ [2 clean: 2m 1000L 50f]
```

JSON metadata includes module stats, signals, dependency graph.

### Tests

- `test_dashboard_crawler`: crawl produces correct metadata
- `test_cycle_signals_tree`: tree output format correct
- `test_signals_detection`: S1/S2/S3/S4 correctly detected
- `test_publish_dashboard`: HTML generation works

### Dependencies

- Requires existing codebase to crawl
- Uses ruff/mypy output for error counting

### Design Improvements

- Provides observability for the cycle loop
- Enables data-driven audit decisions
- Prevents drift by making project health visible

---

## After This Ticket

We can run:
```bash
python -m alloc.lib.cycle_signals alloc/docs_dashboard_metadata.json --format tree --depth 2
```

And get a real-time view of project health signals. This becomes the Scan phase observability for alloc cycles.
