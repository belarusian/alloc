# TICKET-013: GitHub Actions CI — Test Infrastructure

**Files:** `.github/workflows/ci.yml`
**Priority:** High — gives visibility into build health, blocks merges on failure.

## What to Implement

Create a GitHub Actions workflow that runs the full gate on every push and pull request.

### Workflow: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10"]

    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest ruff mypy
      - name: Run tests
        run: pytest tests/ -x -q
      - name: Lint
        run: ruff check alloc/
      - name: Type check
        run: mypy alloc/ --ignore-missing-imports
```

### Behavior

- Runs on push to main and on pull requests to main.
- Uses Python 3.10.
- Installs dependencies from `requirements.txt` plus pytest/ruff/mypy.
- Three gate steps: `pytest tests/ -x -q`, `ruff check alloc/`, `mypy alloc/ --ignore-missing-imports`.
- Fails the workflow if any step fails.

### Metrics

The workflow provides:
- Test pass/fail status per commit/PR
- Number of tests run (visible in pytest output)
- Lint violations surfaced in GitHub UI
- Type errors surfaced in GitHub UI

### Dependencies

- `TICKET-005` — pyproject.toml must exist for tool configs
- `TICKET-001` — package structure must be importable

### Tests

No tests. Verification:
- Push a test commit or open a PR branch — workflow appears on GitHub Actions tab and runs to completion.
- Workflow succeeds when gate passes locally.

### Improvements Over Seed

The seed has no CI. Alloc adds automated gate enforcement on every change, preventing broken code from being merged.

### Notes

If `requirements.txt` is minimal, install `python-dotenv` as well for settings tests. Add it to the install step if needed.
