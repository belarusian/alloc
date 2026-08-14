# TICKET-035: Create README.md

**Module:** `README.md` (new)
**Priority:** High — project documentation is essential for onboarding and usage

## What to Implement

Create `README.md` with:
1. Project overview and purpose
2. Installation instructions (pip install -e .)
3. Usage examples (python -m alloc --help, CLI examples)
4. Architecture diagram (modules and data flow)
5. Testing instructions (pytest, ruff, mypy)
6. License and contribution guidelines

## Verification

- README.md exists and is well-formatted
- Links work, examples are accurate
- ruff check alloc/ still passes
- mypy alloc/ --ignore-missing-imports still passes
