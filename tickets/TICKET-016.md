# TICKET-016: Fix CI Build - Add Missing Dependencies

**Module:** `requirements.txt`, `.github/workflows/ci.yml`
**Test:** `tests/test_actor_critic.py`
**Priority:** Critical — CI is failing

## Problem

GitHub Actions CI fails during test collection:
```
ImportError while importing test module 'tests/test_actor_critic.py'
ModuleNotFoundError: No module named 'tensorflow'
```

The test file imports tensorflow at module level, but tensorflow is not in `requirements.txt`.

## What to Implement

1. **Add tensorflow to requirements.txt** with appropriate version
2. **Update CI workflow** to handle heavy dependencies (optional: use pip cache)
3. **Verify build passes** on PR #23

### Changes

**requirements.txt:**
Add `tensorflow>=2.13.0` to the requirements

**Optional improvements:**
- Add `pip install --upgrade pip && pip install -r requirements.txt --index-url https://pypi.org/simple/` with caching
- Split tests into tensorflow-requiring and non-tensorflow tests

### Tests

- Verify `python -m pytest tests/test_actor_critic.py -x -q` passes locally
- Verify CI builds green

### Dependencies

- tensorflow is heavy (~500MB)
- May increase CI time significantly

### Design Improvements

This unblocks CI for all future PRs. The build must pass before merge.

---

## After This Ticket

CI will pass for all tests. Build safety is restored.
