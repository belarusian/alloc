# TICKET-050: ReplayBuffer.add() should use counter-based DEBUG logging on overflow

- **GitHub issue:** #98 (open)
- **Original ticket:** TICKET-031
- **Target module:** `alloc/models/networks.py` — `ReplayBuffer`
- **Status:** open

## Evidence

`alloc/models/networks.py` lines 94-99 — `ReplayBuffer.add()` currently logs
**per-overwrite** at DEBUG:

    if len(self._buffer) == self._capacity:
        logger.debug(
            "ReplayBuffer full (capacity=%d), overwriting oldest entry",
            self._capacity,
        )
    self._buffer.append((state, action, float(reward), next_state))

- There is **no `_overwrite_count` counter** anywhere in the class.
- `ReplayBuffer.__init__` (lines 50-72) has no counter field and no throttle
  parameter.
- Issue #98 spec: use a **counter-based** approach — increment a counter on each
  overwrite and log **every 1000th** overwrite at DEBUG. Log format:
  `"ReplayBuffer full (capacity=%d), %d total overwrites"`.

The current implementation only avoids *INFO*-level spam. It does **not** avoid
*DEBUG*-level spam: once the buffer is full, every `add()` emits one DEBUG line.

## Impact

- **DEBUG spam during training.** The default `ActorCriticNetworks` uses
  `buffer_capacity=1_000_000` (networks.py:283). A training run pushes millions
  of transitions; after warmup the buffer is full and every subsequent `add()`
  logs. At DEBUG that is ~1M log lines — the exact spam issue #98 targets.
- **Existing test will break.** `tests/test_replay_buffer.py:213`
  (`test_debug_log_when_overwriting`) asserts the per-overwrite message
  `"overwriting oldest entry"` (line 237). Changing the message text and the
  emission cadence breaks this test. It must be updated in the same change.

## Suggestion

Make the throttle counter-based and configurable so it is testable.

### Implementation plan

1. **`ReplayBuffer.__init__`** (networks.py:50-72):
   - Add a `log_interval: int = 1000` parameter (default 1000 for production).
   - Add `self._overwrite_count: int = 0` and `self._log_interval: int = log_interval`
     (place after line 71, before the `logger.info` at line 72).
   - Validate `log_interval >= 1` (raise `ValueError` otherwise), mirroring the
     existing `capacity <= 0` guard at line 63.

2. **`ReplayBuffer.add()`** (networks.py:74-99): replace the per-overwrite log
   with a counter + throttle:

       if len(self._buffer) == self._capacity:
           self._overwrite_count += 1
           if self._overwrite_count % self._log_interval == 0:
               logger.debug(
                   "ReplayBuffer full (capacity=%d), %d total overwrites",
                   self._capacity,
                   self._overwrite_count,
               )
       self._buffer.append((state, action, float(reward), next_state))

3. **Update `tests/test_replay_buffer.py::TestReplayBufferDebugLogging`**
   (lines 210-251):
   - Rewrite `test_debug_log_when_overwriting` (line 213): construct
     `ReplayBuffer(capacity=3, log_interval=1)`, add 4 transitions (1 overwrite),
     assert `buf._overwrite_count == 1` and that a DEBUG record containing
     `"total overwrites"` was emitted (via `caplog.set_level(logging.DEBUG)`).
   - Add `test_counter_increments_on_every_overwrite`: `capacity=3`, add 5
     transitions, assert `buf._overwrite_count == 2`.
   - Add `test_no_log_below_interval`: `capacity=3`, default `log_interval=1000`,
     add 5 transitions (2 overwrites), assert **no** DEBUG record containing
     `"total overwrites"` was emitted (2 < 1000).
   - Keep `test_no_debug_log_before_full` (line 239) — still valid.

### Spec tension (flagged)

Issue #98's test description ("add 5 transitions, assert `_overwrite_count == 2`,
verify a DEBUG log was emitted") is internally inconsistent with the every-1000th
rule: 2 overwrites < 1000, so no log is emitted. The `log_interval` parameter
resolves this by making the throttle configurable for tests while defaulting to
1000 in production.

## Verification

- `pytest tests/test_replay_buffer.py -xvs` — all pass, including the 3 updated/new
  overflow-logging tests.
- `ruff check alloc/models/networks.py` — clean.
- `mypy alloc/models/networks.py --ignore-missing-imports` — clean.
