# TICKET-031: ReplayBuffer.add should log at DEBUG level when buffer is full and overwriting

## Evidence
- alloc/models/networks.py lines 58-78: ReplayBuffer.add() method:
  def add(self, state, action, reward, next_state) -> None:
      self._buffer.append((state, action, float(reward), next_state))
- The underlying collections.deque(maxlen=capacity) silently evicts the oldest entry when len(self._buffer) == self._capacity and a new item is appended.
- There is NO logging in add() - overwrites are invisible at runtime.
- __init__ logs at INFO level (line 53): logger.info("ReplayBuffer initialised with capacity=%d", capacity) - this is appropriate for initialization.
- The audit requirement: when the buffer is full and an overwrite occurs, log at DEBUG level (not INFO, to avoid log spam during training).

## Impact
- Silent data loss: During RL training, the buffer is typically full after the warmup phase. Every add() call after that evicts the oldest transition, but there is no visibility into this.
- Debugging difficulty: When investigating training instability or reward distribution shifts, operators cannot tell if the buffer is churning transitions at the expected rate.
- No INFO-level spam: Logging at INFO would produce one log line per transition (potentially millions during training). DEBUG is the correct level - visible when logging.DEBUG is enabled, silent in production.

## Suggestion
1. In ReplayBuffer.add(), check if the buffer is already at capacity before appending.
2. Use a counter-based approach to avoid per-transition spam even at DEBUG level:
   - Add self._overwrite_count = 0 to __init__
   - Increment on each overwrite, log every 1000th overwrite at DEBUG level
3. Log format: "ReplayBuffer full (capacity=%d), %d total overwrites"

## Implementation Plan
1. Add self._overwrite_count = 0 to ReplayBuffer.__init__ (after line 52)
2. In ReplayBuffer.add(), before self._buffer.append(...), add:
   if len(self._buffer) == self._capacity:
       self._overwrite_count += 1
       if self._overwrite_count % 1000 == 0:
           logger.debug("ReplayBuffer full (capacity=%d), %d total overwrites", self._capacity, self._overwrite_count)
3. Add test test_add_logs_debug_on_overflow to tests/test_replay_buffer.py:
   - Create buffer with capacity=3
   - Add 5 transitions
   - Assert _overwrite_count == 2
   - Use caplog to verify DEBUG-level log was emitted
4. Run pytest tests/test_replay_buffer.py -xvs to verify no regressions

## Verification
- pytest tests/test_replay_buffer.py -xvs - all tests pass
- New overflow logging test passes with caplog.at_level(logging.DEBUG)
- ruff check alloc/models/networks.py - clean
- mypy alloc/models/networks.py --ignore-missing-imports - clean
