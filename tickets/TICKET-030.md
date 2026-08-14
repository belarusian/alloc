# TICKET-030: ReplayBuffer.sample uses legacy np.random.choice() - migrate to np.random.Generator

## Evidence
- alloc/models/networks.py line 110:
  indices = np.random.choice(len(self), size=batch_size, replace=False)
- This uses the legacy module-level np.random API, which is stateless and not reproducible via seed control.
- NumPy 1.17+ introduced np.random.Generator via np.random.default_rng() which provides:
  - Reproducible, isolated RNG state
  - Better statistical quality (PCG64 default)
  - Explicit seeding per-instance
- Other np.random usage in the same file:
  - Line 266: np.random.seed(seed) - legacy seeding in ActorCriticNetworks.__init__
  - Line 485: np.random.normal(...) - legacy call in get_allocation()

## Impact
- Non-reproducible sampling: np.random.choice() draws from a global RNG state. Any other code path that calls np.random.* between two sample() calls changes the distribution. This breaks reproducibility in RL training loops.
- Thread-safety: The legacy np.random module is not thread-safe. Concurrent training or evaluation can produce data races.
- Deprecation trajectory: NumPy recommends migrating to Generator API; the legacy API may be deprecated in future releases.
- Test flakiness: tests/test_replay_buffer.py tests that rely on sampling order may be subtly affected by global RNG state from other tests.

## Suggestion
1. Add self._rng = np.random.default_rng() to ReplayBuffer.__init__.
2. Accept an optional seed parameter in __init__ for external seed control.
3. Replace line 110: indices = self._rng.choice(len(self), size=batch_size, replace=False)
4. Update tests/test_replay_buffer.py to verify deterministic sampling with a fixed seed.

## Implementation Plan
1. Modify ReplayBuffer.__init__ to accept seed: Optional[int] = None and store self._rng = np.random.default_rng(seed)
2. Replace np.random.choice(...) on line 110 with self._rng.choice(...)
3. Add test test_sample_deterministic_with_seed to tests/test_replay_buffer.py:
   - Create two buffers with same seed, same data
   - Assert sample() returns identical indices
4. Run pytest tests/test_replay_buffer.py -xvs to verify no regressions
5. (Out of scope for this ticket but noted) Consider migrating lines 266 and 485 in a follow-up ticket

## Verification
- pytest tests/test_replay_buffer.py -xvs - all existing tests pass
- New deterministic seed test passes
- ruff check alloc/models/networks.py - clean
- mypy alloc/models/networks.py --ignore-missing-imports - clean
