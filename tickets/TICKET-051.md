# TICKET-051: Add a true closed-loop DDPG training-step integration test

- **GitHub issue:** #100 (open)
- **Original ticket:** TICKET-033
- **Target:** `tests/` (new file `tests/test_ddpg_integration.py`)
- **Status:** open

## Evidence

Issue #100's body states: "There is **no integration test** that exercises the
complete DDPG training loop." That claim is **stale** — a `TestDDPGTrainingStep`
class already exists at `tests/test_actor_critic.py:252` (added in commit
`987b236`, PR #52), with `test_full_training_step` (line 270) exercising
`_sample_action` → `replay_buffer.add` → `replay_buffer.sample` →
`update_critic` → `update_actor` → `_soft_update_targets`.

However, the existing test is **synthetic, not a real training step**:

1. **Uses the private `_sample_action`** (test_actor_critic.py:283, 342, 376)
   rather than the public inference entry point `get_allocation`
   (networks.py:475). A real training loop calls the public API.
2. **`next_states` are independent random draws** (line 276:
   `np.random.randn(batch_size, 10)`), not derived from the action. The DDPG
   Bellman target in `update_critic` (networks.py:565-627) assumes
   `next_state` is the environment's response to `action`. Independent
   `next_states` do not model the closed-loop dynamics.
3. **The buffer is never filled.** `buffer_capacity=1000` (line 264) but only
   `batch_size=8` transitions are added (line 270). The overflow path
   (networks.py:94-99) is never exercised in the integration test.
4. **No reward model.** Rewards are `np.random.randn` (line 275) with no
   relationship to the allocation action, so the test cannot verify that the
   critic learns anything meaningful — only that it does not produce NaN.


## Impact

- **False confidence.** The existing test passes, so issue #100 appears closed,
  but it does not validate the real training loop: public action sampling,
  closed-loop state transitions, buffer overflow, or reward-driven learning.
- **Overflow path untested end-to-end.** The `ReplayBuffer` overflow logging
  (TICKET-050 / issue #98) and the deque eviction are only covered in isolation
  in `tests/test_replay_buffer.py`, never through the full
  `ActorCriticNetworks` training path.
- **Public API untested in a training context.** `get_allocation` is tested in
  isolation (`TestGetAllocation`, test_actor_critic.py:124) but never as the
  action source feeding a training step.

## Suggestion

Add a dedicated integration test file `tests/test_ddpg_integration.py` that
exercises a **true closed-loop** DDPG training step. Do **not** modify the
existing `TestDDPGTrainingStep` (it remains valid as a synthetic smoke test);
add the new file so both perspectives are covered.

### Implementation plan

1. **New file `tests/test_ddpg_integration.py`** with a small deterministic
   environment closure so `next_state` is a function of `(state, action)`:

       def env_step(state, action):
           # Deterministic toy dynamics: next_state depends on action.
           # e.g. next_state = state * 0.9 + action * 0.1 (broadcast over assets)
           ...
           reward = float(np.dot(action, state))  # reward tied to allocation
           return next_state, reward

2. **`test_closed_loop_training_step`** — the core integration test:
   - Build `ActorCriticNetworks(input_dim=10, num_assets=5, seed=42,
     min_cash_allocation=0.05, buffer_capacity=8)` (small capacity so overflow
     is reachable).
   - Loop over `N=16` steps (N > capacity so the buffer overflows):
     - `action = networks.get_allocation(state)` (public API, greedy).
     - `next_state, reward = env_step(state, action)` (closed-loop).
     - `networks.replay_buffer.add(state, action, reward, next_state)`.
     - `state = next_state`.
   - Assert `len(networks.replay_buffer) == 8` (capped at capacity).
   - `states, actions, rewards, next_states = networks.replay_buffer.sample(8)`.
   - `critic_loss = networks.update_critic(states, actions, rewards, next_states)`.
   - `actor_loss = networks.update_actor(states)`.
   - `networks._soft_update_targets()`.
   - Assert both losses are finite floats.

3. **`test_overflow_exercised_in_training`** — verify the buffer actually
   overwrote during the loop:
   - Same setup; after the loop assert the buffer is full and that the oldest
     transitions were evicted (e.g. rewards reflect the most recent 8 steps,
     not the first 8).
   - Optionally assert `networks.replay_buffer._overwrite_count == N - capacity`
     (depends on TICKET-050 landing; guard with `hasattr`).

4. **`test_training_step_reduces_critic_loss`** (soft check):
   - Run several critic updates on a fixed batch; assert the loss is finite and
     does not explode (e.g. `loss_after < 10 * loss_before`), not a strict
     monotonic decrease (DDPG is stochastic).


## Verification

- `pytest tests/test_ddpg_integration.py -xvs` — all new tests pass.
- `pytest tests/test_actor_critic.py -xvs` — existing `TestDDPGTrainingStep`
  still passes (unchanged).
- `pytest tests/test_replay_buffer.py -xvs` — no regressions.
- `ruff check tests/test_ddpg_integration.py` — clean.
- `mypy tests/test_ddpg_integration.py --ignore-missing-imports` — clean.

## Notes

- **Semantics reference:** DDPG semantics were read from
  `~/Research/new-trader/trader/models/networks.py` for understanding only
  (Bellman target, soft target update, actor gradient ascent on Q). Nothing was
  copied; the test targets the `alloc` implementation's actual public API
  (`get_allocation`, `update_critic`, `update_actor`, `_soft_update_targets`).
- **Dependency on TICKET-050:** `test_overflow_exercised_in_training` references
  `_overwrite_count`, which only exists after TICKET-050 (issue #98) lands. The
  test must guard with `hasattr(networks.replay_buffer, "_overwrite_count")` so
  it passes independently, or be added in the same PR as TICKET-050.
- **Keep the synthetic test:** `TestDDPGTrainingStep` (test_actor_critic.py:252)
  stays as a fast smoke test. The new file adds the closed-loop + overflow
  coverage that issue #100 actually requires.
