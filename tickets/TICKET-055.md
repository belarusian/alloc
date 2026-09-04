# TICKET-055: Deeper DDPG coverage — actor Q-improvement trend (update_actor)

- **GitHub issue:** #124
- **Target:** `tests/test_ddpg_integration.py` (add test)
- **Status:** OPEN
- **Depends on:** TICKET-051 (closed-loop DDPG integration test, VERIFIED)

## Evidence

The DDPG integration suite (`tests/test_ddpg_integration.py`, 5 tests) covers
the critic-loss trend (TICKET-054) and the in-loop Polyak soft-update, but the
**actor** side is only ever checked for *finiteness*:

1. **`test_closed_loop_training_step` (line 51)** calls `update_actor(states)`
   once and asserts only `math.isfinite(actor_loss)` (line 71).
2. **`test_update_actor_gradient_ascent` (test_actor_critic.py:390)** computes
   `q_before` and `q_after` around a single `update_actor` call, but the
   assertions are only `np.isfinite(q_before)` / `np.isfinite(q_after)` —
   the docstring says "Q-values should generally increase (gradient ascent)"
   yet the increase is **never asserted**. The comment even notes "We use a
   soft check since one step may not always increase", so the trend is
   deliberately left unverified.

There is no test that runs repeated `update_actor` steps and asserts the
actor's Q-value (the quantity the actor is trained to maximise) actually
improves — i.e. that the actor is performing gradient ascent on Q.

## Impact

- **False confidence on actor learning.** The suite proves the actor update is
  finite, but never that the actor actually *improves* (Q rising / actor loss
  falling). A regression that silently breaks the actor's gradient (e.g. a
  sign flip, a detached tape, or a zeroed actor learning rate) would still
  pass every existing test.
- **Asymmetric coverage.** The critic-learning trend is asserted (TICKET-054);
  the actor-learning trend is not. DDPG is a two-network algorithm and both
  halves should be witnessed learning.

## Verified empirically (prototype, seed=42 fixture, 50 update_actor steps)

- Mean Q rises from `+0.12691` to `+0.13858` (delta `+0.01166`), identical
  across 5 runs (deterministic under the fixed seed).
- Robust across seeds 0/1/7/42 (delta `+0.0017` to `+0.008`), all positive.
- `actor_loss` (= -mean Q) falls correspondingly (`-0.12788` -> `-0.13584`).

## Minimal additive fix

Add one integration test to `tests/test_ddpg_integration.py`:
run repeated `update_actor` steps on a fixed batch and assert the actor's
mean Q-value (measured via `actor.predict` -> `critic.predict`) strictly
improves (later > earlier) and the actor loss decreases. Use a margin that
the prototype shows is robust (strict `>` on the measured Q, plus a bounded
loss-decrease check), so the assertion is meaningful but not flaky.
