# TICKET-054: Deeper DDPG coverage — multi-episode critic-loss trend + in-loop Polyak soft-update

- **GitHub issue:** #122
- **Target:** `tests/test_ddpg_integration.py` (add tests)
- **Status:** VERIFIED (issue #122 closed; merged on main, commit 966d8eb, PR #123)
- **Depends on:** TICKET-051 (closed-loop DDPG integration test, VERIFIED)

## Evidence

The DDPG closed-loop integration test (`tests/test_ddpg_integration.py`, 3
tests) landed in Cycle 37 and is verified. Two genuine coverage gaps remain:

1. **No multi-episode critic-loss trend assertion.**
   - `test_closed_loop_training_step` (line 50) runs 16 closed-loop steps but
     performs exactly **one** `update_critic` at the end — it asserts finiteness
     only, never a loss *trend*.
   - `test_training_step_reduces_critic_loss` (line 92) does 20 critic updates
     but on a **fixed random batch** (not closed-loop episodes), so it does not
     model the multi-episode dynamics the DDPG Bellman target assumes.
   - There is no test that runs a *closed-loop, multi-episode* training loop and
     asserts the critic loss trends (later-episode mean < earlier-episode mean).

2. **No in-loop Polyak soft-update coverage.**
   - `_soft_update_targets` (networks.py:463) is the real Polyak update used by
     the production loop (`core.py:440`). `TestSoftUpdateTargets`
     (test_actor_critic.py:421) covers the *formula* in isolation, but no test
     verifies that, **inside a closed-loop training loop**, the target networks
     track the online networks (targets stay close to, but not identical to,
     online weights, and the target critic's Q-estimate stays near the online
     critic's) as training progresses.

## Impact

- **False confidence on learning.** The integration suite proves the loop is
  finite and the buffer overflows, but never that the critic actually *learns*
  over multiple episodes (loss trending down). A regression that silently
  disables learning (e.g. a broken Bellman target) would still pass.
- **Soft-update untested in context.** The Polyak update is the mechanism that
  stabilises DDPG; testing it only in isolation misses interaction bugs with the
  closed-loop dynamics (e.g. targets drifting away from online weights).

## Minimal additive fix

Add two tests to `tests/test_ddpg_integration.py` (no production-code change):

1. `test_multi_episode_critic_loss_trend` — run a deterministic closed-loop,
   multi-episode training loop (several episodes × several steps each, critic
   update per step through the public `get_allocation` API), collect critic
   losses, and assert: all finite, and the mean loss over the later half of
   updates is strictly below the mean over the earlier half (learning is
   happening). Fixed seed for reproducibility.

2. `test_soft_update_targets_track_online_in_loop` — during the same closed-loop
   training loop, after the Polyak soft-updates, assert the target actor/critic
   weights are close to (but not identical to) the online weights, and the
   target critic's Q-estimate on a probe batch is close to the online critic's
   Q-estimate (Polyak keeps targets lagging but near).

## Verification

- New tests pass under `POLYGON_API_KEY=dummy pytest tests/test_ddpg_integration.py -q`.
- Full suite green at the Cycle 39 gate.
