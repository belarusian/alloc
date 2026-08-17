# TICKET-052: Model persistence round-trip for ActorCriticNetworks

**Status:** OPEN
**Cycle:** 43
**Priority:** High
**Target module:** `alloc/models/networks.py` — `ActorCriticNetworks`

## Summary

Backtest mode (`alloc/core.py` `main`) saves `actor_weights.h5` and
`critic_weights.h5` after training, but there is **no load path**: a saved
model cannot be re-instantiated. The seed's predict mode loads a previously
trained model before producing a live allocation. To close that parity gap
(see TICKET-053), `ActorCriticNetworks` needs a save/load round-trip that
persists both the weights **and** the config needed to reconstruct the
network (input_dim, num_assets, min_cash_allocation, and the hyperparameters
that affect architecture: dropout, gamma, tau, actor_lr, critic_lr).

## Evidence

- `alloc/core.py` lines 734-735: `networks.actor.save_weights(...)` /
  `networks.critic.save_weights(...)` — save only, never load. No config file
  is written, so even the dimensions are not persisted.
- `alloc/models/networks.py` `ActorCriticNetworks.__init__` (lines 285-340)
  builds actor/critic/targets/optimizers from `input_dim`, `num_assets`,
  `min_cash_allocation`, `dropout`. There is no `save_model`/`load_model`
  method and no config file is written.
- Config is stored as instance attributes at lines 304-309:
  `self.input_dim`, `self.num_assets`, `self.gamma`, `self.tau`,
  `self.min_cash_allocation`, `self.dropout`.
- **Gap (refined):** `actor_lr` and `critic_lr` are **not** stored as
  instance attributes. They are only used to construct the optimizers at
  lines 322-323 (`keras.optimizers.Adam(learning_rate=actor_lr)` /
  `...critic_lr`). So `save_model` cannot read them back from the instance
  without either (a) adding `self.actor_lr` / `self.critic_lr` in `__init__`,
  or (b) reading `self.actor_optimizer.learning_rate`. Option (a) is cleaner
  and is required for the round-trip to be lossless.
- The actor architecture depends on `input_dim` and `num_assets` (per-asset
  branch widths scale with the asset index, lines 381-388: `w1 = 32 + i*4`,
  `w2 = 16 + i*2`), and the cash constraint depends on
  `min_cash_allocation` (line 408). Loading weights into a freshly-built
  network with the wrong dimensions will fail or silently mismatch. The
  config must be persisted alongside the weights.

## Implementation plan

1. **`__init__` change:** store `self.actor_lr = actor_lr` and
   `self.critic_lr = critic_lr` (lines ~304-309) so the learning rates are
   part of the persisted config.
2. **`ActorCriticNetworks.save_model(directory)`** — write:
   - `actor_weights.h5`, `critic_weights.h5` (via `keras` `save_weights`).
   - `model_config.json` with `input_dim`, `num_assets`,
     `min_cash_allocation`, `dropout`, `gamma`, `tau`, `actor_lr`,
     `critic_lr`.
3. **`ActorCriticNetworks.load_model(directory)`** (classmethod) — read
   `model_config.json`, construct an `ActorCriticNetworks` with those
   parameters, then `actor.load_weights` / `critic.load_weights`, and
   re-sync the target networks (`actor_target.set_weights`,
   `critic_target.set_weights`). Return the instance.
4. Raise `FileNotFoundError` with a clear message if `model_config.json` or
   either weights file is missing.

## Verification

- `pytest tests/test_actor_critic.py -x -q` — new round-trip test passes:
  build a small network, save to a tmp dir, load, assert
  `get_allocation(state)` is identical (or near-identical) before/after and
  that all config fields (incl. `actor_lr`/`critic_lr`) round-trip.
- `ruff check alloc/models/networks.py` — clean.
- `mypy alloc/models/networks.py --ignore-missing-imports` — clean.
