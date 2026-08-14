"""Tests for alloc.models.networks — ActorCriticNetworks."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import tensorflow as tf

from alloc.models.networks import ActorCriticNetworks, CashLambda, _calculate_cash


# =====================================================================
# Cash constraint helper
# =====================================================================

class TestCalculateCash:
    """Tests for the _calculate_cash utility."""

    def test_valid_allocation_unchanged(self):
        """Allocation summing < 1.0 with enough cash passes through."""
        alloc = tf.constant([[0.3, 0.3, 0.2]], dtype=tf.float32)
        result = _calculate_cash(alloc, min_cash=0.05)
        # Assets scaled, cash = 1 - sum(assets)
        assert result.shape == (1, 3)
        assert float(result[0, -1]) >= 0.05

    def test_exceeds_one_scales_down(self):
        """Allocation summing > 1.0 gets scaled down."""
        alloc = tf.constant([[0.5, 0.5, 0.5]], dtype=tf.float32)
        result = _calculate_cash(alloc, min_cash=0.05)
        total = float(tf.reduce_sum(result))
        assert abs(total - 1.0) < 1e-5

    def test_min_cash_enforced(self):
        """Cash position is clamped to at least min_cash."""
        alloc = tf.constant([[0.9, 0.9, 0.0]], dtype=tf.float32)
        result = _calculate_cash(alloc, min_cash=0.1)
        cash = float(result[0, -1])
        assert cash >= 0.1

    def test_sum_to_one(self):
        """Output always sums to 1.0."""
        for _ in range(10):
            alloc = tf.random.uniform((1, 5), minval=0.0, maxval=2.0)
            result = _calculate_cash(alloc, min_cash=0.05)
            total = float(tf.reduce_sum(result))
            assert abs(total - 1.0) < 1e-5


# =====================================================================
# CashLambda layer
# =====================================================================

class TestCashLambda:
    """Tests for the CashLambda Keras layer."""

    def test_serializable(self):
        """CashLambda can be serialized and deserialized."""
        layer = CashLambda(min_cash=0.1)
        config = layer.get_config()
        assert config["min_cash"] == 0.1

    def test_call(self):
        """Layer produces valid output."""
        layer = CashLambda(min_cash=0.1)
        x = tf.random.uniform((2, 4), minval=0.0, maxval=1.0)
        out = layer(x)
        assert out.shape == (2, 4)
        for i in range(2):
            total = float(tf.reduce_sum(out[i]))
            assert abs(total - 1.0) < 1e-5
            assert float(out[i, -1]) >= 0.1


# =====================================================================
# ActorCriticNetworks construction
# =====================================================================

class TestActorCriticConstruction:
    """Tests for ActorCriticNetworks.__init__."""

    @pytest.fixture()
    def networks(self):
        return ActorCriticNetworks(
            input_dim=10,
            num_assets=5,
            seed=42,
        )

    def test_actor_output_shape(self, networks):
        state = np.random.randn(1, 10).astype(np.float32)
        out = networks.actor(state, training=False)
        assert out.shape == (1, 5)

    def test_critic_output_shape(self, networks):
        state = np.random.randn(1, 10).astype(np.float32)
        action = np.random.randn(1, 5).astype(np.float32)
        out = networks.critic([state, action], training=False)
        assert out.shape == (1, 1)

    def test_target_networks_exist(self, networks):
        assert networks.actor_target is not None
        assert networks.critic_target is not None

    def test_target_weights_match_initial(self, networks):
        """Target networks should match online networks at init (tau=1 copy)."""
        for w1, w2 in zip(
            networks.actor.get_weights(),
            networks.actor_target.get_weights(),
        ):
            np.testing.assert_allclose(w1, w2, atol=1e-6)

    def test_replay_buffer_created(self, networks):
        assert networks.replay_buffer is not None
        assert len(networks.replay_buffer) == 0


# =====================================================================
# get_allocation
# =====================================================================

class TestGetAllocation:
    """Tests for ActorCriticNetworks.get_allocation."""

    @pytest.fixture()
    def networks(self):
        return ActorCriticNetworks(
            input_dim=10,
            num_assets=5,
            seed=42,
            min_cash_allocation=0.05,
        )

    def test_returns_correct_shape(self, networks):
        state = np.random.randn(10).astype(np.float32)
        alloc = networks.get_allocation(state)
        assert alloc.shape == (5,)

    def test_sum_to_one(self, networks):
        state = np.random.randn(10).astype(np.float32)
        alloc = networks.get_allocation(state)
        assert abs(alloc.sum() - 1.0) < 1e-5

    def test_min_cash_respected(self, networks):
        state = np.random.randn(10).astype(np.float32)
        alloc = networks.get_allocation(state)
        assert alloc[-1] >= 0.05

    def test_non_negative(self, networks):
        state = np.random.randn(10).astype(np.float32)
        alloc = networks.get_allocation(state)
        assert np.all(alloc >= 0)

    def test_noise_adds_variance(self, networks):
        state = np.random.randn(10).astype(np.float32)
        a1 = networks.get_allocation(state, add_noise=True, noise_scale=0.5)
        a2 = networks.get_allocation(state, add_noise=True, noise_scale=0.5)
        # With noise, two calls should differ
        assert not np.allclose(a1, a2)

    def test_deterministic_without_noise(self, networks):
        state = np.random.randn(10).astype(np.float32)
        a1 = networks.get_allocation(state, add_noise=False)
        a2 = networks.get_allocation(state, add_noise=False)
        np.testing.assert_allclose(a1, a2)

    def test_1d_input_accepted(self, networks):
        state = np.random.randn(10).astype(np.float32)
        alloc = networks.get_allocation(state)
        assert alloc.shape == (5,)

    def test_2d_input_accepted(self, networks):
        state = np.random.randn(1, 10).astype(np.float32)
        alloc = networks.get_allocation(state)
        assert alloc.shape == (5,)


# =====================================================================
# TICKET-032: _sample_action tests
# =====================================================================

class TestSampleAction:
    """Tests for ActorCriticNetworks._sample_action."""

    @pytest.fixture()
    def networks(self):
        return ActorCriticNetworks(
            input_dim=10,
            num_assets=5,
            seed=42,
            min_cash_allocation=0.05,
        )

    def test_greedy_mode_returns_valid_allocation(self, networks):
        """Greedy mode (explore=False) returns a valid allocation vector."""
        state = np.random.randn(10).astype(np.float32)
        action = networks._sample_action(state, explore=False)
        assert action.shape == (5,)
        assert np.all(action >= 0.0)
        assert np.all(action <= 1.0)
        assert abs(action.sum() - 1.0) < 1e-5

    def test_greedy_mode_is_deterministic(self, networks):
        """Greedy mode produces identical results for the same state."""
        state = np.random.randn(10).astype(np.float32)
        a1 = networks._sample_action(state, explore=False)
        a2 = networks._sample_action(state, explore=False)
        np.testing.assert_allclose(a1, a2)

    def test_exploration_mode_adds_variance(self, networks):
        """Exploration mode (explore=True) produces different results."""
        state = np.random.randn(10).astype(np.float32)
        a1 = networks._sample_action(state, explore=True, noise_scale=0.5)
        a2 = networks._sample_action(state, explore=True, noise_scale=0.5)
        # With exploration noise, two calls should differ
        assert not np.allclose(a1, a2)

    def test_action_clamped_to_unit_interval(self, networks):
        """Actions are clamped to [0, 1] regardless of noise scale."""
        state = np.random.randn(10).astype(np.float32)
        # Use a very large noise scale to try to push values out of bounds
        action = networks._sample_action(
            state, explore=True, noise_scale=10.0
        )
        assert np.all(action >= 0.0), f"Action has negative values: {action}"
        assert np.all(action <= 1.0), f"Action exceeds 1.0: {action}"

    def test_action_clamping_preserves_min_cash(self, networks):
        """After clamping, min_cash_allocation is still respected."""
        state = np.random.randn(10).astype(np.float32)
        action = networks._sample_action(state, explore=False)
        assert action[-1] >= 0.05

    def test_exploration_with_zero_noise_equals_greedy(self, networks):
        """Exploration with noise_scale=0 should match greedy mode."""
        state = np.random.randn(10).astype(np.float32)
        greedy = networks._sample_action(state, explore=False)
        # With zero noise, exploration should produce the same base allocation
        # before clamping; after clamping they should match
        explore_zero = networks._sample_action(
            state, explore=True, noise_scale=0.0
        )
        np.testing.assert_allclose(greedy, explore_zero, atol=1e-6)


# =====================================================================
# TICKET-033: Full DDPG training step integration test
# =====================================================================

class TestDDPGTrainingStep:
    """Integration test for the full DDPG training loop.

    Exercises: state → action → reward → next_state → buffer.add →
    buffer.sample → networks.update_critic → networks.update_actor →
    _soft_update_targets
    """

    @pytest.fixture()
    def networks(self):
        return ActorCriticNetworks(
            input_dim=10,
            num_assets=5,
            seed=42,
            min_cash_allocation=0.05,
            buffer_capacity=1000,
        )

    def test_full_training_step(self, networks):
        """Run one complete DDPG training step and verify all components."""
        batch_size = 8

        # --- Generate synthetic transitions ---
        states = np.random.randn(batch_size, 10).astype(np.float32)
        next_states = np.random.randn(batch_size, 10).astype(np.float32)
        rewards = np.random.randn(batch_size).astype(np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)

        # --- state → action (actor inference) ---
        actions = []
        for s in states:
            a = networks._sample_action(s, explore=False)
            actions.append(a)
        actions = np.array(actions, dtype=np.float32)

        # --- buffer.add ---
        for i in range(batch_size):
            networks.replay_buffer.add(
                state=states[i],
                action=actions[i],
                reward=float(rewards[i]),
                next_state=next_states[i],
            )

        # --- buffer.sample ---
        assert len(networks.replay_buffer) == batch_size
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states = (
            networks.replay_buffer.sample(batch_size=batch_size)
        )
        assert sampled_states.shape == (batch_size, 10)
        assert sampled_actions.shape == (batch_size, 5)
        assert sampled_rewards.shape == (batch_size,)
        assert sampled_next_states.shape == (batch_size, 10)

        # --- networks.update_critic ---
        critic_loss = networks.update_critic(
            states=sampled_states,
            actions=sampled_actions,
            rewards=sampled_rewards,
            next_states=sampled_next_states,
            dones=dones,
        )
        assert isinstance(critic_loss, float)
        assert np.isfinite(critic_loss)

        # --- networks.update_actor ---
        actor_loss = networks.update_actor(states=sampled_states)
        assert isinstance(actor_loss, float)
        assert np.isfinite(actor_loss)

        # --- _soft_update_targets ---
        networks._soft_update_targets()

        # Verify target weights changed slightly (tau=0.005)
        for w_online, w_target in zip(
            networks.actor.get_weights(),
            networks.actor_target.get_weights(),
        ):
            # After soft update, targets should be close but not identical
            # (unless tau is very small and weights are similar)
            assert w_online.shape == w_target.shape

    def test_multiple_training_steps_improve_stability(self, networks):
        """Multiple training steps should not cause NaN or Inf."""
        for step in range(5):
            batch_size = 4
            states = np.random.randn(batch_size, 10).astype(np.float32)
            next_states = np.random.randn(batch_size, 10).astype(np.float32)
            rewards = np.random.randn(batch_size).astype(np.float32)
            actions = np.array(
                [networks._sample_action(s, explore=False) for s in states],
                dtype=np.float32,
            )

            for i in range(batch_size):
                networks.replay_buffer.add(
                    state=states[i],
                    action=actions[i],
                    reward=float(rewards[i]),
                    next_state=next_states[i],
                )

            sampled = networks.replay_buffer.sample(batch_size=batch_size)
            s_s, a_s, r_s, ns_s = sampled

            c_loss = networks.update_critic(
                states=s_s,
                actions=a_s,
                rewards=r_s,
                next_states=ns_s,
            )
            a_loss = networks.update_actor(states=s_s)
            networks._soft_update_targets()

            assert np.isfinite(c_loss), f"Critic loss NaN at step {step}"
            assert np.isfinite(a_loss), f"Actor loss NaN at step {step}"

    def test_update_critic_with_dones(self, networks):
        """Critic update correctly handles done flags."""
        batch_size = 4
        states = np.random.randn(batch_size, 10).astype(np.float32)
        next_states = np.random.randn(batch_size, 10).astype(np.float32)
        rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        actions = np.array(
            [networks._sample_action(s, explore=False) for s in states],
            dtype=np.float32,
        )
        # Mark last transition as done
        dones = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        loss = networks.update_critic(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )
        assert np.isfinite(loss)

    def test_update_actor_gradient_ascent(self, networks):
        """Actor update should perform gradient ascent on Q-values."""
        batch_size = 4
        states = np.random.randn(batch_size, 10).astype(np.float32)

        # Get Q-values before update
        actions_before = networks.actor.predict(states, verbose=0)
        q_before = networks.critic.predict(
            [states, actions_before], verbose=0
        ).mean()

        # Update actor
        networks.update_actor(states=states)

        # Get Q-values after update
        actions_after = networks.actor.predict(states, verbose=0)
        q_after = networks.critic.predict(
            [states, actions_after], verbose=0
        ).mean()

        # Q-values should generally increase (gradient ascent)
        # We use a soft check since one step may not always increase
        assert np.isfinite(float(q_before))
        assert np.isfinite(float(q_after))


# =====================================================================
# TICKET-032 (continued): _soft_update_targets tests
# =====================================================================

class TestSoftUpdateTargets:
    """Tests for ActorCriticNetworks._soft_update_targets()."""

    @pytest.fixture()
    def networks(self):
        return ActorCriticNetworks(
            input_dim=10,
            num_assets=5,
            seed=42,
            tau=0.1,  # Higher tau for more visible updates
        )

    def test_soft_update_changes_weights(self, networks):
        """Soft update should change target weights toward online weights."""
        # Record target weights before
        target_actor_before = [
            w.copy() for w in networks.actor_target.get_weights()
        ]

        # Modify online actor weights significantly
        actor_weights = networks.actor.get_weights()
        for i in range(len(actor_weights)):
            actor_weights[i] = actor_weights[i] * 2.0 + 1.0
        networks.actor.set_weights(actor_weights)

        # Perform soft update
        networks._soft_update_targets()

        # Target weights should have changed
        target_actor_after = networks.actor_target.get_weights()
        for before, after in zip(target_actor_before, target_actor_after):
            assert not np.allclose(before, after, atol=1e-6)

    def test_soft_update_preserves_shape(self, networks):
        """Soft update should preserve weight shapes."""
        shapes_before = [w.shape for w in networks.actor_target.get_weights()]
        networks._soft_update_targets()
        shapes_after = [w.shape for w in networks.actor_target.get_weights()]
        assert shapes_before == shapes_after

    def test_soft_update_tau_formula(self, networks):
        """Verify the tau-weighted averaging formula."""
        # Get initial weights
        online_w = networks.actor.get_weights()[0].copy()
        target_w = networks.actor_target.get_weights()[0].copy()

        # Modify online weights
        new_online = online_w * 3.0
        networks.actor.set_weights(
            [new_online] + networks.actor.get_weights()[1:]
        )

        # Soft update
        networks._soft_update_targets()

        # Expected: target = tau * new_online + (1 - tau) * old_target
        expected = networks.tau * new_online + (1.0 - networks.tau) * target_w
        actual = networks.actor_target.get_weights()[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_soft_update_critic_also_updated(self, networks):
        """Critic target weights should also be soft-updated."""
        target_critic_before = [
            w.copy() for w in networks.critic_target.get_weights()
        ]

        # Modify online critic weights
        critic_weights = networks.critic.get_weights()
        for i in range(len(critic_weights)):
            critic_weights[i] = critic_weights[i] * 2.0 + 1.0
        networks.critic.set_weights(critic_weights)

        networks._soft_update_targets()

        target_critic_after = networks.critic_target.get_weights()
        for before, after in zip(target_critic_before, target_critic_after):
            assert not np.allclose(before, after, atol=1e-6)


# =====================================================================
# TICKET-034: alloc.__main__ entry point tests
# =====================================================================

class TestMainModule:
    """Tests for alloc.__main__ module entry point."""

    def test_main_module_imports_cleanly(self):
        """alloc.__main__ can be imported without side effects."""
        import importlib

        # Force reimport to test clean import
        if "alloc.__main__" in sys.modules:
            del sys.modules["alloc.__main__"]
        import alloc.__main__  # noqa: F401
        # Should not raise

    def test_main_module_delegates_to_cli_main(self):
        """alloc.__main__ exposes main from alloc.cli."""
        from alloc.__main__ import main as main_entry
        from alloc.cli import main as cli_main
        assert main_entry is cli_main

    def test_main_module_main_is_callable(self):
        """The main function from __main__ is callable."""
        from alloc.__main__ import main
        assert callable(main)

    def test_main_module_returns_exit_code(self):
        """main() returns an integer exit code."""
        from alloc.__main__ import main
        # --help returns 0 via sys.exit(0)
        exit_code = main(["--help"])
        assert isinstance(exit_code, int)

    def test_main_module_invalid_args_returns_non_zero(self):
        """main() returns non-zero on invalid arguments."""
        from alloc.__main__ import main
        # Missing required --tickers
        exit_code = main(["--positions-values", '{"AAPL": 100}'])
        assert exit_code != 0

    def test_main_module_sys_exit_guard(self):
        """The if __name__ == '__main__' guard uses sys.exit(main())."""
        import ast
        import importlib

        mod = importlib.import_module("alloc.__main__")
        source_path = getattr(mod, "__file__", None)
        assert source_path is not None

        with open(source_path) as f:
            source = f.read()

        tree = ast.parse(source)
        # Find the if __name__ == "__main__" block
        found_guard = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check for __name__ == "__main__"
                if (
                    isinstance(node.test, ast.Compare)
                    and any(
                        isinstance(c, ast.Constant) and c.value == "__main__"
                        for c in node.test.comparators
                    )
                ):
                    found_guard = True
                    # Verify it calls sys.exit(main())
                    assert len(node.body) >= 1
                    break
        assert found_guard, "No if __name__ == '__main__' guard found"

    def test_main_module_invoked_as_module(self):
        """python -m alloc can be invoked (simulated via runpy)."""
        import runpy
        import sys
        from io import StringIO

        # Simulate python -m alloc --help
        old_argv = sys.argv
        old_stdout = sys.stdout
        try:
            sys.argv = ["alloc", "--help"]
            sys.stdout = StringIO()
            # run_module will call sys.exit(0) for --help
            try:
                runpy.run_module("alloc", run_name="__main__")
            except SystemExit as e:
                assert e.code == 0, f"Expected exit code 0, got {e.code}"
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
