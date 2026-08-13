"""Tests for alloc.models.networks — ActorCriticNetworks."""

from __future__ import annotations

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
