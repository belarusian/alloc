"""Tests for alloc.models.networks — ReplayBuffer class."""

from __future__ import annotations

import numpy as np
import pytest

from alloc.models.networks import ReplayBuffer


class TestReplayBufferInit:
    """Tests for ReplayBuffer.__init__."""

    def test_default_init(self) -> None:
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0
        assert buf._capacity == 100

    def test_zero_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity must be > 0"):
            ReplayBuffer(capacity=0)

    def test_negative_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity must be > 0"):
            ReplayBuffer(capacity=-5)


class TestReplayBufferAdd:
    """Tests for ReplayBuffer.add."""

    def test_add_single_transition(self) -> None:
        buf = ReplayBuffer(capacity=10)
        state = np.array([1.0, 2.0])
        action = np.array([0.5])
        reward = 1.0
        next_state = np.array([3.0, 4.0])
        buf.add(state, action, reward, next_state)
        assert len(buf) == 1

    def test_add_multiple_transitions(self) -> None:
        buf = ReplayBuffer(capacity=10)
        for i in range(5):
            buf.add(
                np.array([float(i)]),
                np.array([float(i)]),
                float(i),
                np.array([float(i + 1)]),
            )
        assert len(buf) == 5

    def test_add_respects_capacity(self) -> None:
        buf = ReplayBuffer(capacity=3)
        for i in range(10):
            buf.add(
                np.array([float(i)]),
                np.array([float(i)]),
                float(i),
                np.array([float(i + 1)]),
            )
        assert len(buf) == 3

    def test_oldest_evicted_on_overflow(self) -> None:
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.add(
                np.array([float(i)]),
                np.array([float(i)]),
                float(i),
                np.array([float(i + 1)]),
            )
        # After 5 adds with capacity 3, indices 2, 3, 4 remain
        states, actions, rewards, next_states = buf.sample(batch_size=3)
        stored_rewards = set(rewards.tolist())
        assert stored_rewards == {2.0, 3.0, 4.0}


class TestReplayBufferSample:
    """Tests for ReplayBuffer.sample."""

    def test_sample_returns_correct_shapes(self) -> None:
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            buf.add(
                np.random.randn(4),
                np.random.randn(2),
                float(i),
                np.random.randn(4),
            )
        states, actions, rewards, next_states = buf.sample(batch_size=8)
        assert states.shape == (8, 4)
        assert actions.shape == (8, 2)
        assert rewards.shape == (8,)
        assert next_states.shape == (8, 4)

    def test_sample_batch_size_equals_length(self) -> None:
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.add(
                np.array([float(i)]),
                np.array([float(i)]),
                float(i),
                np.array([float(i + 1)]),
            )
        states, actions, rewards, next_states = buf.sample(batch_size=10)
        assert len(states) == 10

    def test_sample_batch_size_exceeds_length_raises(self) -> None:
        buf = ReplayBuffer(capacity=10)
        buf.add(np.array([1.0]), np.array([0.0]), 0.0, np.array([2.0]))
        with pytest.raises(ValueError, match="exceeds buffer length"):
            buf.sample(batch_size=5)

    def test_sample_empty_buffer_raises(self) -> None:
        buf = ReplayBuffer(capacity=10)
        with pytest.raises(ValueError, match="exceeds buffer length"):
            buf.sample(batch_size=1)

    def test_sample_contains_added_data(self) -> None:
        buf = ReplayBuffer(capacity=100)
        known_rewards = [float(i) for i in range(50)]
        for r in known_rewards:
            buf.add(
                np.array([r]),
                np.array([r * 2]),
                r,
                np.array([r + 1]),
            )
        _, _, rewards, _ = buf.sample(batch_size=50)
        sampled_set = set(rewards.tolist())
        expected_set = set(known_rewards)
        assert sampled_set == expected_set


class TestReplayBufferLen:
    """Tests for ReplayBuffer.__len__."""

    def test_len_empty(self) -> None:
        buf = ReplayBuffer(capacity=10)
        assert len(buf) == 0

    def test_len_after_adds(self) -> None:
        buf = ReplayBuffer(capacity=10)
        for _ in range(5):
            buf.add(np.array([0.0]), np.array([0.0]), 0.0, np.array([0.0]))
        assert len(buf) == 5

    def test_len_capped_at_capacity(self) -> None:
        buf = ReplayBuffer(capacity=5)
        for _ in range(20):
            buf.add(np.array([0.0]), np.array([0.0]), 0.0, np.array([0.0]))
        assert len(buf) == 5
