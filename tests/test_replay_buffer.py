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


class TestReplayBufferRng:
    """Tests for TICKET-030: np.random.Generator integration."""

    def test_default_rng_is_generator(self) -> None:
        """Default rng should be a np.random.Generator instance."""
        buf = ReplayBuffer(capacity=10)
        assert isinstance(buf.rng, np.random.Generator)

    def test_custom_rng_is_preserved(self) -> None:
        """A user-provided rng should be stored as-is."""
        custom_rng = np.random.default_rng(seed=42)
        buf = ReplayBuffer(capacity=10, rng=custom_rng)
        assert buf.rng is custom_rng

    def test_reproducible_sampling_with_seed(self) -> None:
        """Two buffers with the same seed should produce identical samples."""
        rng = np.random.default_rng(seed=123)
        buf1 = ReplayBuffer(capacity=100, rng=rng)
        rng = np.random.default_rng(seed=123)
        buf2 = ReplayBuffer(capacity=100, rng=rng)

        for i in range(20):
            buf1.add(
                np.array([float(i)]),
                np.array([float(i)]),
                float(i),
                np.array([float(i + 1)]),
            )
            buf2.add(
                np.array([float(i)]),
                np.array([float(i)]),
                float(i),
                np.array([float(i + 1)]),
            )

        s1, a1, r1, ns1 = buf1.sample(batch_size=5)
        s2, a2, r2, ns2 = buf2.sample(batch_size=5)

        np.testing.assert_array_equal(r1, r2)

    def test_sample_uses_rng_choice_not_np_random(self) -> None:
        """sample() must use self.rng.choice, not np.random.choice."""
        custom_rng = np.random.default_rng(seed=99)
        buf = ReplayBuffer(capacity=10, rng=custom_rng)
        for i in range(5):
            buf.add(
                np.array([float(i)]),
                np.array([float(i)]),
                float(i),
                np.array([float(i + 1)]),
            )
        # Should not raise — self.rng.choice exists and works
        _, _, rewards, _ = buf.sample(batch_size=3)
        assert len(rewards) == 3


class TestReplayBufferDebugLogging:
    """Tests for TICKET-031: counter-based DEBUG logging on overflow."""

    def test_debug_log_when_overwriting(self, caplog) -> None:
        """A DEBUG message is emitted when the counter hits the interval."""
        import logging

        caplog.set_level(logging.DEBUG)
        buf = ReplayBuffer(capacity=3, log_interval=1)
        for i in range(3):
            buf.add(
                np.array([float(i)]),
                np.array([float(i)]),
                float(i),
                np.array([float(i + 1)]),
            )
        # Buffer is now full; next add is the first overwrite
        buf.add(
            np.array([99.0]),
            np.array([99.0]),
            99.0,
            np.array([100.0]),
        )
        assert buf._overwrite_count == 1
        debug_messages = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG and "total overwrites" in r.message
        ]
        assert len(debug_messages) == 1

    def test_counter_increments_on_every_overwrite(self) -> None:
        """The overwrite counter tracks every eviction, not just logged ones."""
        buf = ReplayBuffer(capacity=3, log_interval=1000)
        for i in range(5):
            buf.add(
                np.array([float(i)]),
                np.array([float(i)]),
                float(i),
                np.array([float(i + 1)]),
            )
        # 5 adds into a capacity-3 buffer -> 2 overwrites
        assert buf._overwrite_count == 2

    def test_no_log_below_interval(self, caplog) -> None:
        """No DEBUG overflow log is emitted below the configured interval."""
        import logging

        caplog.set_level(logging.DEBUG)
        buf = ReplayBuffer(capacity=3)  # default log_interval=1000
        for i in range(5):
            buf.add(
                np.array([float(i)]),
                np.array([float(i)]),
                float(i),
                np.array([float(i + 1)]),
            )
        assert buf._overwrite_count == 2
        debug_messages = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG and "total overwrites" in r.message
        ]
        assert len(debug_messages) == 0

    def test_no_debug_log_before_full(self, caplog) -> None:
        """No DEBUG overwrite message when buffer is not yet full."""
        import logging

        caplog.set_level(logging.DEBUG)
        buf = ReplayBuffer(capacity=10, log_interval=1)
        for i in range(5):
            buf.add(
                np.array([float(i)]),
                np.array([float(i)]),
                float(i),
                np.array([float(i + 1)]),
            )
        assert buf._overwrite_count == 0
        debug_messages = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG and "total overwrites" in r.message
        ]
        assert len(debug_messages) == 0

    def test_invalid_log_interval_raises(self) -> None:
        """log_interval < 1 is rejected at construction."""
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=3, log_interval=0)
