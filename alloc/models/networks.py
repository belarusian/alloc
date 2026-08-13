"""alloc.models.networks — RL infrastructure components.

Provides the ReplayBuffer class, a fixed-capacity circular buffer for
DDPG-style experience replay.  Stores (state, action, reward, next_state)
tuples and returns batched numpy arrays for training.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Fixed-capacity circular buffer for DDPG experience replay.

    Uses :class:`collections.deque` with ``maxlen`` so that oldest
    transitions are automatically evicted when capacity is exceeded.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to retain.

    Example
    -------
    >>> buf = ReplayBuffer(capacity=1_000)
    >>> buf.add(state, action, reward, next_state)
    >>> states, actions, rewards, next_states = buf.sample(batch_size=64)
    """

    def __init__(self, capacity: int) -> None:
        """Initialise the buffer with *capacity* slots."""
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self._buffer: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray]] = deque(
            maxlen=capacity
        )
        self._capacity = capacity
        logger.info("ReplayBuffer initialised with capacity=%d", capacity)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
    ) -> None:
        """Push a single transition onto the buffer.

        Parameters
        ----------
        state : np.ndarray
            Observation at time *t*.
        action : np.ndarray
            Action taken at time *t*.
        reward : float
            Scalar reward received after taking *action*.
        next_state : np.ndarray
            Observation at time *t+1*.
        """
        self._buffer.append((state, action, float(reward), next_state))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a random sample of *batch_size* transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        states : np.ndarray
            Shape ``(batch_size, *state_shape)``.
        actions : np.ndarray
            Shape ``(batch_size, *action_shape)``.
        rewards : np.ndarray
            Shape ``(batch_size,)``.
        next_states : np.ndarray
            Shape ``(batch_size, *state_shape)``.

        Raises
        ------
        ValueError
            If *batch_size* exceeds the current buffer length.
        """
        if batch_size > len(self):
            raise ValueError(
                f"batch_size={batch_size} exceeds buffer length={len(self)}"
            )

        indices = np.random.choice(len(self), size=batch_size, replace=False)

        states = np.stack([self._buffer[i][0] for i in indices])
        actions = np.stack([self._buffer[i][1] for i in indices])
        rewards = np.array([self._buffer[i][2] for i in indices])
        next_states = np.stack([self._buffer[i][3] for i in indices])

        return states, actions, rewards, next_states

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self._buffer)
