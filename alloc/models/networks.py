"""alloc.models.networks — RL infrastructure components.

Provides :class:`ReplayBuffer` (a fixed-capacity circular buffer for
DDPG-style experience replay) and :class:`ActorCriticNetworks` (DDPG
actor-critic pair for portfolio allocation).
"""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

logger = logging.getLogger(__name__)


# ===================================================================
# Replay Buffer
# ===================================================================


class ReplayBuffer:
    """Fixed-capacity circular buffer for DDPG experience replay.

    Uses :class:`collections.deque` with ``maxlen`` so that oldest
    transitions are automatically evicted when capacity is exceeded.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to retain.
    rng : np.random.Generator, optional
        NumPy random generator for reproducible sampling.
        Defaults to ``np.random.default_rng()``.

    Example
    -------
    >>> buf = ReplayBuffer(capacity=1_000)
    >>> buf.add(state, action, reward, next_state)
    >>> states, actions, rewards, next_states = buf.sample(batch_size=64)
    """

    def __init__(
        self, capacity: int, rng: np.random.Generator | None = None
    ) -> None:
        """Initialise the buffer with *capacity* slots.

        Parameters
        ----------
        capacity : int
            Maximum number of transitions to retain.
        rng : np.random.Generator, optional
            NumPy random generator for reproducible sampling.
            Defaults to ``np.random.default_rng()``.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self._buffer: deque[
            tuple[np.ndarray, np.ndarray, float, np.ndarray]
        ] = deque(maxlen=capacity)
        self._capacity = capacity
        self.rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )
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
        if len(self._buffer) == self._capacity:
            logger.debug(
                "ReplayBuffer full (capacity=%d), overwriting oldest entry",
                self._capacity,
            )
        self._buffer.append((state, action, float(reward), next_state))

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        indices = self.rng.choice(len(self), size=batch_size, replace=False)

        states = np.stack([self._buffer[i][0] for i in indices])
        actions = np.stack([self._buffer[i][1] for i in indices])
        rewards = np.array([self._buffer[i][2] for i in indices])
        next_states = np.stack([self._buffer[i][3] for i in indices])

        return states, actions, rewards, next_states

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self._buffer)


# ===================================================================
# Cash-constraint helpers
# ===================================================================


def _calculate_cash(allocations: tf.Tensor, min_cash: float) -> tf.Tensor:
    """Enforce that portfolio allocations sum to 1.0 with a minimum cash floor.

    Parameters
    ----------
    allocations : tf.Tensor
        Per-asset allocations (already sigmoided), shape
        ``(batch, num_assets)``.
    min_cash : float
        Minimum fraction reserved for cash.

    Returns
    -------
    tf.Tensor
        Adjusted allocations where the last dimension is set to the
        residual cash position, guaranteed >= *min_cash*.
    """
    num_assets = tf.shape(allocations)[-1]
    asset_cols_raw = allocations[:, : num_assets - 1]
    used = tf.reduce_sum(asset_cols_raw, axis=-1, keepdims=True)
    cash = 1.0 - used
    cash_clamped = tf.maximum(cash, min_cash)
    scale = tf.where(
        cash < min_cash,
        (1.0 - min_cash) / tf.maximum(used, 1e-8),
        1.0,
    )
    adjusted_assets = asset_cols_raw * scale
    return tf.concat([adjusted_assets, cash_clamped], axis=-1)


class CashLayer(layers.Layer):
    """Keras layer that enforces the cash-constraint on portfolio allocations.

    Ensures the output vector sums to 1.0 with a configurable minimum
    cash floor.  The last element of the input is treated as the cash
    position and is replaced by the residual after scaling.

    Parameters
    ----------
    min_cash : float
        Minimum fraction of the portfolio that must remain in cash.
    **kwargs : dict
        Passed to :class:`keras.layers.Layer`.
    """

    def __init__(self, min_cash: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.min_cash = float(min_cash)

    def call(self, allocations: tf.Tensor) -> tf.Tensor:
        return _calculate_cash(allocations, self.min_cash)

    def get_config(self):
        config = super().get_config()
        config.update({"min_cash": self.min_cash})
        return config


@register_keras_serializable()
class CashLambda(layers.Lambda):
    """Lambda wrapper around :func:`_calculate_cash` for model serialization.

    Parameters
    ----------
    min_cash : float
        Minimum cash fraction.
    **kwargs : dict
        Passed to :class:`keras.layers.Lambda`.
    """

    def __init__(self, min_cash: float = 0.0, **kwargs):
        self._min_cash = float(min_cash)
        super().__init__(
            function=lambda x: _calculate_cash(x, self._min_cash),
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"min_cash": self._min_cash})
        return config


# ===================================================================
# Actor-Critic Networks
# ===================================================================


class ActorCriticNetworks:
    """DDPG actor-critic pair for portfolio allocation.

    The **actor** maps market state to a portfolio allocation vector
    (per-asset weights + cash).  The **critic** estimates the Q-value
    of a (state, action) pair.  Soft-target networks are maintained for
    stable learning.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the state observation vector.
    num_assets : int
        Number of tradeable assets (cash is implicit as the residual).
    actor_lr : float
        Learning rate for the actor optimizer.
    critic_lr : float
        Learning rate for the critic optimizer.
    dropout : float
        Dropout rate applied in both actor and critic.
    gamma : float
        Discount factor for future rewards.
    tau : float
        Soft-update coefficient for target networks (0 < tau <= 1).
    min_cash_allocation : float
        Minimum portfolio fraction reserved for cash.
    buffer_capacity : int
        Maximum size of the replay buffer.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        input_dim: int,
        num_assets: int,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        dropout: float = 0.1,
        gamma: float = 0.99,
        tau: float = 0.005,
        min_cash_allocation: float = 0.0,
        buffer_capacity: int = 1_000_000,
        seed: Optional[int] = None,
    ) -> None:
        # --- seeding ---
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        self.input_dim = input_dim
        self.num_assets = num_assets
        self.gamma = gamma
        self.tau = tau
        self.min_cash_allocation = min_cash_allocation
        self.dropout = dropout

        # --- networks ---
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_target = self._build_actor()
        self.critic_target = self._build_critic()

        # Hard-copy targets at init so they match online networks exactly
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        # --- optimizers ---
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=critic_lr)

        # --- replay buffer ---
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        logger.info(
            "ActorCriticNetworks initialised: input_dim=%d, num_assets=%d, "
            "gamma=%.3f, tau=%.4f, min_cash=%.3f",
            input_dim,
            num_assets,
            gamma,
            tau,
            min_cash_allocation,
        )

    # ------------------------------------------------------------------
    # Actor
    # ------------------------------------------------------------------

    def _build_actor(self) -> keras.Model:
        """Build the actor network.

        Architecture
        ------------
        Input(input_dim)
          -> Dense(128, relu) -> Dropout
          -> Dense(128, relu) -> Dropout
          -> Dense(64, relu)  -> Dropout
          -> Dense(64, relu)  [shared representation]
          -> per-asset branches:
               Dense(32 + i*4, relu) -> Dense(16 + i*2, relu)
               -> GaussianNoise(0.1) -> Dropout
          -> Concatenate [cross-asset interaction]
          -> Dense(64, relu) [global mixing]
          -> per-asset Dense(1, sigmoid)
          -> Concatenate
          -> CashLambda(min_cash) [enforce cash constraint]

        Returns
        -------
        keras.Model
            Actor model mapping state -> allocation vector.
        """
        state_input = layers.Input(
            shape=(self.input_dim,), name="state_input"
        )

        # Shared backbone
        x = layers.Dense(128, activation="relu")(state_input)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        shared = layers.Dense(64, activation="relu")(x)

        # Per-asset branches with varying widths
        branches = []
        for i in range(self.num_assets):
            w1 = 32 + i * 4
            w2 = 16 + i * 2
            b = layers.Dense(w1, activation="relu")(shared)
            b = layers.Dense(w2, activation="relu")(b)
            b = layers.GaussianNoise(0.1)(b)
            b = layers.Dropout(self.dropout)(b)
            branches.append(b)

        # Cross-asset concatenation
        cross = layers.Concatenate(name="cross_asset")(branches)
        global_mix = layers.Dense(64, activation="relu")(cross)

        # Per-asset sigmoid outputs
        asset_outputs = []
        for i in range(self.num_assets):
            a = layers.Dense(
                1, activation="sigmoid", name=f"asset_{i}_raw"
            )(global_mix)
            asset_outputs.append(a)

        raw_allocations = layers.Concatenate(name="raw_allocations")(
            asset_outputs
        )

        # Cash constraint layer
        final = CashLambda(
            min_cash=self.min_cash_allocation, name="cash_constraint"
        )(raw_allocations)

        model = keras.Model(inputs=state_input, outputs=final, name="actor")
        return model

    # ------------------------------------------------------------------
    # Critic
    # ------------------------------------------------------------------

    def _build_critic(self) -> keras.Model:
        """Build the critic network.

        Architecture
        ------------
        [state_input, action_input]
          -> Concatenate
          -> Dense(128, relu) -> Dropout
          -> Dense(64, relu)  -> Dropout
          -> Dense(1)

        Returns
        -------
        keras.Model
            Critic model mapping (state, action) -> Q-value.
        """
        state_input = layers.Input(
            shape=(self.input_dim,), name="critic_state_input"
        )
        action_input = layers.Input(
            shape=(self.num_assets,), name="critic_action_input"
        )

        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        q_value = layers.Dense(1, name="q_value")(x)

        model = keras.Model(
            inputs=[state_input, action_input],
            outputs=q_value,
            name="critic",
        )
        return model

    # ------------------------------------------------------------------
    # Target network updates
    # ------------------------------------------------------------------

    def _soft_update_targets(self) -> None:
        """Soft-update target networks toward online networks.

        ``target = tau * online + (1 - tau) * target``
        """
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()

        target_actor_weights = self.actor_target.get_weights()
        target_critic_weights = self.critic_target.get_weights()

        for i in range(len(actor_weights)):
            target_actor_weights[i] = (
                self.tau * actor_weights[i]
                + (1.0 - self.tau) * target_actor_weights[i]
            )
        for i in range(len(critic_weights)):
            target_critic_weights[i] = (
                self.tau * critic_weights[i]
                + (1.0 - self.tau) * target_critic_weights[i]
            )

        self.actor_target.set_weights(target_actor_weights)
        self.critic_target.set_weights(target_critic_weights)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_allocation(
        self,
        state: np.ndarray,
        add_noise: bool = False,
        noise_scale: float = 0.1,
    ) -> np.ndarray:
        """Compute a portfolio allocation for the given market state.

        Parameters
        ----------
        state : np.ndarray
            Market observation, shape ``(input_dim,)`` or
            ``(1, input_dim)``.
        add_noise : bool
            If ``True``, add exploration noise (scaled by *noise_scale*)
            to encourage exploration.
        noise_scale : float
            Standard deviation of the exploration noise.

        Returns
        -------
        np.ndarray
            Allocation vector of length *num_assets*, summing to 1.0
            with the last element as the cash position
            (>= *min_cash_allocation*).
        """
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        allocation = self.actor.predict(state, verbose=0)[0]

        if add_noise:
            noise = np.random.normal(
                0.0, noise_scale, size=self.num_assets
            )
            allocation = allocation + noise

        # Re-normalise to ensure valid probabilities
        allocation = np.maximum(allocation, 0.0)
        # Enforce min cash on the last element
        allocation[-1] = max(allocation[-1], self.min_cash_allocation)
        # Re-normalise so the vector sums to 1
        total = allocation.sum()
        if total > 0:
            allocation = allocation / total
        else:
            # Fallback: uniform with min cash
            allocation = np.ones(self.num_assets) / self.num_assets
            allocation[-1] = max(
                allocation[-1], self.min_cash_allocation
            )
            allocation = allocation / allocation.sum()

        return allocation

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    def _sample_action(
        self,
        state: np.ndarray,
        explore: bool = False,
        noise_scale: float = 0.1,
    ) -> np.ndarray:
        """Sample an action from the actor, optionally with exploration noise.

        Parameters
        ----------
        state : np.ndarray
            Market observation, shape ``(input_dim,)`` or ``(1, input_dim)``.
        explore : bool
            If ``True``, add Gaussian noise for exploration.
        noise_scale : float
            Standard deviation of the exploration noise.

        Returns
        -------
        np.ndarray
            Clamped allocation vector of length *num_assets*.
        """
        action = self.get_allocation(state, add_noise=explore, noise_scale=noise_scale)
        # Clamp to [0, 1]
        action = np.clip(action, 0.0, 1.0)
        return action

    # ------------------------------------------------------------------
    # Training step methods
    # ------------------------------------------------------------------

    def update_critic(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray | None = None,
    ) -> float:
        """Perform one gradient step on the critic network.

        Minimises the MSE between the critic's Q-value estimate and the
        Bellman target computed from the target actor/critic pair.

        Parameters
        ----------
        states : np.ndarray
            Batch of states, shape ``(batch, input_dim)``.
        actions : np.ndarray
            Batch of actions, shape ``(batch, num_assets)``.
        rewards : np.ndarray
            Batch of rewards, shape ``(batch,)``.
        next_states : np.ndarray
            Batch of next states, shape ``(batch, input_dim)``.
        dones : np.ndarray, optional
            Batch of done flags, shape ``(batch,)``.  If ``None``,
            assumes no episode terminated.

        Returns
        -------
        float
            The critic loss (MSE) after the update.
        """
        if dones is None:
            dones = np.zeros_like(rewards)

        # Convert to tensorflow tensors for gradient tape compatibility
        s = tf.convert_to_tensor(states, dtype=tf.float32)
        a = tf.convert_to_tensor(actions, dtype=tf.float32)
        r = tf.convert_to_tensor(rewards, dtype=tf.float32)
        ns = tf.convert_to_tensor(next_states, dtype=tf.float32)
        d = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # Target Q-values from target networks
            next_actions = self.actor_target(ns, training=False)
            target_q = tf.squeeze(self.critic_target(
                [ns, next_actions], training=False
            ), axis=-1)

            # Bellman target
            targets = r + (1.0 - d) * self.gamma * target_q

            # Current Q-values
            current_q = tf.squeeze(self.critic([s, a], training=True), axis=-1)

            critic_loss = tf.reduce_mean(tf.square(current_q - targets))

        grads = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(
            zip(grads, self.critic.trainable_weights)
        )

        return float(critic_loss)

    def update_actor(
        self,
        states: np.ndarray,
    ) -> float:
        """Perform one gradient step on the actor network.

        Maximises the critic's Q-value estimate for actions produced by
        the actor (i.e. gradient ascent on Q via the actor's outputs).

        Parameters
        ----------
        states : np.ndarray
            Batch of states, shape ``(batch, input_dim)``.

        Returns
        -------
        float
            The actor loss (negative mean Q-value) after the update.
        """
        s = tf.convert_to_tensor(states, dtype=tf.float32)

        with tf.GradientTape() as tape:
            actions = self.actor(s, training=True)
            q_values = tf.squeeze(self.critic([s, actions], training=False), axis=-1)
            actor_loss = -tf.reduce_mean(q_values)

        grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(
            zip(grads, self.actor.trainable_weights)
        )

        return float(actor_loss)
