"""Tests for alloc.models.data.StateBuilder.

Covers construction, normalisation, padding, and full state-building
with edge cases (empty data, zero prices, insufficient history).
"""

from __future__ import annotations

import numpy as np
import pytest

from alloc.models.data import StateBuilder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def builder() -> StateBuilder:
    """Default StateBuilder with standard window sizes."""
    return StateBuilder()


@pytest.fixture()
def builder_small() -> StateBuilder:
    """StateBuilder with small windows for easy manual verification."""
    return StateBuilder(hourly_window=3, daily_window=5, weekly_window=2)


@pytest.fixture()
def sample_price_data() -> dict[str, dict[str, list[float]]]:
    """Two-ticker price data with enough history for small windows."""
    return {
        "AAPL": {
            "hourly": [100.0, 101.0, 102.0, 103.0, 104.0],
            "daily": [90.0, 92.0, 94.0, 96.0, 98.0, 100.0],
            "weekly": [80.0, 85.0, 90.0],
        },
        "MSFT": {
            "hourly": [200.0, 202.0, 204.0, 206.0, 208.0],
            "daily": [180.0, 185.0, 190.0, 195.0, 200.0, 205.0],
            "weekly": [170.0, 180.0, 190.0],
        },
    }


# =====================================================================
# __init__
# =====================================================================

class TestStateBuilderInit:
    """Tests for StateBuilder construction."""

    def test_default_windows(self) -> None:
        b = StateBuilder()
        assert b.hourly_window == 168
        assert b.daily_window == 365
        assert b.weekly_window == 52

    def test_custom_windows(self) -> None:
        b = StateBuilder(hourly_window=10, daily_window=20, weekly_window=30)
        assert b.hourly_window == 10
        assert b.daily_window == 20
        assert b.weekly_window == 30

    def test_zero_window(self) -> None:
        b = StateBuilder(hourly_window=0, daily_window=0, weekly_window=0)
        assert b.hourly_window == 0
        assert b.daily_window == 0
        assert b.weekly_window == 0

    def test_single_custom_window(self) -> None:
        b = StateBuilder(hourly_window=50)
        assert b.hourly_window == 50
        assert b.daily_window == 365  # default
        assert b.weekly_window == 52  # default


# =====================================================================
# _normalize_window
# =====================================================================

class TestNormalizeWindow:
    """Tests for StateBuilder._normalize_window."""

    def test_basic_normalization(self, builder_small: StateBuilder) -> None:
        prices = [100.0, 110.0, 120.0]
        result = builder_small._normalize_window(prices)
        # (100/120) = 0.8333..., (110/120) = 0.9166..., (120/120) = 1.0
        assert len(result) == 3
        assert abs(result[0] - (100.0 / 120.0)) < 1e-10
        assert abs(result[1] - (110.0 / 120.0)) < 1e-10
        assert abs(result[2] - 1.0) < 1e-10

    def test_last_element_is_one(self, builder_small: StateBuilder) -> None:
        prices = [50.0, 60.0, 70.0, 80.0]
        result = builder_small._normalize_window(prices)
        assert abs(result[-1] - 1.0) < 1e-10

    def test_single_price(self, builder_small: StateBuilder) -> None:
        prices = [100.0]
        result = builder_small._normalize_window(prices)
        assert result == [1.0]

    def test_empty_prices(self, builder_small: StateBuilder) -> None:
        result = builder_small._normalize_window([])
        assert result == []

    def test_zero_last_price(self, builder_small: StateBuilder) -> None:
        prices = [100.0, 50.0, 0.0]
        result = builder_small._normalize_window(prices)
        assert result == [0.0, 0.0, 0.0]

    def test_all_zero_prices(self, builder_small: StateBuilder) -> None:
        prices = [0.0, 0.0, 0.0]
        result = builder_small._normalize_window(prices)
        assert result == [0.0, 0.0, 0.0]

    def test_preserves_length(self, builder_small: StateBuilder) -> None:
        prices = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = builder_small._normalize_window(prices)
        assert len(result) == len(prices)

    def test_negative_returns(self, builder_small: StateBuilder) -> None:
        prices = [200.0, 150.0, 100.0]
        result = builder_small._normalize_window(prices)
        assert result[0] == 2.0  # 200/100
        assert result[1] == 1.5  # 150/100
        assert abs(result[2] - 1.0) < 1e-10

    def test_no_nan_or_inf(self, builder_small: StateBuilder) -> None:
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = builder_small._normalize_window(prices)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()


# =====================================================================
# _pad_window
# =====================================================================

class TestPadWindow:
    """Tests for StateBuilder._pad_window."""

    def test_no_padding_needed(self, builder_small: StateBuilder) -> None:
        prices = [1.0, 2.0, 3.0]
        result = builder_small._pad_window(prices, 3)
        assert result == [1.0, 2.0, 3.0]

    def test_pad_with_zeros(self, builder_small: StateBuilder) -> None:
        prices = [1.0, 2.0]
        result = builder_small._pad_window(prices, 5)
        assert result == [0.0, 0.0, 0.0, 1.0, 2.0]

    def test_truncate_to_target(self, builder_small: StateBuilder) -> None:
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = builder_small._pad_window(prices, 3)
        assert result == [3.0, 4.0, 5.0]

    def test_exact_length(self, builder_small: StateBuilder) -> None:
        prices = [1.0, 2.0, 3.0]
        result = builder_small._pad_window(prices, 3)
        assert result == [1.0, 2.0, 3.0]

    def test_empty_prices_padded(self, builder_small: StateBuilder) -> None:
        result = builder_small._pad_window([], 4)
        assert result == [0.0, 0.0, 0.0, 0.0]

    def test_target_length_zero(self, builder_small: StateBuilder) -> None:
        prices = [1.0, 2.0]
        result = builder_small._pad_window(prices, 0)
        assert result == []

    def test_preserves_order(self, builder_small: StateBuilder) -> None:
        prices = [10.0, 20.0]
        result = builder_small._pad_window(prices, 5)
        # Leading zeros, then original order preserved
        assert result[-2:] == [10.0, 20.0]

    def test_large_padding(self, builder_small: StateBuilder) -> None:
        prices = [1.0]
        result = builder_small._pad_window(prices, 100)
        assert len(result) == 100
        assert result[0] == 0.0
        assert result[-1] == 1.0


# =====================================================================
# build_state
# =====================================================================

class TestBuildState:
    """Tests for StateBuilder.build_state."""

    def test_returns_numpy_array(
        self, builder_small: StateBuilder, sample_price_data: dict
    ) -> None:
        state = builder_small.build_state(sample_price_data, [0.5, 0.5])
        assert isinstance(state, np.ndarray)

    def test_shape_is_1d(
        self, builder_small: StateBuilder, sample_price_data: dict
    ) -> None:
        """StateBuilder returns 1-D array matching legacy build_state_vector."""
        state = builder_small.build_state(sample_price_data, [0.5, 0.5])
        assert state.ndim == 1

    def test_shape_width_correct(
        self, builder_small: StateBuilder, sample_price_data: dict
    ) -> None:
        # 2 tickers * (3 hourly + 5 daily + 2 weekly) + 2 allocation = 20
        expected_n = 2 * (3 + 5 + 2) + 2
        state = builder_small.build_state(sample_price_data, [0.5, 0.5])
        assert state.shape == (expected_n,)

    def test_single_ticker(self, builder_small: StateBuilder) -> None:
        data = {
            "AAPL": {
                "hourly": [100.0, 101.0, 102.0],
                "daily": [90.0, 92.0, 94.0, 96.0, 98.0],
                "weekly": [80.0, 85.0],
            },
        }
        state = builder_small.build_state(data, [1.0])
        expected_n = 1 * (3 + 5 + 2) + 1
        assert state.shape == (expected_n,)

    def test_allocation_appended(self, builder_small: StateBuilder) -> None:
        data = {
            "AAPL": {
                "hourly": [100.0, 101.0, 102.0],
                "daily": [90.0, 92.0, 94.0, 96.0, 98.0],
                "weekly": [80.0, 85.0],
            },
        }
        state = builder_small.build_state(data, [1.0])
        assert state[-1] == 1.0

    def test_two_ticker_allocation(
        self, builder_small: StateBuilder, sample_price_data: dict
    ) -> None:
        state = builder_small.build_state(sample_price_data, [0.3, 0.7])
        assert state[-2] == 0.3
        assert state[-1] == 0.7

    def test_normalization_in_state(self, builder_small: StateBuilder) -> None:
        data = {
            "AAPL": {
                "hourly": [100.0, 200.0, 300.0],
                "daily": [10.0, 20.0, 30.0, 40.0, 50.0],
                "weekly": [5.0, 10.0],
            },
        }
        state = builder_small.build_state(data, [1.0])
        # Hourly: 100/300=0.3333, 200/300=0.6667, 300/300=1.0
        assert abs(state[0] - (100.0 / 300.0)) < 1e-10
        assert abs(state[1] - (200.0 / 300.0)) < 1e-10
        assert abs(state[2] - 1.0) < 1e-10

    def test_padding_in_state(self, builder_small: StateBuilder) -> None:
        data = {
            "AAPL": {
                "hourly": [100.0],  # only 1 bar, window=3
                "daily": [10.0, 20.0],  # only 2 bars, window=5
                "weekly": [5.0],  # only 1 bar, window=2
            },
        }
        state = builder_small.build_state(data, [1.0])
        # Hourly: pad 2 zeros + normalize [100] -> [0.0, 0.0, 1.0]
        assert state[0] == 0.0
        assert state[1] == 0.0
        assert abs(state[2] - 1.0) < 1e-10  # 100/100 = 1.0

    def test_empty_data(self, builder_small: StateBuilder) -> None:
        data = {
            "AAPL": {
                "hourly": [],
                "daily": [],
                "weekly": [],
            },
        }
        state = builder_small.build_state(data, [1.0])
        expected_n = 1 * (3 + 5 + 2) + 1
        assert state.shape == (expected_n,)
        # All zeros except allocation
        assert state[-1] == 1.0
        assert np.all(state[:-1] == 0.0)

    def test_zero_prices_no_crash(self, builder_small: StateBuilder) -> None:
        data = {
            "AAPL": {
                "hourly": [0.0, 0.0, 0.0],
                "daily": [0.0, 0.0, 0.0, 0.0, 0.0],
                "weekly": [0.0, 0.0],
            },
        }
        state = builder_small.build_state(data, [1.0])
        assert not np.isnan(state).any()
        assert not np.isinf(state).any()

    def test_dtype_float64(
        self, builder_small: StateBuilder, sample_price_data: dict
    ) -> None:
        state = builder_small.build_state(sample_price_data, [0.5, 0.5])
        assert state.dtype == np.float64

    def test_ticker_order_sorted(self, builder_small: StateBuilder) -> None:
        """Tickers are sorted alphabetically for deterministic ordering."""
        data = {
            "MSFT": {
                "hourly": [200.0, 201.0, 202.0],
                "daily": [180.0, 185.0, 190.0, 195.0, 200.0],
                "weekly": [170.0, 180.0],
            },
            "AAPL": {
                "hourly": [100.0, 101.0, 102.0],
                "daily": [90.0, 92.0, 94.0, 96.0, 98.0],
                "weekly": [80.0, 85.0],
            },
        }
        state = builder_small.build_state(data, [0.5, 0.5])
        # AAPL comes first (alphabetical), its hourly normalized:
        # 100/102, 101/102, 102/102
        assert abs(state[0] - (100.0 / 102.0)) < 1e-10

    def test_default_builder_large_windows(
        self, builder: StateBuilder
    ) -> None:
        """With default large windows, small data gets heavily padded."""
        data = {
            "AAPL": {
                "hourly": [100.0, 101.0],
                "daily": [90.0, 91.0],
                "weekly": [80.0],
            },
        }
        state = builder.build_state(data, [1.0])
        expected_n = 1 * (168 + 365 + 52) + 1
        assert state.shape == (expected_n,)

    def test_three_tickers(self, builder_small: StateBuilder) -> None:
        data = {
            "GOOGL": {
                "hourly": [1.0, 2.0, 3.0],
                "daily": [1.0, 2.0, 3.0, 4.0, 5.0],
                "weekly": [1.0, 2.0],
            },
            "AAPL": {
                "hourly": [10.0, 20.0, 30.0],
                "daily": [10.0, 20.0, 30.0, 40.0, 50.0],
                "weekly": [10.0, 20.0],
            },
            "MSFT": {
                "hourly": [100.0, 200.0, 300.0],
                "daily": [100.0, 200.0, 300.0, 400.0, 500.0],
                "weekly": [100.0, 200.0],
            },
        }
        state = builder_small.build_state(data, [0.33, 0.33, 0.34])
        expected_n = 3 * (3 + 5 + 2) + 3
        assert state.shape == (expected_n,)
        # Last 3 elements are allocation
        assert abs(state[-3] - 0.33) < 1e-10
        assert abs(state[-2] - 0.33) < 1e-10
        assert abs(state[-1] - 0.34) < 1e-10
