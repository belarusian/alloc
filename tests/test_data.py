"""Tests for alloc.models.data — market data fetching and state vector construction.

All tests mock the PolygonClient to avoid real API calls.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alloc.models.data import (
    build_state_vector,
    fetch_latest_prices,
    get_multi_asset_data,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(close: float, timestamp: int | None = None) -> SimpleNamespace:
    """Create a mock aggregate bar object."""
    return SimpleNamespace(
        close=close,
        open=close - 0.1,
        high=close + 0.1,
        low=close - 0.2,
        volume=1000,
        vwap=close,
        timestamp=timestamp,
    )


def _make_trade(price: float, timestamp: int | None = None) -> SimpleNamespace:
    """Create a mock last-trade object."""
    return SimpleNamespace(price=price, timestamp=timestamp or 1700000000000)


def _mock_client() -> MagicMock:
    """Return a MagicMock standing in for PolygonClient."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_client() -> MagicMock:
    return _mock_client()


@pytest.fixture()
def end_date() -> datetime:
    return datetime(2024, 6, 15, 12, 0, 0)


@pytest.fixture()
def tickers() -> list[str]:
    return ["AAPL", "MSFT"]


# =====================================================================
# get_multi_asset_data
# =====================================================================

class TestGetMultiAssetData:
    """Tests for get_multi_asset_data."""

    def test_returns_dict_with_ticker_keys(self, mock_client: MagicMock, end_date: datetime) -> None:
        mock_client.get_aggs.return_value = [_make_bar(100.0)]
        result = get_multi_asset_data(
            tickers=["AAPL"], client=mock_client, end_date=end_date,
            hourly_days=1, daily_days=1, weekly_weeks=1,
        )
        assert "AAPL" in result

    def test_returns_dict_with_frequency_keys(self, mock_client: MagicMock, end_date: datetime) -> None:
        mock_client.get_aggs.return_value = [_make_bar(100.0)]
        result = get_multi_asset_data(
            tickers=["AAPL"], client=mock_client, end_date=end_date,
            hourly_days=1, daily_days=1, weekly_weeks=1,
        )
        for freq in ("hourly", "daily", "weekly"):
            assert freq in result["AAPL"]

    def test_hourly_prices_extracted_correctly(self, mock_client: MagicMock, end_date: datetime) -> None:
        bars = [_make_bar(100.0), _make_bar(101.0), _make_bar(102.0)]
        mock_client.get_aggs.return_value = bars
        result = get_multi_asset_data(
            tickers=["AAPL"], client=mock_client, end_date=end_date,
            hourly_days=1, daily_days=1, weekly_weeks=1,
        )
        assert result["AAPL"]["hourly"] == [100.0, 101.0, 102.0]

    def test_daily_prices_extracted_correctly(self, mock_client: MagicMock, end_date: datetime) -> None:
        bars = [_make_bar(200.0), _make_bar(205.0)]
        mock_client.get_aggs.return_value = bars
        result = get_multi_asset_data(
            tickers=["AAPL"], client=mock_client, end_date=end_date,
            hourly_days=1, daily_days=1, weekly_weeks=1,
        )
        assert result["AAPL"]["daily"] == [200.0, 205.0]

    def test_weekly_prices_extracted_correctly(self, mock_client: MagicMock, end_date: datetime) -> None:
        bars = [_make_bar(300.0)]
        mock_client.get_aggs.return_value = bars
        result = get_multi_asset_data(
            tickers=["AAPL"], client=mock_client, end_date=end_date,
            hourly_days=1, daily_days=1, weekly_weeks=1,
        )
        assert result["AAPL"]["weekly"] == [300.0]

    def test_calls_get_aggs_with_correct_timespan_hourly(
        self, mock_client: MagicMock, end_date: datetime
    ) -> None:
        mock_client.get_aggs.return_value = []
        get_multi_asset_data(
            tickers=["AAPL"], client=mock_client, end_date=end_date,
            hourly_days=7, daily_days=30, weekly_weeks=12,
        )
        # First call should be hourly
        call_args = mock_client.get_aggs.call_args_list[0]
        assert call_args[1]["timespan"] == "hour"

    def test_calls_get_aggs_with_correct_timespan_daily(
        self, mock_client: MagicMock, end_date: datetime
    ) -> None:
        mock_client.get_aggs.return_value = []
        get_multi_asset_data(
            tickers=["AAPL"], client=mock_client, end_date=end_date,
            hourly_days=7, daily_days=30, weekly_weeks=12,
        )
        # Second call should be daily
        call_args = mock_client.get_aggs.call_args_list[1]
        assert call_args[1]["timespan"] == "day"

    def test_calls_get_aggs_with_correct_timespan_weekly(
        self, mock_client: MagicMock, end_date: datetime
    ) -> None:
        mock_client.get_aggs.return_value = []
        get_multi_asset_data(
            tickers=["AAPL"], client=mock_client, end_date=end_date,
            hourly_days=7, daily_days=30, weekly_weeks=12,
        )
        # Third call should be week
        call_args = mock_client.get_aggs.call_args_list[2]
        assert call_args[1]["timespan"] == "week"

    def test_handles_empty_bars(self, mock_client: MagicMock, end_date: datetime) -> None:
        mock_client.get_aggs.return_value = []
        result = get_multi_asset_data(
            tickers=["AAPL"], client=mock_client, end_date=end_date,
            hourly_days=1, daily_days=1, weekly_weeks=1,
        )
        assert result["AAPL"]["hourly"] == []
        assert result["AAPL"]["daily"] == []
        assert result["AAPL"]["weekly"] == []

    def test_handles_multiple_tickers(self, mock_client: MagicMock, end_date: datetime) -> None:
        mock_client.get_aggs.return_value = [_make_bar(100.0)]
        result = get_multi_asset_data(
            tickers=["AAPL", "MSFT"], client=mock_client, end_date=end_date,
            hourly_days=1, daily_days=1, weekly_weeks=1,
        )
        assert "AAPL" in result
        assert "MSFT" in result

    def test_ticker_uppercased(self, mock_client: MagicMock, end_date: datetime) -> None:
        mock_client.get_aggs.return_value = []
        get_multi_asset_data(
            tickers=["aapl"], client=mock_client, end_date=end_date,
            hourly_days=1, daily_days=1, weekly_weeks=1,
        )
        # First call's ticker arg should be uppercased
        call_args = mock_client.get_aggs.call_args_list[0]
        assert call_args[1]["ticker"] == "AAPL"

    def test_defaults_end_date_to_today(self, mock_client: MagicMock) -> None:
        mock_client.get_aggs.return_value = []
        with patch("alloc.models.data.datetime") as mock_dt:
            mock_dt.today.return_value = datetime(2024, 1, 1)
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            get_multi_asset_data(
                tickers=["AAPL"], client=mock_client,
                hourly_days=1, daily_days=1, weekly_weeks=1,
            )
            # Should have been called
            assert mock_client.get_aggs.called

    def test_error_on_ticker_does_not_crash_others(
        self, mock_client: MagicMock, end_date: datetime
    ) -> None:
        """If one ticker raises, others should still be processed."""
        def side_effect(*args, **kwargs):
            if kwargs.get("ticker") == "BAD":
                raise RuntimeError("API error")
            return [_make_bar(100.0)]

        mock_client.get_aggs.side_effect = side_effect
        result = get_multi_asset_data(
            tickers=["GOOD", "BAD"], client=mock_client, end_date=end_date,
            hourly_days=1, daily_days=1, weekly_weeks=1,
        )
        assert "GOOD" in result
        assert "BAD" in result
        # BAD ticker should have empty data
        assert result["BAD"]["hourly"] == []

    def test_uses_injected_client_not_singleton(self, mock_client: MagicMock, end_date: datetime) -> None:
        """Verify the function uses the passed client, not a module-level singleton."""
        mock_client.get_aggs.return_value = [_make_bar(42.0)]
        result = get_multi_asset_data(
            tickers=["AAPL"], client=mock_client, end_date=end_date,
            hourly_days=1, daily_days=1, weekly_weeks=1,
        )
        assert result["AAPL"]["hourly"] == [42.0]
        assert mock_client.get_aggs.called


# =====================================================================
# build_state_vector
# =====================================================================

class TestBuildStateVector:
    """Tests for build_state_vector."""

    def _make_data(self, prices: dict[str, dict[str, list[float]]]) -> dict:
        return prices

    def test_basic_shape(self) -> None:
        """State vector length = n_tickers * (n_h + n_d + n_w) + n_tickers."""
        data = {
            "AAPL": {"hourly": [1.0] * 5, "daily": [1.0] * 5, "weekly": [1.0] * 5},
            "MSFT": {"hourly": [1.0] * 5, "daily": [1.0] * 5, "weekly": [1.0] * 5},
        }
        alloc = [0.5, 0.5]
        vec = build_state_vector(data, alloc, ["AAPL", "MSFT"], n_hourly=3, n_daily=3, n_weekly=3)
        expected_len = 2 * (3 + 3 + 3) + 2  # 2 tickers * 9 features + 2 alloc
        assert len(vec) == expected_len

    def test_normalization_by_last_price(self) -> None:
        """Prices should be normalized as pct change from the most recent price."""
        data = {
            "AAPL": {
                "hourly": [100.0, 101.0, 102.0],
                "daily": [200.0, 202.0],
                "weekly": [300.0],
            },
        }
        alloc = [1.0]
        vec = build_state_vector(data, alloc, ["AAPL"], n_hourly=3, n_daily=2, n_weekly=1)
        # Hourly: last=102.0, so [100/102, 101/102, 102/102] = [0.9804, 0.9902, 1.0]
        hourly_norm = vec[:3]
        assert abs(hourly_norm[2] - 1.0) < 1e-6
        assert abs(hourly_norm[0] - 100.0 / 102.0) < 1e-6
        assert abs(hourly_norm[1] - 101.0 / 102.0) < 1e-6

    def test_padding_with_zeros(self) -> None:
        """If fewer prices than N, pad with zeros at the front."""
        data = {
            "AAPL": {
                "hourly": [100.0, 101.0],  # only 2 bars
                "daily": [200.0],
                "weekly": [300.0],
            },
        }
        alloc = [1.0]
        vec = build_state_vector(data, alloc, ["AAPL"], n_hourly=5, n_daily=3, n_weekly=2)
        # Hourly: need 5, have 2 → pad 3 zeros, then normalize last=101.0
        # [0, 0, 0, 100/101, 101/101]
        hourly_part = vec[:5]
        assert hourly_part[0] == 0.0
        assert hourly_part[1] == 0.0
        assert hourly_part[2] == 0.0
        assert abs(hourly_part[3] - 100.0 / 101.0) < 1e-6
        assert abs(hourly_part[4] - 1.0) < 1e-6

    def test_appends_allocation(self) -> None:
        """Current allocation percentages appended at the end."""
        data = {
            "AAPL": {"hourly": [1.0], "daily": [1.0], "weekly": [1.0]},
            "MSFT": {"hourly": [1.0], "daily": [1.0], "weekly": [1.0]},
        }
        alloc = [0.6, 0.4]
        vec = build_state_vector(data, alloc, ["AAPL", "MSFT"], n_hourly=1, n_daily=1, n_weekly=1)
        # Last 2 elements should be the allocation
        assert vec[-2:] == pytest.approx([0.6, 0.4])

    def test_empty_history_pads_all_zeros(self) -> None:
        """When all price lists are empty, the vector should be all zeros except allocation."""
        data = {
            "AAPL": {"hourly": [], "daily": [], "weekly": []},
        }
        alloc = [1.0]
        vec = build_state_vector(data, alloc, ["AAPL"], n_hourly=3, n_daily=3, n_weekly=3)
        # First 9 elements should be zeros, last element = 1.0
        assert all(v == 0.0 for v in vec[:-1])
        assert vec[-1] == 1.0

    def test_returns_numpy_array(self) -> None:
        data = {
            "AAPL": {"hourly": [1.0], "daily": [1.0], "weekly": [1.0]},
        }
        alloc = [1.0]
        vec = build_state_vector(data, alloc, ["AAPL"], n_hourly=1, n_daily=1, n_weekly=1)
        assert isinstance(vec, np.ndarray)

    def test_ticker_order_preserved(self) -> None:
        """State vector should follow the order of the tickers list."""
        data = {
            "AAPL": {"hourly": [100.0, 200.0], "daily": [1.0], "weekly": [1.0]},
            "MSFT": {"hourly": [300.0, 400.0], "daily": [1.0], "weekly": [1.0]},
        }
        alloc = [0.5, 0.5]
        vec = build_state_vector(data, alloc, ["AAPL", "MSFT"], n_hourly=2, n_daily=1, n_weekly=1)
        # AAPL hourly normalized: [100/200, 200/200] = [0.5, 1.0]
        assert abs(vec[0] - 0.5) < 1e-6
        assert abs(vec[1] - 1.0) < 1e-6
        # MSFT hourly starts at index 4 (AAPL: 2h+1d+1w=4)
        # MSFT hourly normalized: [300/400, 400/400] = [0.75, 1.0]
        assert abs(vec[4] - 0.75) < 1e-6
        assert abs(vec[5] - 1.0) < 1e-6

    def test_division_by_zero_handled(self) -> None:
        """If last price is 0, avoid division by zero."""
        data = {
            "AAPL": {"hourly": [0.0, 0.0], "daily": [0.0], "weekly": [0.0]},
        }
        alloc = [1.0]
        vec = build_state_vector(data, alloc, ["AAPL"], n_hourly=2, n_daily=1, n_weekly=1)
        # Should not raise; values should be 0.0
        assert not np.isnan(vec).any()
        assert not np.isinf(vec).any()

    def test_single_ticker(self) -> None:
        data = {
            "AAPL": {"hourly": [10.0, 20.0], "daily": [30.0], "weekly": [40.0]},
        }
        alloc = [1.0]
        vec = build_state_vector(data, alloc, ["AAPL"], n_hourly=2, n_daily=1, n_weekly=1)
        expected_len = 1 * (2 + 1 + 1) + 1  # 5
        assert len(vec) == expected_len


# =====================================================================
# fetch_latest_prices
# =====================================================================

class TestFetchLatestPrices:
    """Tests for fetch_latest_prices."""

    def test_returns_dict_with_prices(self, mock_client: MagicMock) -> None:
        mock_client.get_last_trade.return_value = _make_trade(150.0)
        result = fetch_latest_prices(tickers=["AAPL"], client=mock_client)
        assert result["AAPL"] == 150.0

    def test_calls_get_last_trade_for_each_ticker(self, mock_client: MagicMock) -> None:
        mock_client.get_last_trade.return_value = _make_trade(100.0)
        fetch_latest_prices(tickers=["AAPL", "MSFT"], client=mock_client)
        assert mock_client.get_last_trade.call_count == 2

    def test_ticker_uppercased_in_call(self, mock_client: MagicMock) -> None:
        mock_client.get_last_trade.return_value = _make_trade(100.0)
        fetch_latest_prices(tickers=["aapl"], client=mock_client)
        mock_client.get_last_trade.assert_called_with("AAPL")

    def test_handles_missing_trade(self, mock_client: MagicMock) -> None:
        mock_client.get_last_trade.return_value = None
        result = fetch_latest_prices(tickers=["AAPL"], client=mock_client)
        assert result["AAPL"] == 0.0

    def test_handles_trade_without_price_attr(self, mock_client: MagicMock) -> None:
        mock_client.get_last_trade.return_value = SimpleNamespace()
        result = fetch_latest_prices(tickers=["AAPL"], client=mock_client)
        assert result["AAPL"] == 0.0

    def test_handles_api_error(self, mock_client: MagicMock) -> None:
        mock_client.get_last_trade.side_effect = RuntimeError("network error")
        result = fetch_latest_prices(tickers=["AAPL"], client=mock_client)
        assert result["AAPL"] == 0.0

    def test_error_on_one_ticker_does_not_crash_others(self, mock_client: MagicMock) -> None:
        def side_effect(ticker):
            if ticker == "BAD":
                raise RuntimeError("fail")
            return _make_trade(100.0)

        mock_client.get_last_trade.side_effect = side_effect
        result = fetch_latest_prices(tickers=["GOOD", "BAD"], client=mock_client)
        assert result["GOOD"] == 100.0
        assert result["BAD"] == 0.0

    def test_uses_injected_client(self, mock_client: MagicMock) -> None:
        mock_client.get_last_trade.return_value = _make_trade(42.0)
        result = fetch_latest_prices(tickers=["AAPL"], client=mock_client)
        assert result["AAPL"] == 42.0
        assert mock_client.get_last_trade.called

    def test_multiple_tickers_different_prices(self, mock_client: MagicMock) -> None:
        def side_effect(ticker):
            prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 140.0}
            return _make_trade(prices[ticker])

        mock_client.get_last_trade.side_effect = side_effect
        result = fetch_latest_prices(tickers=["AAPL", "MSFT", "GOOGL"], client=mock_client)
        assert result["AAPL"] == 150.0
        assert result["MSFT"] == 300.0
        assert result["GOOGL"] == 140.0
