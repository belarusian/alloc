"""Tests for alloc.lib.utils — utility helpers and price preprocessing."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from alloc.lib.utils import (
    PriceIndex,
    PricePreprocessor,
    create_timestamp_string,
    ensure_scalar_price,
    format_allocation,
    safe_divide,
)

# =====================================================================
# ensure_scalar_price
# =====================================================================


class TestEnsureScalarPrice:
    """Tests for ensure_scalar_price()."""

    def test_plain_float(self) -> None:
        assert ensure_scalar_price(1.5) == 1.5

    def test_plain_int(self) -> None:
        assert ensure_scalar_price(42) == 42.0

    def test_numpy_scalar(self) -> None:
        np = pytest.importorskip("numpy")
        arr = np.float64(3.14)
        assert ensure_scalar_price(arr) == 3.14

    def test_numpy_array_single(self) -> None:
        np = pytest.importorskip("numpy")
        arr = np.array([2.71])
        assert ensure_scalar_price(arr) == 2.71

    def test_numpy_array_multi(self) -> None:
        np = pytest.importorskip("numpy")
        arr = np.array([1.0, 2.0])
        # .item() raises on multi-element arrays
        with pytest.raises(ValueError):
            ensure_scalar_price(arr)

    def test_single_element_list(self) -> None:
        assert ensure_scalar_price([9.9]) == 9.9

    def test_single_element_tuple(self) -> None:
        assert ensure_scalar_price((7.7,)) == 7.7

    def test_multi_element_list_raises(self) -> None:
        with pytest.raises(TypeError):
            ensure_scalar_price([1.0, 2.0])

    def test_string_numeric(self) -> None:
        assert ensure_scalar_price("3.5") == 3.5

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(ValueError):
            ensure_scalar_price("not_a_number")

    def test_none_raises(self) -> None:
        with pytest.raises(TypeError):
            ensure_scalar_price(None)


# =====================================================================
# format_allocation
# =====================================================================


class TestFormatAllocation:
    """Tests for format_allocation()."""

    def test_single_ticker(self) -> None:
        result = format_allocation({"AAPL": 1.0})
        assert result == "AAPL: 100.0000%"

    def test_multiple_tickers_sorted_desc(self) -> None:
        alloc = {"AAPL": 0.3, "GOOG": 0.5, "MSFT": 0.2}
        result = format_allocation(alloc)
        parts = result.split(", ")
        assert parts[0].startswith("GOOG")
        assert parts[1].startswith("AAPL")
        assert parts[2].startswith("MSFT")

    def test_custom_precision(self) -> None:
        result = format_allocation({"X": 0.123456}, precision=2)
        assert "12.35%" in result

    def test_zero_precision(self) -> None:
        result = format_allocation({"X": 0.5}, precision=0)
        assert "50%" in result

    def test_empty_dict(self) -> None:
        result = format_allocation({})
        assert result == ""

    def test_equal_weights(self) -> None:
        alloc = {"A": 0.5, "B": 0.5}
        result = format_allocation(alloc)
        # Both should appear; order is stable but arbitrary for ties
        assert "A: 50.0000%" in result
        assert "B: 50.0000%" in result


# =====================================================================
# create_timestamp_string
# =====================================================================


class TestCreateTimestampString:
    """Tests for create_timestamp_string()."""

    def test_format(self) -> None:
        with patch("alloc.lib.utils.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 1, 15, 10, 30, 45)
            result = create_timestamp_string()
            assert result == "20250115_103045"

    def test_length(self) -> None:
        result = create_timestamp_string()
        assert len(result) == 15  # YYYYMMDD_HHMMSS

    def test_pattern(self) -> None:
        result = create_timestamp_string()
        # Should match digits_digits_digits_digits_digits_digits_digits_
        # digits_digits_digits_digits_digits_digits
        parts = result.split("_")
        assert len(parts) == 2
        assert len(parts[0]) == 8  # date
        assert len(parts[1]) == 6  # time


# =====================================================================
# safe_divide
# =====================================================================


class TestSafeDivide:
    """Tests for safe_divide()."""

    def test_normal_division(self) -> None:
        assert safe_divide(10, 2) == 5.0

    def test_zero_denominator_default(self) -> None:
        assert safe_divide(10, 0) == 0.0

    def test_zero_denominator_custom_default(self) -> None:
        assert safe_divide(10, 0, default=-1.0) == -1.0

    def test_zero_numerator(self) -> None:
        assert safe_divide(0, 5) == 0.0

    def test_both_zero(self) -> None:
        assert safe_divide(0, 0) == 0.0

    def test_float_result(self) -> None:
        result = safe_divide(1, 3)
        assert abs(result - 0.3333333333333333) < 1e-10

    def test_negative_values(self) -> None:
        assert safe_divide(-10, 2) == -5.0
        assert safe_divide(10, -2) == -5.0


# =====================================================================
# PriceIndex dataclass
# =====================================================================


class TestPriceIndex:
    """Tests for the PriceIndex dataclass."""

    def test_defaults(self) -> None:
        idx = PriceIndex()
        assert idx.days_available == 0
        assert idx.prices == {}
        assert idx.complete is False
        assert idx.missing_tickers == []
        assert idx.dates == []

    def test_custom_values(self) -> None:
        idx = PriceIndex(
            days_available=10,
            prices={0: {"A": 1.0}},
            complete=True,
            missing_tickers=[],
            dates=[datetime(2025, 1, 1)],
        )
        assert idx.days_available == 10
        assert idx.complete is True

    def test_equality(self) -> None:
        a = PriceIndex(days_available=5)
        b = PriceIndex(days_available=5)
        assert a == b

    def test_repr(self) -> None:
        idx = PriceIndex(days_available=3)
        assert "PriceIndex" in repr(idx)


# =====================================================================
# PricePreprocessor
# =====================================================================


class TestPricePreprocessor:
    """Tests for the PricePreprocessor class."""

    @pytest.fixture()
    def preprocessor(self) -> PricePreprocessor:
        return PricePreprocessor(gap_tolerance_days=5)

    @pytest.fixture()
    def simple_price_data(self) -> dict[str, dict]:
        """Two tickers with 5 consecutive trading days."""
        base = datetime(2025, 1, 6)  # Monday
        dates = [base + timedelta(days=i) for i in range(5)]
        return {
            "AAPL": {
                "daily": [100 + i for i in range(5)],
                "daily_timestamps": [d.timestamp() * 1000 for d in dates],
            },
            "GOOG": {
                "daily": [200 + i for i in range(5)],
                "daily_timestamps": [d.timestamp() * 1000 for d in dates],
            },
        }

    def test_basic_preprocess(self, preprocessor, simple_price_data) -> None:
        result = preprocessor.preprocess(["AAPL", "GOOG"], simple_price_data)
        assert result.days_available == 5
        assert result.complete is True
        assert result.missing_tickers == []
        assert len(result.dates) == 5

    def test_day_indexed_prices(self, preprocessor, simple_price_data) -> None:
        result = preprocessor.preprocess(["AAPL", "GOOG"], simple_price_data)
        assert 0 in result.prices
        assert "last" in result.prices
        assert result.prices[0]["AAPL"] == 100.0

    def test_last_key_matches_final_day(self, preprocessor, simple_price_data) -> None:
        result = preprocessor.preprocess(["AAPL", "GOOG"], simple_price_data)
        last_idx = result.days_available - 1
        assert result.prices["last"] == result.prices[last_idx]

    def test_missing_ticker(self, preprocessor, simple_price_data) -> None:
        result = preprocessor.preprocess(
            ["AAPL", "GOOG", "MSFT"], simple_price_data
        )
        assert result.complete is False
        assert "MSFT" in result.missing_tickers
        assert result.days_available == 0

    def test_empty_price_data(self, preprocessor) -> None:
        result = preprocessor.preprocess(["AAPL"], {})
        assert result.complete is False
        assert result.days_available == 0

    def test_datetime_timestamps(self, preprocessor) -> None:
        """Test with datetime objects instead of epoch timestamps."""
        base = datetime(2025, 1, 6)
        dates = [base + timedelta(days=i) for i in range(3)]
        data = {
            "X": {
                "daily": [10, 20, 30],
                "daily_timestamps": dates,
            },
        }
        result = preprocessor.preprocess(["X"], data)
        assert result.days_available == 3
        assert result.complete is True

    def test_daily_dates_key(self, preprocessor) -> None:
        """Test with 'daily_dates' key instead of 'daily_timestamps'."""
        base = datetime(2025, 1, 6)
        dates = [base + timedelta(days=i) for i in range(3)]
        data = {
            "X": {
                "daily": [10, 20, 30],
                "daily_dates": dates,
            },
        }
        result = preprocessor.preprocess(["X"], data)
        assert result.days_available == 3

    def test_gap_tolerance(self) -> None:
        """A weekend gap (3 days) should be tolerated with default tolerance."""
        # Mon, Tue, Wed, then skip Fri-Sun, then Mon
        dates = [
            datetime(2025, 1, 6),   # Mon
            datetime(2025, 1, 7),   # Tue
            datetime(2025, 1, 8),   # Wed
            datetime(2025, 1, 13),  # Mon (gap of 5 days from Wed)
        ]
        data = {
            "X": {
                "daily": [10, 20, 30, 40],
                "daily_timestamps": dates,
            },
        }
        result = PricePreprocessor(gap_tolerance_days=5).preprocess(["X"], data)
        assert result.days_available == 4

    def test_gap_exceeds_tolerance(self) -> None:
        """A gap exceeding tolerance should truncate the window."""
        dates = [
            datetime(2025, 1, 6),   # Mon
            datetime(2025, 1, 7),   # Tue
            datetime(2025, 1, 20),  # Mon (gap of 13 days)
        ]
        data = {
            "X": {
                "daily": [10, 20, 30],
                "daily_timestamps": dates,
            },
        }
        result = PricePreprocessor(gap_tolerance_days=5).preprocess(["X"], data)
        # Only the last contiguous run: Jan 20
        assert result.days_available == 1
        assert result.prices[0]["X"] == 30.0

    def test_no_common_dates(self) -> None:
        """Two tickers with completely disjoint date ranges."""
        data = {
            "A": {
                "daily": [1, 2],
                "daily_timestamps": [
                    datetime(2025, 1, 6),
                    datetime(2025, 1, 7),
                ],
            },
            "B": {
                "daily": [3, 4],
                "daily_timestamps": [
                    datetime(2025, 2, 3),
                    datetime(2025, 2, 4),
                ],
            },
        }
        result = PricePreprocessor().preprocess(["A", "B"], data)
        assert result.days_available == 0
        assert result.complete is False

    def test_single_ticker(self, preprocessor) -> None:
        base = datetime(2025, 1, 6)
        dates = [base + timedelta(days=i) for i in range(3)]
        data = {
            "X": {
                "daily": [100, 200, 300],
                "daily_timestamps": dates,
            },
        }
        result = preprocessor.preprocess(["X"], data)
        assert result.days_available == 3
        assert result.complete is True
        assert len(result.dates) == 3

    def test_dates_are_sorted(self, preprocessor, simple_price_data) -> None:
        result = preprocessor.preprocess(["AAPL", "GOOG"], simple_price_data)
        assert result.dates == sorted(result.dates)

    def test_custom_gap_tolerance(self) -> None:
        """Tighter tolerance should break the window sooner."""
        dates = [
            datetime(2025, 1, 6),
            datetime(2025, 1, 7),
            datetime(2025, 1, 10),  # gap of 3 days
        ]
        data = {
            "X": {
                "daily": [10, 20, 30],
                "daily_timestamps": dates,
            },
        }
        # With tolerance=2, the 3-day gap breaks the window
        result = PricePreprocessor(gap_tolerance_days=2).preprocess(["X"], data)
        assert result.days_available == 1  # Only Jan 10

    def test_missing_daily_key(self, preprocessor) -> None:
        data = {
            "X": {
                "other_key": [1, 2],
            },
        }
        result = preprocessor.preprocess(["X"], data)
        assert result.complete is False
        assert "X" in result.missing_tickers

    def test_missing_timestamps_key(self, preprocessor) -> None:
        data = {
            "X": {
                "daily": [1, 2, 3],
            },
        }
        result = preprocessor.preprocess(["X"], data)
        assert result.complete is False
        assert "X" in result.missing_tickers

    def test_ticker_not_in_price_data(self, preprocessor) -> None:
        result = preprocessor.preprocess(["UNKNOWN"], {})
        assert result.complete is False
        assert "UNKNOWN" in result.missing_tickers
