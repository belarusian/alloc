"""Market data fetching and state vector construction for alloc."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from alloc.lib.cache import cache_historical, cache_latest_prices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StateBuilder class
# ---------------------------------------------------------------------------

class StateBuilder:
    """Build fixed-dimension state vectors from multi-frequency price data.

    Parameters
    ----------
    hourly_window : int
        Number of hourly bars to include per ticker (default 168 = 1 week).
    daily_window : int
        Number of daily bars to include per ticker (default 365 = 1 year).
    weekly_window : int
        Number of weekly bars to include per ticker (default 52 = 1 year).
    """

    def __init__(
        self,
        hourly_window: int = 168,
        daily_window: int = 365,
        weekly_window: int = 52,
    ) -> None:
        self.hourly_window = hourly_window
        self.daily_window = daily_window
        self.weekly_window = weekly_window

    def _normalize_window(self, prices: list[float]) -> list[float]:
        """Normalise *prices* by dividing each by the last price.

        The result expresses each price as a fraction of the most recent
        price.  The last element is always ``1.0`` (matching the legacy
        ``build_state_vector`` normalisation for backward compatibility).

        Parameters
        ----------
        prices : list[float]
            Raw price series (ordered oldest -> newest).

        Returns
        -------
        list[float]
            Normalised series.  Returns ``[0.0] * len(prices)`` when the
            last price is zero or the list is empty.
        """
        if not prices:
            return []

        last = prices[-1]
        if last == 0.0:
            return [0.0] * len(prices)

        return [p / last for p in prices]

    def _pad_window(self, prices: list[float], target_length: int) -> list[float]:
        """Pad *prices* with leading zeros so the result has *target_length* elements.

        If *prices* already has >= *target_length* elements, only the last
        *target_length* values are returned (no truncation of the newest data).

        Parameters
        ----------
        prices : list[float]
            Price series (ordered oldest -> newest).
        target_length : int
            Desired length of the output window.

        Returns
        -------
        list[float]
            Padded / truncated series of exactly *target_length* elements.
        """
        if target_length <= 0:
            return []

        if len(prices) >= target_length:
            return prices[-target_length:]

        pad_len = target_length - len(prices)
        return [0.0] * pad_len + prices

    def build_state(
        self,
        price_data: dict[str, dict[str, list[float]]],
        allocation: list[float],
    ) -> np.ndarray:
        """Build a state vector from multi-frequency price data.

        For each ticker the last *N* prices per frequency are taken,
        normalised (divide by last price minus 1), padded if necessary,
        and concatenated.  The current allocation percentages are appended
        at the end.

        Parameters
        ----------
        price_data : dict
            ``{ticker: {"hourly": [float], "daily": [float], "weekly": [float]}}``
        allocation : list[float]
            Current portfolio weights (one per ticker).

        Returns
        -------
        np.ndarray
            1-D float64 array of shape ``(N,)`` where
            ``N = len(tickers) * (hourly_window + daily_window + weekly_window)
                + len(allocation)``.  Matches the legacy ``build_state_vector``
            output shape for drop-in replacement in the training pipeline.
        """
        tickers = sorted(price_data.keys())
        parts: list[float] = []

        for ticker in tickers:
            freq_data = price_data.get(ticker, {})
            hourly = freq_data.get("hourly", [])
            daily = freq_data.get("daily", [])
            weekly = freq_data.get("weekly", [])

            # Normalise then pad each frequency window
            parts.extend(
                self._pad_window(
                    self._normalize_window(hourly), self.hourly_window
                )
            )
            parts.extend(
                self._pad_window(
                    self._normalize_window(daily), self.daily_window
                )
            )
            parts.extend(
                self._pad_window(
                    self._normalize_window(weekly), self.weekly_window
                )
            )

        # Append allocation
        parts.extend(allocation)

        return np.array(parts, dtype=np.float64)


# ---------------------------------------------------------------------------
# Legacy function -- kept for backward compatibility
# ---------------------------------------------------------------------------

@cache_historical()
def get_multi_asset_data(
    tickers: list[str],
    client: Any,
    end_date: datetime | None = None,
    hourly_days: int = 7,
    daily_days: int = 365,
    weekly_weeks: int = 52,
) -> dict[str, dict[str, list[float]]]:
    """Fetch hourly, daily, and weekly close prices for *tickers*.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to fetch.
    client : Any
        PolygonClient (or compatible) instance -- injected, never a
        module-level singleton.
    end_date : datetime, optional
        End date for the query window.  Defaults to ``datetime.today()``.
    hourly_days : int
        How many days of hourly bars to retrieve.
    daily_days : int
        How many days of daily bars to retrieve.
    weekly_weeks : int
        How many weeks of weekly bars to retrieve.

    Returns
    -------
    dict
        ``{ticker: {"hourly": [float], "daily": [float], "weekly": [float]}}``
    """
    if end_date is None:
        end_date = datetime.today()

    hourly_start = end_date - timedelta(days=hourly_days)
    daily_start = end_date - timedelta(days=daily_days)
    weekly_start = end_date - timedelta(weeks=weekly_weeks)

    result: dict[str, dict[str, list[float]]] = {
        t: {"hourly": [], "daily": [], "weekly": []} for t in tickers
    }

    for ticker in tickers:
        formatted = ticker.upper()
        try:
            # --- hourly ---
            bars = client.get_aggregate_bars(
                symbol=formatted,
                multiplier=1,
                timespan="hour",
                from_date=hourly_start.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d"),
                limit=5000,
            )
            if bars:
                result[ticker]["hourly"] = [float(b.close) for b in bars]

            # --- daily ---
            bars = client.get_aggregate_bars(
                symbol=formatted,
                multiplier=1,
                timespan="day",
                from_date=daily_start.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d"),
                limit=5000,
            )
            if bars:
                result[ticker]["daily"] = [float(b.close) for b in bars]

            # --- weekly ---
            bars = client.get_aggregate_bars(
                symbol=formatted,
                multiplier=1,
                timespan="week",
                from_date=weekly_start.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d"),
                limit=5000,
            )
            if bars:
                result[ticker]["weekly"] = [float(b.close) for b in bars]

            logger.debug("Fetched multi-freq data for %s", formatted)

        except Exception as exc:
            logger.warning("Error fetching data for %s: %s", formatted, exc)
            # Leave empty lists for this ticker

    return result


# ---------------------------------------------------------------------------
# State vector construction (legacy)
# ---------------------------------------------------------------------------

def build_state_vector(
    multi_freq_data: dict[str, dict[str, list[float]]],
    current_allocation: list[float],
    tickers: list[str],
    n_hourly: int,
    n_daily: int,
    n_weekly: int,
) -> np.ndarray:
    """Build a fixed-dimension state vector from multi-frequency price data.

    For each ticker the last *N* prices per frequency are taken,
    normalised by dividing by the most-recent price in that frequency
    (so the last element is always 1.0).  If fewer than *N* prices
    exist the front is padded with zeros.  All ticker slices are
    concatenated, then the current allocation percentages are appended.

    Parameters
    ----------
    multi_freq_data : dict
        Output of :func:`get_multi_asset_data`.
    current_allocation : list[float]
        Current portfolio weights (length == len(tickers)).
    tickers : list[str]
        Ordered list of ticker symbols.
    n_hourly : int
        Number of hourly bars to include per ticker.
    n_daily : int
        Number of daily bars to include per ticker.
    n_weekly : int
        Number of weekly bars to include per ticker.

    Returns
    -------
    np.ndarray
        1-D float64 array of shape
        ``(len(tickers) * (n_hourly + n_daily + n_weekly) + len(tickers),)``.
    """

    def _normalise(prices: list[float], n: int) -> list[float]:
        """Take last *n* prices, normalise by the most recent, pad with 0."""
        if not prices:
            return [0.0] * n

        last = prices[-1]
        if last == 0.0:
            # Avoid division by zero
            return [0.0] * n

        # Take the last n prices (or all if fewer)
        tail = prices[-n:]
        normalised = [p / last for p in tail]

        # Pad front with zeros if insufficient history
        pad_len = n - len(normalised)
        return [0.0] * pad_len + normalised

    parts: list[float] = []

    for ticker in tickers:
        freq_data = multi_freq_data.get(ticker, {})
        hourly = freq_data.get("hourly", [])
        daily = freq_data.get("daily", [])
        weekly = freq_data.get("weekly", [])

        parts.extend(_normalise(hourly, n_hourly))
        parts.extend(_normalise(daily, n_daily))
        parts.extend(_normalise(weekly, n_weekly))

    # Append allocation
    parts.extend(current_allocation)

    return np.array(parts, dtype=np.float64)


# ---------------------------------------------------------------------------
# Latest prices
# ---------------------------------------------------------------------------

@cache_latest_prices()
def fetch_latest_prices(
    tickers: list[str],
    client: Any,
) -> dict[str, float]:
    """Fetch the latest trade price for each *ticker*.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols.
    client : Any
        PolygonClient (or compatible) instance.

    Returns
    -------
    dict
        ``{ticker: price}``.  Price is ``0.0`` on error or missing data.
    """
    prices: dict[str, float] = {}

    cache_valid = True

    for ticker in tickers:
        formatted = ticker.upper()
        try:
            trade = client.get_last_trade(formatted)
            if trade is not None and hasattr(trade, "price"):
                price = float(trade.price)
                prices[ticker] = price
                logger.debug("Latest price for %s: %.2f", formatted, price)
            else:
                prices[ticker] = 0.0
                cache_valid = False
                logger.warning("No valid trade for %s", formatted)
        except Exception as exc:
            logger.warning("Error fetching latest price for %s: %s", formatted, exc)
            prices[ticker] = 0.0
            cache_valid = False

    if not cache_valid:
        prices["__cache_valid__"] = False

    return prices
