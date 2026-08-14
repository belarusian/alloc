"""Market data fetching and state vector construction for alloc."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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
        PolygonClient (or compatible) instance — injected, never a
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
            bars = client.get_aggs(
                ticker=formatted,
                multiplier=1,
                timespan="hour",
                from_=hourly_start.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=5000,
            )
            if bars:
                result[ticker]["hourly"] = [float(b.close) for b in bars]

            # --- daily ---
            bars = client.get_aggs(
                ticker=formatted,
                multiplier=1,
                timespan="day",
                from_=daily_start.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=5000,
            )
            if bars:
                result[ticker]["daily"] = [float(b.close) for b in bars]

            # --- weekly ---
            bars = client.get_aggs(
                ticker=formatted,
                multiplier=1,
                timespan="week",
                from_=weekly_start.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
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
# State vector construction
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
                logger.warning("No valid trade for %s", formatted)
        except Exception as exc:
            logger.warning("Error fetching latest price for %s: %s", formatted, exc)
            prices[ticker] = 0.0

    return prices
