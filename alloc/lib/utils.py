"""alloc.lib.utils — utility helpers for the allocation engine.

Provides scalar-price coercion, allocation formatting, timestamp generation,
safe division, and a structured price-index dataclass with a configurable
preprocessor that builds contiguous trailing windows across tickers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def ensure_scalar_price(price: Any) -> float:
    """Coerce *price* to a plain Python ``float``.

    Handles numpy scalars/arrays (via ``.item()``), single-element
    sequences, and direct numeric types.

    Parameters
    ----------
    price:
        Any price-like value.

    Returns
    -------
    float
        The scalar price.

    Raises
    ------
    TypeError
        If *price* cannot be converted to ``float``.
    """
    if hasattr(price, "item"):
        # numpy scalar or array
        return float(price.item())
    if isinstance(price, (list, tuple)) and len(price) == 1:
        return float(price[0])
    return float(price)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_allocation(allocation: dict[str, float], precision: int = 4) -> str:
    """Return a human-readable allocation string sorted by weight descending.

    Parameters
    ----------
    allocation:
        Mapping of ticker symbol to fractional weight (e.g. ``0.42``).
    precision:
        Number of decimal places in the percentage display.

    Returns
    -------
    str
        Comma-separated ``"TICKER: XX.XXXX%"`` entries.
    """
    fmt = f"{{0}}: {{1:.{precision}f}}%"
    sorted_items = sorted(allocation.items(), key=lambda kv: kv[1], reverse=True)
    parts = [fmt.format(ticker, weight * 100) for ticker, weight in sorted_items]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def create_timestamp_string() -> str:
    """Return the current local time as ``YYYYMMDD_HHMMSS``."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Safe arithmetic
# ---------------------------------------------------------------------------

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide *numerator* by *denominator*, returning *default* on zero.

    Parameters
    ----------
    numerator:
        The dividend.
    denominator:
        The divisor.
    default:
        Value returned when *denominator* is ``0``.

    Returns
    -------
    float
    """
    if denominator == 0:
        return default
    return numerator / denominator


# ---------------------------------------------------------------------------
# Price index dataclass
# ---------------------------------------------------------------------------

@dataclass
class PriceIndex:
    """Structured result of price-data preprocessing.

    Attributes
    ----------
    days_available:
        Number of contiguous trading days available for *all* tickers.
    prices:
        Day-indexed mapping ``{day_index: {ticker: price}}``.  A special
        key ``"last"`` holds the most recent day's prices.
    complete:
        ``True`` when every requested ticker has data.
    missing_tickers:
        Tickers that had no usable price data.
    dates:
        Chronological list of ``datetime`` objects for each trading day.
    """

    days_available: int = 0
    prices: dict[Any, Any] = field(default_factory=dict)
    complete: bool = False
    missing_tickers: list[str] = field(default_factory=list)
    dates: list[datetime] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Price preprocessor
# ---------------------------------------------------------------------------

class PricePreprocessor:
    """Build a contiguous trailing price window across multiple tickers.

    Parameters
    ----------
    gap_tolerance_days:
        Maximum allowed gap (in calendar days) between consecutive trading
        dates before the contiguous window is considered broken.  Defaults
        to 5 to absorb weekends and holidays.
    """

    def __init__(self, gap_tolerance_days: int = 5) -> None:
        self.gap_tolerance = timedelta(days=gap_tolerance_days)

    def preprocess(
        self,
        tickers: list[str],
        price_data: dict[str, dict],
    ) -> PriceIndex:
        """Preprocess raw price data into a structured :class:`PriceIndex`.

        The algorithm:

        1. For each ticker, normalise timestamps and sort chronologically.
        2. Walk backwards from the most recent date, collecting a trailing
           contiguous window (gaps up to *gap_tolerance* are allowed).
        3. Intersect the trailing windows across all tickers.
        4. Trim to the most recent contiguous run shared by every ticker.

        Parameters
        ----------
        tickers:
            Ordered list of ticker symbols.
        price_data:
            Nested mapping ``{ticker: {"daily": [...], "daily_timestamps": [...]}}``.
            The ``daily_timestamps`` key may also be ``daily_dates``.

        Returns
        -------
        PriceIndex
        """
        missing_tickers: list[str] = []
        ticker_price_maps: dict[str, dict[datetime, float]] = {}
        ticker_tail_dates: list[list[datetime]] = []

        for ticker in tickers:
            raw = price_data.get(ticker, {})
            daily_prices = raw.get("daily", [])
            timestamps = raw.get("daily_timestamps") or raw.get("daily_dates")

            if not daily_prices or not timestamps:
                missing_tickers.append(ticker)
                continue

            # Normalise to (datetime, float) pairs
            normalised: list[tuple[datetime, float]] = []
            for price, ts in zip(daily_prices, timestamps):
                if isinstance(ts, datetime):
                    dt = ts
                else:
                    dt = datetime.fromtimestamp(ts / 1000, tz=None)
                normalised.append((dt, float(price)))

            if not normalised:
                missing_tickers.append(ticker)
                continue

            normalised.sort(key=lambda pair: pair[0])

            # Build trailing contiguous window
            tail_dates: list[datetime] = [normalised[-1][0]]
            for date, _price in reversed(normalised[:-1]):
                if tail_dates[0] - date <= self.gap_tolerance:
                    tail_dates.insert(0, date)
                else:
                    break

            date_to_price = {
                dt: price for dt, price in normalised if dt in tail_dates
            }
            ticker_price_maps[ticker] = date_to_price
            ticker_tail_dates.append(tail_dates)

        # Early exit if any ticker is missing
        if len(ticker_price_maps) != len(tickers):
            return PriceIndex(
                days_available=0,
                prices={},
                complete=False,
                missing_tickers=missing_tickers or tickers,
                dates=[],
            )

        # Intersection of all trailing windows
        common_dates: set[datetime] = set(ticker_tail_dates[0])
        for tail in ticker_tail_dates[1:]:
            common_dates &= set(tail)

        if not common_dates:
            return PriceIndex(
                days_available=0,
                prices={},
                complete=False,
                missing_tickers=tickers,
                dates=[],
            )

        # Trim to most recent contiguous run
        sorted_common = sorted(common_dates)
        contiguous_desc: list[datetime] = []
        last_dt: datetime | None = None
        for current in reversed(sorted_common):
            if last_dt is None or (last_dt - current) <= self.gap_tolerance:
                contiguous_desc.append(current)
                last_dt = current
            else:
                break

        final_dates = list(reversed(contiguous_desc))

        if not final_dates:
            return PriceIndex(
                days_available=0,
                prices={},
                complete=False,
                missing_tickers=tickers,
                dates=[],
            )

        # Build day-indexed price map
        prices: dict[Any, Any] = {}
        for day_idx, trading_dt in enumerate(final_dates):
            prices[day_idx] = {
                ticker: ticker_price_maps[ticker][trading_dt] for ticker in tickers
            }

        if final_dates:
            prices["last"] = prices[len(final_dates) - 1]

        return PriceIndex(
            days_available=len(final_dates),
            prices=prices,
            complete=len(missing_tickers) == 0,
            missing_tickers=missing_tickers,
            dates=final_dates,
        )
