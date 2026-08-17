"""alloc.lib.rebalance — live rebalance entry point.

Provides :func:`rebalance_portfolio`, which loads a persisted
:class:`~alloc.models.networks.ActorCriticNetworks`, builds the current
market state from live prices and the caller's holdings, asks the actor for
a recommended allocation, and executes the rebalance against a
:class:`~alloc.models.portfolio.Portfolio` to produce concrete orders.

The function is the single public entry point for the ``--rebalance`` CLI
mode.  All market-data access goes through the public data-pipeline API
(:func:`~alloc.models.data.fetch_latest_prices`,
:func:`~alloc.models.data.get_multi_asset_data`,
:func:`~alloc.models.data.build_state_vector`) so that tests can patch those
references on this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from alloc.models.data import (
    build_state_vector,
    fetch_latest_prices,
    get_multi_asset_data,
)
from alloc.models.networks import ActorCriticNetworks
from alloc.models.portfolio import Portfolio

logger = logging.getLogger(__name__)


def rebalance_portfolio(
    model_path: str | Path,
    tickers: list[str],
    positions: dict[str, float],
    client: Any,
    n_hourly: int = 5,
    n_daily: int = 5,
    n_weekly: int = 5,
    transaction_cost: float = 0.0,
    initial_value: float | None = None,
) -> dict[str, Any]:
    """Compute a live rebalance recommendation for *tickers*.

    Parameters
    ----------
    model_path : str | Path
        Directory produced by
        :meth:`~alloc.models.networks.ActorCriticNetworks.save_model`.
    tickers : list[str]
        Ordered list of tradeable ticker symbols.
    positions : dict
        Current holdings as ``{ticker: dollar_value}``.  The dollar value of
        each position is used to derive the number of shares held
        (``dollar_value / price``) and the residual cash.
    client : Any
        PolygonClient (or compatible) instance used for price/data fetching.
    n_hourly : int
        Number of hourly bars per ticker for the state vector (default 5).
    n_daily : int
        Number of daily bars per ticker for the state vector (default 5).
    n_weekly : int
        Number of weekly bars per ticker for the state vector (default 5).
    transaction_cost : float
        Fractional cost applied to total traded value (default 0.0).
    initial_value : float | None
        Total portfolio value.  If ``None`` it is derived as the sum of the
        position dollar values (i.e. the residual cash is assumed to be 0).

    Returns
    -------
    dict
        ``recommended_allocation`` (ticker -> weight, incl. ``'cash'``),
        ``recommended_orders`` (list of ``{ticker, action, shares, price,
        value}``), ``portfolio_value_before``, ``portfolio_value_after``,
        ``total_transaction_costs``, and ``scale_factor``.
    """
    # (a) Load the persisted model.
    networks = ActorCriticNetworks.load_model(model_path)

    # (b) Fetch latest prices.
    prices = fetch_latest_prices(tickers, client)

    # (c) Fetch multi-frequency history for the state vector.
    multi_freq = get_multi_asset_data(tickers, client)

    # (d) Build the current non-cash allocation (per-ticker weights).
    position_values = {t: float(positions.get(t, 0.0)) for t in tickers}
    invested = sum(position_values.values())
    if initial_value is None:
        initial_value = invested
    cash = initial_value - invested
    if initial_value > 0:
        current_allocation = [position_values[t] / initial_value for t in tickers]
    else:
        current_allocation = [0.0 for t in tickers]

    # (e) Build the state vector.
    state = build_state_vector(
        multi_freq, current_allocation, tickers, n_hourly, n_daily, n_weekly
    )

    # (f) Ask the actor for a recommended allocation.
    allocation = networks.get_allocation(state)

    # (g) Seed the portfolio with the current holdings.
    portfolio = Portfolio(
        tickers=tickers,
        initial_cash=initial_value,
        transaction_cost=transaction_cost,
    )
    for t in tickers:
        price = prices.get(t, 0.0)
        if price > 0:
            portfolio.shares_held[t] = position_values[t] / price
        else:
            portfolio.shares_held[t] = 0.0
    portfolio.cash = cash

    shares_before = dict(portfolio.shares_held)

    # (h) Build the target allocation dict (tickers + cash).
    target: dict[str, float] = {
        t: float(allocation[i]) for i, t in enumerate(tickers)
    }
    target["cash"] = float(allocation[-1])

    # (i) Execute the rebalance.
    execution = portfolio.execute_trades(target, prices)

    # Derive per-ticker orders from the change in shares held.
    recommended_orders: list[dict[str, Any]] = []
    for t in tickers:
        price = prices.get(t, 0.0)
        delta_shares = portfolio.shares_held[t] - shares_before[t]
        value = abs(delta_shares) * price
        if abs(delta_shares) < 1e-9:
            action = "hold"
        elif delta_shares > 0:
            action = "buy"
        else:
            action = "sell"
        recommended_orders.append(
            {
                "ticker": t,
                "action": action,
                "shares": round(delta_shares, 6),
                "price": price,
                "value": round(value, 6),
            }
        )

    recommended_allocation: dict[str, float] = {
        t: float(allocation[i]) for i, t in enumerate(tickers)
    }
    recommended_allocation["cash"] = float(allocation[-1])

    portfolio_value_before = float(execution["portfolio_value_before"])
    portfolio_value_after = float(portfolio.get_portfolio_value(prices))

    return {
        "recommended_allocation": recommended_allocation,
        "recommended_orders": recommended_orders,
        "portfolio_value_before": portfolio_value_before,
        "portfolio_value_after": portfolio_value_after,
        "total_transaction_costs": float(execution["total_transaction_costs"]),
        "scale_factor": float(execution["scale_factor"]),
    }
