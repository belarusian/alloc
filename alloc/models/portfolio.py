"""alloc.models.portfolio — Portfolio management and reward calculation.

Provides :class:`Portfolio` for tracking holdings, executing trades with
transaction costs and shortfall scaling, and
:func:`calculate_portfolio_reward` for computing a composite reward signal
combining return, risk, transaction costs, diversification, and concentration.

HHI note (issue #112)
---------------------
The concentration index reported by
:meth:`Portfolio.calculate_portfolio_statistics` is computed on the
*renormalised non-cash* weights: the per-ticker allocations are divided by
their sum so they total 1.0 before squaring.  This keeps the Herfindahl–
Hirschman index well-defined and bounded in ``[1/n, 1]`` (normalised form in
``[0, 1]``) regardless of how much cash the portfolio holds.

This deliberately differs from the seed reference, which squares the raw
non-cash fractions without renormalising.  In a cash-heavy portfolio those raw
fractions sum to well below 1.0, so the seed's normalised HHI can fall below
zero (and is not a valid concentration measure).  The two values are therefore
not directly comparable; the renormalised form used here is the one that stays
in ``[0, 1]``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ── Portfolio ───────────────────────────────────────────────────────

class Portfolio:
    """Track shares, cash, and execute allocation-driven trades.

    Attributes
    ----------
    tickers : list[str]
        Universe of tradeable symbols.
    cash : float
        Current cash balance.
    transaction_cost : float
        Fractional cost applied to *total traded value* (buys + sells).
    shares_held : dict[str, float]
        Number of shares held per ticker.
    """

    def __init__(
        self,
        tickers: list[str],
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
    ) -> None:
        self.tickers = list(tickers)
        self.cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.shares_held: dict[str, float] = {t: 0.0 for t in self.tickers}
        # Value history: one entry per recorded valuation (seeded with initial cash).
        self.portfolio_values: list[float] = [float(initial_cash)]

    # -- valuation --------------------------------------------------

    def get_portfolio_value(self, prices: dict[str, float]) -> float:
        """Total portfolio value = sum(shares × price) + cash."""
        asset_value = sum(
            self.shares_held.get(t, 0.0) * prices.get(t, 0.0)
            for t in self.tickers
        )
        return asset_value + self.cash

    def get_allocation(self, prices: dict[str, float]) -> dict[str, float]:
        """Current allocation percentages (including ``'cash'``).

        Returns a dict mapping each ticker and ``'cash'`` to its fraction
        of total portfolio value.  If total value ≤ 0 the entire portfolio
        is reported as 100 % cash.
        """
        total = self.get_portfolio_value(prices)
        if total <= 0:
            alloc: dict[str, float] = {t: 0.0 for t in self.tickers}
            alloc["cash"] = 1.0
            return alloc

        alloc = {}
        for t in self.tickers:
            alloc[t] = (self.shares_held.get(t, 0.0) * prices.get(t, 0.0)) / total
        alloc["cash"] = self.cash / total
        return alloc

    # -- trade execution --------------------------------------------

    def execute_trades(
        self,
        target_allocation: dict[str, float],
        prices: dict[str, float],
    ) -> dict[str, Any]:
        """Rebalance the portfolio toward *target_allocation*.

        Strategy
        --------
        1. Compute value deltas per ticker (target − current).
        2. Separate into buys and sells.
        3. Execute sells first to raise cash.
        4. If cash is insufficient for all buys, scale buys down
           proportionally (shortfall scaling).
        5. Apply transaction costs on total traded value.

        Parameters
        ----------
        target_allocation : dict
            Desired weight per ticker plus ``'cash'``.  Values need not
            sum to 1.0 — they will be normalised.
        prices : dict
            Current price per ticker.

        Returns
        -------
        dict
            Execution metadata including ``scale_factor`` and
            ``total_transaction_costs``.
        """
        portfolio_value = self.get_portfolio_value(prices)

        # Normalise target so it sums to 1.0
        total_target = sum(target_allocation.values())
        if total_target > 0 and abs(total_target - 1.0) > 1e-9:
            target_allocation = {k: v / total_target for k, v in target_allocation.items()}

        current_alloc = self.get_allocation(prices)

        # Target and current asset values
        target_values: dict[str, float] = {}
        current_values: dict[str, float] = {}
        for t in self.tickers:
            if t not in prices:
                continue
            target_values[t] = target_allocation.get(t, 0.0) * portfolio_value
            current_values[t] = self.shares_held.get(t, 0.0) * prices[t]

        # Classify into buys / sells
        buys: dict[str, float] = {}
        sells: dict[str, float] = {}
        for t in self.tickers:
            if t not in prices:
                continue
            delta = target_values.get(t, 0.0) - current_values.get(t, 0.0)
            if delta > 0:
                buys[t] = delta
            elif delta < 0:
                sells[t] = -delta

        total_buy_value = sum(buys.values())
        total_sell_value = sum(sells.values())

        # --- Step 1: execute sells (raise cash) --------------------
        for t, sell_value in sells.items():
            price = prices[t]
            if price <= 0:
                continue
            shares_to_sell = min(sell_value / price, self.shares_held.get(t, 0.0))
            self.shares_held[t] -= shares_to_sell
            self.cash += shares_to_sell * price

        # --- Step 2: scale buys if insufficient cash ---------------
        total_traded = total_buy_value + total_sell_value
        tx_cost = total_traded * self.transaction_cost
        required_cash = total_buy_value + tx_cost
        scale_factor = 1.0

        if required_cash > self.cash + 1e-9:
            # Solve for scale_factor s.t.
            #   cash >= s*total_buy + (s*total_buy + total_sell)*tx
            # => s = (cash - total_sell*tx) / (total_buy*(1+tx))
            denom = total_buy_value * (1 + self.transaction_cost)
            if denom > 0:
                scale_factor = max(
                    0.0,
                    (self.cash - total_sell_value * self.transaction_cost) / denom,
                )
            else:
                scale_factor = 0.0
            scale_factor = min(1.0, scale_factor)

        # --- Step 3: execute scaled buys ---------------------------
        actual_buy_value = 0.0
        for t, buy_value in buys.items():
            scaled_value = buy_value * scale_factor
            price = prices[t]
            if price <= 0:
                continue
            shares_to_buy = scaled_value / price
            self.shares_held[t] += shares_to_buy
            self.cash -= scaled_value
            actual_buy_value += scaled_value

        # --- Step 4: deduct transaction costs ----------------------
        actual_traded = actual_buy_value + total_sell_value
        actual_tx = actual_traded * self.transaction_cost
        self.cash -= actual_tx

        # Ensure cash never goes negative due to float drift
        if self.cash < 0:
            self.cash = 0.0

        return {
            "scale_factor": scale_factor,
            "total_transaction_costs": actual_tx,
            "total_buy_value": actual_buy_value,
            "total_sell_value": total_sell_value,
            "portfolio_value_before": portfolio_value,
            "current_allocation": current_alloc,
            "target_allocation": target_allocation,
        }


    # -- value history ----------------------------------------------

    def record_value(self, prices: dict[str, float]) -> float:
        """Append the current portfolio value to :attr:`portfolio_values`.

        Call this once per valuation step (e.g. after each day's trades)
        so that :meth:`calculate_returns` has a series to analyse.

        Parameters
        ----------
        prices : dict
            Current price per ticker.

        Returns
        -------
        float
            The value that was recorded.
        """
        value = self.get_portfolio_value(prices)
        self.portfolio_values.append(value)
        return value

    # -- returns analysis -------------------------------------------

    def calculate_returns(self, lookback: int | None = None) -> dict[str, Any]:
        """Compute return metrics from :attr:`portfolio_values`.

        Metrics
        -------
        * ``daily_returns`` — per-period simple returns (length = n-1).
        * ``cumulative_return`` — total growth over the window.
        * ``annualized_return`` — geometric annualisation at 252 periods/yr.
        * ``sharpe_ratio`` — annualised, risk-free rate 2% (daily rf = 0.02/252).
        * ``max_drawdown`` — largest peak-to-trough decline (fraction, ≥ 0).

        Parameters
        ----------
        lookback : int, optional
            If given, only the trailing *lookback* values are used.

        Returns
        -------
        dict
            The metrics above.  With fewer than two recorded values every
            metric is returned at its neutral value (empty list / 0.0).
        """
        neutral = {
            "daily_returns": [],
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }
        if len(self.portfolio_values) < 2:
            return neutral

        values = self.portfolio_values
        if lookback is not None and lookback < len(values):
            values = values[-lookback:]

        daily_returns = [
            (values[i] / values[i - 1]) - 1.0
            for i in range(1, len(values))
            if values[i - 1] > 0
        ]

        cumulative_return = (values[-1] / values[0]) - 1.0 if values[0] > 0 else 0.0

        # Annualised return (geometric, 252 periods/yr)
        if daily_returns:
            avg_daily = sum(daily_returns) / len(daily_returns)
            annualized_return = (1.0 + avg_daily) ** 252 - 1.0
        else:
            avg_daily = 0.0
            annualized_return = 0.0

        # Sharpe ratio (rf = 2% annual)
        if len(daily_returns) > 1:
            daily_std = float(np.std(daily_returns))
            rf_daily = 0.02 / 252.0
            sharpe_ratio = (
                ((avg_daily - rf_daily) / daily_std) * (252.0 ** 0.5)
                if daily_std > 0
                else 0.0
            )
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        peak = values[0]
        max_drawdown = 0.0
        for value in values:
            peak = max(peak, value)
            if peak > 0:
                dd = (peak - value) / peak
                max_drawdown = max(max_drawdown, dd)

        return {
            "daily_returns": daily_returns,
            "cumulative_return": cumulative_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }

    # -- portfolio statistics ---------------------------------------

    def calculate_portfolio_statistics(self, prices: dict[str, float]) -> dict[str, Any]:
        """Report per-position detail and a concentration summary.

        Parameters
        ----------
        prices : dict
            Current price per ticker.

        Returns
        -------
        dict
            ``portfolio_value``, ``cash``, ``cash_allocation``, ``positions``
            (per-ticker shares/price/value/allocation), and ``concentration``
            with the Herfindahl-Hirschman index (raw + normalised) and the
            number of assets held.  A ``returns`` block is included when at
            least two values have been recorded.
        """
        portfolio_value = self.get_portfolio_value(prices)
        allocation = self.get_allocation(prices)

        positions: dict[str, dict[str, float]] = {}
        for t in self.tickers:
            if t in prices:
                positions[t] = {
                    "shares": self.shares_held.get(t, 0.0),
                    "price": prices[t],
                    "value": self.shares_held.get(t, 0.0) * prices[t],
                    "allocation": allocation.get(t, 0.0),
                }

        # HHI over non-cash positions with a positive allocation.
        # The non-cash weights are renormalised to sum to 1 so the index
        # is well-defined regardless of the cash position: HHI then lies in
        # [1/n, 1] and the normalised form in [0, 1] (0 = equal, 1 = single).
        non_cash = [
            allocation[t]
            for t in self.tickers
            if allocation.get(t, 0.0) > 0
        ]
        if non_cash:
            total_nc = sum(non_cash)
            weights = [w / total_nc for w in non_cash] if total_nc > 0 else []
            hhi = float(np.sum(np.square(weights)))
            n = len(weights)
            if n > 1:
                hhi_normalized = (hhi - (1.0 / n)) / (1.0 - (1.0 / n))
            else:
                hhi_normalized = 1.0
        else:
            hhi = 0.0
            hhi_normalized = 0.0

        stats: dict[str, Any] = {
            "portfolio_value": portfolio_value,
            "cash": self.cash,
            "cash_allocation": allocation.get("cash", 0.0),
            "positions": positions,
            "concentration": {
                "hhi": hhi,
                "hhi_normalized": hhi_normalized,
                "num_assets_held": len(non_cash),
            },
        }

        if len(self.portfolio_values) > 1:
            stats["returns"] = self.calculate_returns()

        return stats


# ── Reward ──────────────────────────────────────────────────────────

def calculate_portfolio_reward(
    current: np.ndarray,
    previous: np.ndarray,
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    risk_aversion: float = 0.5,
    transaction_cost: float = 0.001,
    diversification_weight: float = 0.05,
    concentration_penalty: float = 0.02,
) -> float:
    """Composite portfolio reward.

    Components
    ----------
    * **Return** — weighted sum of asset returns (ex-cash).
    * **Risk penalty** — ``risk_aversion × portfolio_volatility``.
    * **Transaction cost penalty** — ``L1(Δalloc) × transaction_cost``.
    * **Diversification bonus** — entropy + HHI + participation ratio,
      normalised and scaled by ``diversification_weight``.
    * **Concentration penalty** — quadratic penalty when any single
      position exceeds a threshold, scaled by ``concentration_penalty``.

    Parameters
    ----------
    current : np.ndarray
        Current allocation including cash as the last element.
    previous : np.ndarray
        Previous allocation (same shape).
    returns : np.ndarray
        Period returns for each asset (ex-cash, length = n_assets).
    risk_free_rate : float
        Risk-free rate (used for Sharpe-style normalisation).
    risk_aversion : float
        Weight on volatility penalty.
    transaction_cost : float
        Cost per unit of allocation change.
    diversification_weight : float
        Multiplier on the diversification bonus.
    concentration_penalty : float
        Multiplier on the concentration penalty.

    Returns
    -------
    float
        Composite reward value.
    """
    # Separate assets from cash (last element)
    asset_alloc = current[:-1]
    n_assets = len(asset_alloc)

    # --- Return component ------------------------------------------
    portfolio_return = float(np.sum(asset_alloc * returns))

    # --- Risk penalty ----------------------------------------------
    portfolio_vol = float(np.std(asset_alloc * returns)) if n_assets > 1 else 0.0
    risk_penalty = risk_aversion * portfolio_vol

    # --- Transaction cost penalty ----------------------------------
    transactions = float(np.sum(np.abs(current - previous)))
    tx_penalty = transactions * transaction_cost

    # --- Diversification bonus -------------------------------------
    eps = 1e-10

    # Shannon entropy
    entropy = 0.0
    for a in asset_alloc:
        if a > eps:
            entropy -= a * math.log(a)

    # HHI
    hhi = float(np.sum(current ** 2))

    # Participation ratio
    participation = float(np.sum(asset_alloc > 0.01)) / n_assets if n_assets > 0 else 0.0

    # Normalise entropy
    max_entropy = -math.log(1.0 / n_assets) if n_assets > 0 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Normalise HHI (1 = most diverse, 0 = least)
    ideal_hhi = 1.0 / n_assets if n_assets > 0 else 0.0
    if n_assets > 1 and (1 - ideal_hhi) > 0:
        norm_hhi = 1.0 - (hhi - ideal_hhi) / (1 - ideal_hhi)
    else:
        norm_hhi = 0.0

    diversification_score = 0.4 * norm_entropy + 0.3 * norm_hhi + 0.3 * participation
    diversification_score = diversification_score ** 0.5

    scaling_factor = math.log(n_assets + 1) / math.log(5) if n_assets > 0 else 0.0
    diversification_bonus = diversification_score * diversification_weight * scaling_factor

    # --- Concentration penalty -------------------------------------
    ideal_alloc = 1.0 / n_assets if n_assets > 0 else 0.0
    concentration_threshold = min(0.4, ideal_alloc * 1.5)
    max_alloc = float(np.max(asset_alloc)) if n_assets > 0 else 0.0

    if max_alloc > concentration_threshold:
        excess = max_alloc - concentration_threshold
        conc_penalty = excess ** 2 * concentration_penalty * (n_assets ** 0.5) * 3.0
    else:
        conc_penalty = 0.0

    # --- Composite reward ------------------------------------------
    reward = (
        portfolio_return
        - risk_penalty
        - tx_penalty
        + diversification_bonus
        - conc_penalty
    )
    return float(reward)
