"""Tests for alloc.models.portfolio — Portfolio class and calculate_portfolio_reward."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from alloc.models.portfolio import Portfolio, calculate_portfolio_reward


# ── Fixtures ────────────────────────────────────────────────────────

TICKERS = ["AAPL", "GOOGL", "MSFT"]
INITIAL_CASH = 100_000.0
TX_COST = 0.001  # 0.1%


@pytest.fixture
def prices():
    return {"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 300.0}


@pytest.fixture
def portfolio(prices):
    return Portfolio(tickers=TICKERS, initial_cash=INITIAL_CASH, transaction_cost=TX_COST)


# ── 1. Initialization ──────────────────────────────────────────────

class TestPortfolioInit:
    def test_initial_cash(self):
        p = Portfolio(tickers=TICKERS, initial_cash=50_000.0)
        assert p.cash == 50_000.0

    def test_shares_held_zero(self):
        p = Portfolio(tickers=TICKERS)
        for t in TICKERS:
            assert p.shares_held[t] == 0

    def test_tickers_stored(self):
        p = Portfolio(tickers=TICKERS)
        assert p.tickers == TICKERS

    def test_transaction_cost_default(self):
        p = Portfolio(tickers=TICKERS)
        assert p.transaction_cost == 0.001

    def test_custom_transaction_cost(self):
        p = Portfolio(tickers=TICKERS, transaction_cost=0.005)
        assert p.transaction_cost == 0.005


# ── 2. Portfolio Value ─────────────────────────────────────────────

class TestPortfolioValue:
    def test_empty_portfolio_value(self, portfolio, prices):
        assert portfolio.get_portfolio_value(prices) == INITIAL_CASH

    def test_value_with_holdings(self, portfolio, prices):
        portfolio.shares_held["AAPL"] = 100
        expected = INITIAL_CASH + 100 * 150.0
        assert portfolio.get_portfolio_value(prices) == pytest.approx(expected)

    def test_value_all_holdings(self, portfolio, prices):
        portfolio.shares_held["AAPL"] = 10
        portfolio.shares_held["GOOGL"] = 5
        portfolio.shares_held["MSFT"] = 20
        asset_val = 10 * 150 + 5 * 2800 + 20 * 300
        assert portfolio.get_portfolio_value(prices) == pytest.approx(
            INITIAL_CASH + asset_val
        )

    def test_value_zero_price(self, portfolio):
        prices = {"AAPL": 0.0, "GOOGL": 2800.0, "MSFT": 300.0}
        portfolio.shares_held["AAPL"] = 100
        # Zero price → zero value for AAPL
        assert portfolio.get_portfolio_value(prices) == pytest.approx(INITIAL_CASH)


# ── 3. Allocation ──────────────────────────────────────────────────

class TestGetAllocation:
    def test_all_cash_allocation(self, portfolio, prices):
        alloc = portfolio.get_allocation(prices)
        assert alloc["cash"] == pytest.approx(1.0)
        for t in TICKERS:
            assert alloc[t] == pytest.approx(0.0)

    def test_allocation_with_holdings(self, portfolio, prices):
        portfolio.shares_held["AAPL"] = 100  # $15,000
        total = INITIAL_CASH + 15_000
        alloc = portfolio.get_allocation(prices)
        assert alloc["AAPL"] == pytest.approx(15_000 / total)
        assert alloc["cash"] == pytest.approx(INITIAL_CASH / total)

    def test_allocation_sums_to_one(self, portfolio, prices):
        portfolio.shares_held["AAPL"] = 100
        portfolio.shares_held["GOOGL"] = 10
        alloc = portfolio.get_allocation(prices)
        assert sum(alloc.values()) == pytest.approx(1.0)

    def test_allocation_zero_portfolio(self):
        p = Portfolio(tickers=TICKERS, initial_cash=0.0)
        alloc = p.get_allocation({"AAPL": 150, "GOOGL": 2800, "MSFT": 300})
        assert alloc["cash"] == pytest.approx(1.0)


# ── 4. Trade Execution ────────────────────────────────────────────

class TestExecuteTrades:
    def test_no_op_when_already_target(self, portfolio, prices):
        target = {t: 0.0 for t in TICKERS}
        target["cash"] = 1.0
        result = portfolio.execute_trades(target, prices)
        assert portfolio.cash == pytest.approx(INITIAL_CASH)
        assert result["total_transaction_costs"] == pytest.approx(0.0)

    def test_buy_single_ticker(self, portfolio, prices):
        target = {"AAPL": 0.1, "GOOGL": 0.0, "MSFT": 0.0, "cash": 0.9}
        portfolio.execute_trades(target, prices)
        # Should have bought ~$10k of AAPL (minus tx costs)
        assert portfolio.shares_held["AAPL"] > 0
        assert portfolio.cash < INITIAL_CASH

    def test_sell_all(self, portfolio, prices):
        portfolio.shares_held["AAPL"] = 100
        portfolio.shares_held["GOOGL"] = 50
        target = {t: 0.0 for t in TICKERS}
        target["cash"] = 1.0
        portfolio.execute_trades(target, prices)
        assert portfolio.shares_held["AAPL"] == pytest.approx(0.0, abs=0.01)
        assert portfolio.shares_held["GOOGL"] == pytest.approx(0.0, abs=0.01)
        assert portfolio.cash > INITIAL_CASH

    def test_sells_before_buys(self, portfolio, prices):
        """Selling should happen first to raise cash for buys."""
        portfolio.shares_held["AAPL"] = 200  # $30,000
        portfolio.shares_held["MSFT"] = 0
        # Sell AAPL, buy MSFT
        target = {"AAPL": 0.0, "GOOGL": 0.0, "MSFT": 0.3, "cash": 0.7}
        portfolio.execute_trades(target, prices)
        assert portfolio.shares_held["MSFT"] > 0
        assert portfolio.shares_held["AAPL"] == pytest.approx(0.0, abs=0.01)

    def test_shortfall_scaling(self, portfolio, prices):
        """When transaction costs create a genuine cash gap, buys scale down."""
        # Start with only GOOGL shares and negligible cash.
        # With high transaction costs, the cost of selling + buying exceeds available cash.
        portfolio.cash = 0.01
        portfolio.shares_held["GOOGL"] = 10  # $28,000
        portfolio.transaction_cost = 0.05  # 5% — high enough to create a real gap
        # Target: flip entirely to MSFT
        target = {"AAPL": 0.0, "GOOGL": 0.0, "MSFT": 1.0, "cash": 0.0}
        result = portfolio.execute_trades(target, prices)
        # Cash should not go negative
        assert portfolio.cash >= -0.01
        # MSFT should have been bought but scaled down
        assert portfolio.shares_held["MSFT"] > 0
        # Verify scale factor was applied
        assert result["scale_factor"] < 1.0

    def test_transaction_costs_applied(self, portfolio, prices):
        initial_value = portfolio.get_portfolio_value(prices)
        target = {"AAPL": 0.5, "GOOGL": 0.0, "MSFT": 0.0, "cash": 0.5}
        portfolio.execute_trades(target, prices)
        final_value = portfolio.get_portfolio_value(prices)
        # Portfolio value should decrease by at least the transaction cost amount
        assert final_value < initial_value

    def test_result_has_scale_factor(self, portfolio, prices):
        target = {"AAPL": 0.1, "GOOGL": 0.0, "MSFT": 0.0, "cash": 0.9}
        result = portfolio.execute_trades(target, prices)
        assert "scale_factor" in result
        assert 0 <= result["scale_factor"] <= 1

    def test_result_has_transaction_costs(self, portfolio, prices):
        target = {"AAPL": 0.1, "GOOGL": 0.0, "MSFT": 0.0, "cash": 0.9}
        result = portfolio.execute_trades(target, prices)
        assert "total_transaction_costs" in result
        assert result["total_transaction_costs"] >= 0

    def test_target_normalization(self, portfolio, prices):
        """Target allocation that doesn't sum to 1 should be normalized."""
        target = {"AAPL": 0.5, "GOOGL": 0.5, "MSFT": 0.0, "cash": 0.0}
        portfolio.execute_trades(target, prices)
        # Should have split equally between AAPL and GOOGL
        alloc = portfolio.get_allocation(prices)
        assert alloc["AAPL"] == pytest.approx(alloc["GOOGL"], abs=0.05)

    def test_cash_not_negative_after_trades(self, portfolio, prices):
        """Cash should never go negative after trade execution."""
        portfolio.shares_held["AAPL"] = 1
        target = {"AAPL": 0.0, "GOOGL": 0.9, "MSFT": 0.0, "cash": 0.1}
        portfolio.execute_trades(target, prices)
        assert portfolio.cash >= -0.01


# ── 5. Reward Calculation ──────────────────────────────────────────

class TestCalculatePortfolioReward:
    def test_reward_components_return(self):
        """Pure return component when no risk, no tx, no diversification."""
        current = np.array([0.5, 0.5, 0.0])  # 50/50 split, no cash
        previous = np.array([0.5, 0.5, 0.0])
        returns = np.array([0.1, 0.05])
        reward = calculate_portfolio_reward(
            current=current,
            previous=previous,
            returns=returns,
            risk_free_rate=0.0,
            risk_aversion=0.0,
            transaction_cost=0.0,
            diversification_weight=0.0,
            concentration_penalty=0.0,
        )
        expected = 0.5 * 0.1 + 0.5 * 0.05  # 0.075
        assert reward == pytest.approx(expected)

    def test_risk_penalty(self):
        """Risk aversion should reduce reward."""
        current = np.array([0.5, 0.5, 0.0])
        previous = np.array([0.5, 0.5, 0.0])
        returns = np.array([0.1, -0.1])  # volatile
        reward_no_risk = calculate_portfolio_reward(
            current=current,
            previous=previous,
            returns=returns,
            risk_free_rate=0.0,
            risk_aversion=0.0,
            transaction_cost=0.0,
            diversification_weight=0.0,
            concentration_penalty=0.0,
        )
        reward_with_risk = calculate_portfolio_reward(
            current=current,
            previous=previous,
            returns=returns,
            risk_free_rate=0.0,
            risk_aversion=1.0,
            transaction_cost=0.0,
            diversification_weight=0.0,
            concentration_penalty=0.0,
        )
        assert reward_with_risk < reward_no_risk

    def test_transaction_cost_penalty(self):
        """Large allocation changes should incur transaction penalty."""
        current = np.array([1.0, 0.0, 0.0])
        previous = np.array([0.0, 1.0, 0.0])
        returns = np.array([0.0, 0.0])
        reward = calculate_portfolio_reward(
            current=current,
            previous=previous,
            returns=returns,
            risk_free_rate=0.0,
            risk_aversion=0.0,
            transaction_cost=0.01,
            diversification_weight=0.0,
            concentration_penalty=0.0,
        )
        # Transaction penalty = sum(|delta|) * cost = 2.0 * 0.01 = 0.02
        assert reward == pytest.approx(-0.02)

    def test_diversification_bonus_equal(self):
        """Equal allocation should get diversification bonus."""
        current = np.array([1/3, 1/3, 1/3, 0.0])  # 3 assets equal, cash=0
        previous = np.array([1/3, 1/3, 1/3, 0.0])
        returns = np.array([0.0, 0.0, 0.0])
        reward_no_div = calculate_portfolio_reward(
            current=current,
            previous=previous,
            returns=returns,
            risk_free_rate=0.0,
            risk_aversion=0.0,
            transaction_cost=0.0,
            diversification_weight=0.0,
            concentration_penalty=0.0,
        )
        reward_with_div = calculate_portfolio_reward(
            current=current,
            previous=previous,
            returns=returns,
            risk_free_rate=0.0,
            risk_aversion=0.0,
            transaction_cost=0.0,
            diversification_weight=0.05,
            concentration_penalty=0.0,
        )
        assert reward_with_div > reward_no_div

    def test_concentration_penalty(self):
        """Highly concentrated position should incur penalty."""
        current = np.array([0.9, 0.05, 0.05, 0.0])
        previous = np.array([0.9, 0.05, 0.05, 0.0])
        returns = np.array([0.0, 0.0, 0.0])
        reward_no_pen = calculate_portfolio_reward(
            current=current,
            previous=previous,
            returns=returns,
            risk_free_rate=0.0,
            risk_aversion=0.0,
            transaction_cost=0.0,
            diversification_weight=0.0,
            concentration_penalty=0.0,
        )
        reward_with_pen = calculate_portfolio_reward(
            current=current,
            previous=previous,
            returns=returns,
            risk_free_rate=0.0,
            risk_aversion=0.0,
            transaction_cost=0.0,
            diversification_weight=0.0,
            concentration_penalty=0.02,
        )
        assert reward_with_pen < reward_no_pen

    def test_entropy_component(self):
        """Higher entropy (more diverse) should yield higher diversification score."""
        # Equal allocation → max entropy
        equal = np.array([0.25, 0.25, 0.25, 0.25, 0.0])
        # Concentrated → low entropy
        conc = np.array([0.97, 0.01, 0.01, 0.01, 0.0])
        returns = np.array([0.0, 0.0, 0.0, 0.0])
        prev = equal.copy()

        reward_equal = calculate_portfolio_reward(
            current=equal, previous=prev, returns=returns,
            risk_free_rate=0.0, risk_aversion=0.0,
            transaction_cost=0.0, diversification_weight=0.05,
            concentration_penalty=0.0,
        )
        prev2 = conc.copy()
        reward_conc = calculate_portfolio_reward(
            current=conc, previous=prev2, returns=returns,
            risk_free_rate=0.0, risk_aversion=0.0,
            transaction_cost=0.0, diversification_weight=0.05,
            concentration_penalty=0.0,
        )
        assert reward_equal > reward_conc

    def test_hhi_component(self):
        """HHI should penalize concentration."""
        equal = np.array([0.5, 0.5, 0.0])
        conc = np.array([0.9, 0.1, 0.0])
        returns = np.array([0.0, 0.0])
        prev_eq = equal.copy()
        prev_conc = conc.copy()

        reward_equal = calculate_portfolio_reward(
            current=equal, previous=prev_eq, returns=returns,
            risk_free_rate=0.0, risk_aversion=0.0,
            transaction_cost=0.0, diversification_weight=0.05,
            concentration_penalty=0.0,
        )
        reward_conc = calculate_portfolio_reward(
            current=conc, previous=prev_conc, returns=returns,
            risk_free_rate=0.0, risk_aversion=0.0,
            transaction_cost=0.0, diversification_weight=0.05,
            concentration_penalty=0.0,
        )
        assert reward_equal > reward_conc

    def test_participation_ratio(self):
        """More participating assets should increase diversification bonus."""
        # 2 of 3 participating (3 assets + cash)
        partial = np.array([0.5, 0.5, 0.0, 0.0])
        # 3 of 3 participating
        full = np.array([1/3, 1/3, 1/3, 0.0])
        returns_partial = np.array([0.0, 0.0, 0.0])
        returns_full = np.array([0.0, 0.0, 0.0])
        prev_p = partial.copy()
        prev_f = full.copy()

        reward_partial = calculate_portfolio_reward(
            current=partial, previous=prev_p, returns=returns_partial,
            risk_free_rate=0.0, risk_aversion=0.0,
            transaction_cost=0.0, diversification_weight=0.05,
            concentration_penalty=0.0,
        )
        reward_full = calculate_portfolio_reward(
            current=full, previous=prev_f, returns=returns_full,
            risk_free_rate=0.0, risk_aversion=0.0,
            transaction_cost=0.0, diversification_weight=0.05,
            concentration_penalty=0.0,
        )
        assert reward_full > reward_partial

    def test_full_reward_integration(self):
        """Full reward with all components should be computable."""
        current = np.array([0.3, 0.3, 0.2, 0.2])
        previous = np.array([0.25, 0.25, 0.25, 0.25])
        returns = np.array([0.05, 0.03, -0.02])
        reward = calculate_portfolio_reward(
            current=current,
            previous=previous,
            returns=returns,
            risk_free_rate=0.02,
            risk_aversion=0.5,
            transaction_cost=0.001,
            diversification_weight=0.05,
            concentration_penalty=0.02,
        )
        assert isinstance(reward, (float, np.floating))
        assert not math.isnan(reward)
        assert not math.isinf(reward)

    def test_zero_allocation_no_crash(self):
        """All-zero allocation should not crash."""
        current = np.array([0.0, 0.0, 1.0])
        previous = np.array([0.0, 0.0, 1.0])
        returns = np.array([0.0, 0.0])
        reward = calculate_portfolio_reward(
            current=current,
            previous=previous,
            returns=returns,
            risk_free_rate=0.0,
            risk_aversion=0.0,
            transaction_cost=0.0,
            diversification_weight=0.05,
            concentration_penalty=0.02,
        )
        assert not math.isnan(reward)


# ── Value history + returns analysis (seed-parity) ─────────────────

class TestValueHistory:
    def test_seeded_with_initial_cash(self):
        p = Portfolio(tickers=TICKERS, initial_cash=50_000.0)
        assert p.portfolio_values == [50_000.0]

    def test_record_value_appends(self, portfolio, prices):
        portfolio.shares_held["AAPL"] = 100
        v = portfolio.record_value(prices)
        assert v == pytest.approx(INITIAL_CASH + 100 * 150.0)
        assert len(portfolio.portfolio_values) == 2
        assert portfolio.portfolio_values[-1] == pytest.approx(v)


class TestCalculateReturns:
    def test_neutral_when_insufficient_history(self, portfolio):
        r = portfolio.calculate_returns()
        assert r["daily_returns"] == []
        assert r["cumulative_return"] == 0.0
        assert r["sharpe_ratio"] == 0.0
        assert r["max_drawdown"] == 0.0

    def test_cumulative_and_daily(self, portfolio, prices):
        portfolio.shares_held["AAPL"] = 100
        portfolio.record_value({"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 300.0})
        portfolio.record_value({"AAPL": 160.0, "GOOGL": 2800.0, "MSFT": 300.0})
        r = portfolio.calculate_returns()
        # history: [100000, 115000, 116000] -> 2 daily returns
        assert len(r["daily_returns"]) == 2
        assert r["daily_returns"][0] == pytest.approx(115000.0 / 100000.0 - 1.0)
        assert r["daily_returns"][1] == pytest.approx(116000.0 / 115000.0 - 1.0)
        assert r["cumulative_return"] == pytest.approx(116000.0 / 100000.0 - 1.0)

    def test_max_drawdown(self, portfolio):
        # Build a value path with a known drawdown: 100 -> 120 -> 90 -> 110
        portfolio.portfolio_values = [100.0, 120.0, 90.0, 110.0]
        r = portfolio.calculate_returns()
        # peak 120 -> trough 90 = 25% drawdown
        assert r["max_drawdown"] == pytest.approx(0.25)

    def test_lookback_limits_window(self, portfolio):
        portfolio.portfolio_values = [100.0, 110.0, 121.0, 133.1]
        full = portfolio.calculate_returns()
        limited = portfolio.calculate_returns(lookback=2)
        assert len(full["daily_returns"]) == 3
        assert len(limited["daily_returns"]) == 1
        # last step: 133.1/121 - 1
        assert limited["daily_returns"][0] == pytest.approx(133.1 / 121.0 - 1.0)

    def test_sharpe_positive_for_rising_series(self, portfolio):
        # steadily rising values -> positive mean return -> positive Sharpe
        portfolio.portfolio_values = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        r = portfolio.calculate_returns()
        assert r["sharpe_ratio"] > 0

    def test_annualized_return_present(self, portfolio):
        portfolio.portfolio_values = [100.0, 110.0, 121.0]
        r = portfolio.calculate_returns()
        assert isinstance(r["annualized_return"], float)


# ── Portfolio statistics (seed-parity) ─────────────────────────────

class TestPortfolioStatistics:
    def test_positions_reported(self, portfolio, prices):
        portfolio.shares_held["AAPL"] = 100
        stats = portfolio.calculate_portfolio_statistics(prices)
        pos = stats["positions"]["AAPL"]
        assert pos["shares"] == 100
        assert pos["price"] == pytest.approx(150.0)
        assert pos["value"] == pytest.approx(15000.0)
        assert pos["allocation"] == pytest.approx(15000.0 / stats["portfolio_value"])

    def test_concentration_single_asset(self, portfolio, prices):
        portfolio.shares_held["AAPL"] = 100
        stats = portfolio.calculate_portfolio_statistics(prices)
        c = stats["concentration"]
        assert c["num_assets_held"] == 1
        # single asset -> normalized HHI = 1.0
        assert c["hhi_normalized"] == pytest.approx(1.0)

    def test_concentration_equal_assets(self, portfolio, prices):
        # equal dollar positions -> minimal HHI
        portfolio.shares_held["AAPL"] = 100      # 15000
        portfolio.shares_held["GOOGL"] = 15000.0 / 2800.0  # 15000
        portfolio.shares_held["MSFT"] = 15000.0 / 300.0    # 15000
        stats = portfolio.calculate_portfolio_statistics(prices)
        c = stats["concentration"]
        assert c["num_assets_held"] == 3
        # non-cash weights renormalised to sum to 1 -> equal 1/3 each
        assert c["hhi"] == pytest.approx(1.0 / 3.0)
        # equal weights -> minimal HHI -> normalized ~ 0.0
        assert c["hhi_normalized"] == pytest.approx(0.0, abs=1e-9)

    def test_no_returns_block_without_history(self, portfolio, prices):
        stats = portfolio.calculate_portfolio_statistics(prices)
        assert "returns" not in stats

    def test_returns_block_with_history(self, portfolio, prices):
        portfolio.record_value(prices)
        stats = portfolio.calculate_portfolio_statistics(prices)
        assert "returns" in stats


# ── alloc.__main__ entry point (issue #101) ────────────────────────

class TestMainEntryPoint:
    """Complementary coverage for the ``python -m alloc`` entry point.

    Note: the "zero coverage" claim in issue #101 is stale — ``__main__``
    is already exercised by ``tests/test_cli.py::TestMainModule`` and
    ``tests/test_actor_critic.py::TestMainModule``.  These tests add a
    behavioural check that the entry point delegates to ``alloc.cli.main``
    and propagates its exit code.
    """

    def test_main_is_cli_main(self):
        from alloc.__main__ import main as entry
        from alloc.cli import main as cli_main
        assert entry is cli_main

    def test_main_propagates_exit_code(self):
        from alloc.__main__ import main
        # --help short-circuits with exit code 0
        assert main(["--help"]) == 0
