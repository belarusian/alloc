"""Tests for alloc.lib.rebalance — live rebalance entry point (issue #118)."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from alloc.lib import rebalance as rebalance_module
from alloc.lib.rebalance import rebalance_portfolio
from alloc.models.networks import ActorCriticNetworks

TICKERS = ["AAPL", "MSFT"]
# 2 tickers * (5 hourly + 5 daily + 5 weekly) + 2 allocation = 32
INPUT_DIM = 2 * (5 + 5 + 5) + 2
NUM_ASSETS = len(TICKERS) + 1  # +1 for cash
MIN_CASH = 0.05


class _StubClient:
    """Minimal stand-in for a PolygonClient (never actually called)."""


@pytest.fixture()
def saved_model(tmp_path):
    """A small persisted ActorCriticNetworks model on disk."""
    networks = ActorCriticNetworks(
        input_dim=INPUT_DIM,
        num_assets=NUM_ASSETS,
        min_cash_allocation=MIN_CASH,
        seed=42,
    )
    model_dir = tmp_path / "model"
    networks.save_model(model_dir)
    return model_dir


def _fake_prices(tickers, client):
    return {t: 100.0 for t in tickers}


def _fake_multi(tickers, client):
    return {
        t: {
            "hourly": [100.0, 101.0, 102.0, 103.0, 104.0],
            "daily": [90.0, 95.0, 100.0, 105.0, 110.0],
            "weekly": [80.0, 90.0, 100.0, 110.0, 120.0],
        }
        for t in tickers
    }


def _run(saved_model, transaction_cost=0.0, initial_value=None):
    """Run rebalance_portfolio with the data functions patched on the module."""
    with (
        patch.object(
            rebalance_module, "fetch_latest_prices", _fake_prices
        ),
        patch.object(
            rebalance_module, "get_multi_asset_data", _fake_multi
        ),
    ):
        return rebalance_portfolio(
            model_path=saved_model,
            tickers=TICKERS,
            positions={"AAPL": 1000.0, "MSFT": 500.0},
            client=_StubClient(),
            transaction_cost=transaction_cost,
            initial_value=initial_value,
        )


class TestRebalancePortfolio:
    """Behavioural tests for rebalance_portfolio."""

    def test_allocation_sums_to_one(self, saved_model):
        result = _run(saved_model)
        total = sum(result["recommended_allocation"].values())
        assert abs(total - 1.0) < 1e-5

    def test_cash_respects_min_cash(self, saved_model):
        result = _run(saved_model)
        assert result["recommended_allocation"]["cash"] >= MIN_CASH - 1e-9

    def test_orders_produced_for_all_tickers(self, saved_model):
        result = _run(saved_model)
        orders = result["recommended_orders"]
        assert len(orders) == len(TICKERS)
        tickers_in_orders = {o["ticker"] for o in orders}
        assert tickers_in_orders == set(TICKERS)
        for order in orders:
            assert order["action"] in {"buy", "sell", "hold"}
            assert set(order) == {"ticker", "action", "shares", "price", "value"}

    def test_portfolio_value_after_is_finite(self, saved_model):
        result = _run(saved_model)
        assert math.isfinite(result["portfolio_value_after"])
        assert result["portfolio_value_after"] > 0

    def test_value_before_matches_positions(self, saved_model):
        """With initial_value=None the value before equals the invested sum."""
        result = _run(saved_model, initial_value=None)
        assert result["portfolio_value_before"] == pytest.approx(1500.0)

    def test_transaction_costs_applied(self, saved_model):
        """A non-zero transaction cost produces a positive cost and shrinks value."""
        no_cost = _run(saved_model, transaction_cost=0.0)
        with_cost = _run(saved_model, transaction_cost=0.01)

        assert no_cost["total_transaction_costs"] == 0.0
        assert with_cost["total_transaction_costs"] > 0.0
        # Transaction costs reduce the post-trade portfolio value.
        assert with_cost["portfolio_value_after"] < no_cost["portfolio_value_after"]

    def test_scale_factor_in_unit_interval(self, saved_model):
        result = _run(saved_model)
        assert 0.0 <= result["scale_factor"] <= 1.0

    def test_initial_value_derives_cash(self, saved_model):
        """Providing initial_value > invested leaves a positive cash residual."""
        result = _run(saved_model, initial_value=2000.0)
        # Value before should reflect the full initial value.
        assert result["portfolio_value_before"] == pytest.approx(2000.0)

    def test_result_keys(self, saved_model):
        result = _run(saved_model)
        expected = {
            "recommended_allocation",
            "recommended_orders",
            "portfolio_value_before",
            "portfolio_value_after",
            "total_transaction_costs",
            "scale_factor",
        }
        assert expected.issubset(result.keys())
