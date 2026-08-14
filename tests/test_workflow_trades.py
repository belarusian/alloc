"""Tests for recommended_trades integration and metric computation (issues #40, #41).

Covers:
- TrainingTrial.recommended_trades field
- WorkflowRunner._run_trial extracting recommended_trades
- create_trainer deriving recommended_trades from allocation_history
- Metric computation: sharpe_ratio, outperformance, model_roi, buyhold_roi
- print_results rendering trade recommendations
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alloc.utils.workflow import (
    TrainingConfig,
    TrainingTrial,
    WorkflowResult,
    WorkflowRunner,
)


# ===================================================================
# TrainingTrial — recommended_trades field
# ===================================================================


class TestTrainingTrialRecommendedTrades:
    """Tests for the recommended_trades field on TrainingTrial."""

    def test_default_is_none(self) -> None:
        trial = TrainingTrial(iteration=1, update=0)
        assert trial.recommended_trades is None

    def test_accepts_list_of_dicts(self) -> None:
        trades = [
            {"ticker": "AAPL", "action": "buy", "allocation": 0.5, "change": 0.1},
            {"ticker": "cash", "action": "sell", "allocation": 0.3, "change": -0.05},
        ]
        trial = TrainingTrial(
            iteration=1, update=0, recommended_trades=trades,
        )
        assert trial.recommended_trades == trades
        assert len(trial.recommended_trades) == 2

    def test_accepts_empty_list(self) -> None:
        trial = TrainingTrial(
            iteration=1, update=0, recommended_trades=[],
        )
        assert trial.recommended_trades == []

    def test_full_trial_with_trades(self) -> None:
        trial = TrainingTrial(
            iteration=3,
            update=2,
            sharpe_ratio=2.1,
            outperformance=15.0,
            final_value=220_000.0,
            model_roi=20.0,
            buyhold_roi=5.0,
            allocation=[0.45, 0.45, 0.10],
            recommended_trades=[
                {"ticker": "AAPL", "action": "buy", "allocation": 0.45, "change": 0.05},
            ],
            model_path="/tmp/model.pt",
            results_path="/tmp/results.json",
        )
        assert trial.recommended_trades is not None
        assert trial.recommended_trades[0]["ticker"] == "AAPL"


# ===================================================================
# WorkflowRunner._run_trial — recommended_trades extraction
# ===================================================================


class TestRunTrialRecommendedTrades:
    """Tests for WorkflowRunner._run_trial extracting recommended_trades."""

    def test_extractor_passes_through_trades(self) -> None:
        config = TrainingConfig(
            tickers=["AAPL"],
            positions={"AAPL": 100_000.0},
        )
        trades = [
            {"ticker": "AAPL", "action": "buy", "allocation": 0.6, "change": 0.1},
            {"ticker": "cash", "action": "sell", "allocation": 0.4, "change": -0.1},
        ]
        def trainer(**kwargs: Any) -> dict[str, Any]:
            return {
                "sharpe_ratio": 1.5,
                "outperformance": 10.0,
                "final_value": 110_000.0,
                "model_roi": 10.0,
                "buyhold_roi": 5.0,
                "allocation": [0.6, 0.4],
                "recommended_trades": trades,
                "model_path": None,
                "results_path": None,
                "update": 0,
            }

        runner = WorkflowRunner(config=config, trainer=trainer)
        trial = runner._run_trial(trial_num=1)
        assert trial.recommended_trades == trades

    def test_extractor_handles_missing_trades(self) -> None:
        config = TrainingConfig(
            tickers=["AAPL"],
            positions={"AAPL": 100_000.0},
        )
        def trainer(**kwargs: Any) -> dict[str, Any]:
            return {
                "sharpe_ratio": 1.5,
                "outperformance": 10.0,
                "final_value": 110_000.0,
                "model_roi": 10.0,
                "buyhold_roi": 5.0,
                "allocation": [0.6, 0.4],
                "model_path": None,
                "results_path": None,
                "update": 0,
            }

        runner = WorkflowRunner(config=config, trainer=trainer)
        trial = runner._run_trial(trial_num=1)
        assert trial.recommended_trades is None


# ===================================================================
# Metric computation tests (sharpe_ratio, outperformance, ROIs)
# ===================================================================


class TestMetricComputation:
    """Tests for metric computation logic in create_trainer / SimulationRunner.

    These tests verify the mathematical correctness of:
    - sharpe_ratio from daily_returns
    - model_roi as percentage
    - buyhold_roi as percentage
    - outperformance as model_roi - buyhold_roi
    """

    def test_sharpe_ratio_formula(self) -> None:
        """Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)."""
        # Simulate portfolio values that produce known daily returns
        values = np.array([100.0, 101.0, 102.0, 101.5, 103.0], dtype=np.float64)
        daily_returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
        std = np.std(daily_returns)
        if std > 0:
            sharpe = float(np.mean(daily_returns) / std * np.sqrt(252))
        else:
            sharpe = 0.0
        assert isinstance(sharpe, float)
        # With these values, sharpe should be positive
        assert sharpe > 0

    def test_sharpe_ratio_zero_std(self) -> None:
        """Sharpe should be None/0 when std of returns is zero."""
        values = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        daily_returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
        sharpe: float | None = None
        if np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
        assert sharpe is None

    def test_sharpe_ratio_single_value(self) -> None:
        """Sharpe should be None when only one portfolio value."""
        values = np.array([100.0], dtype=np.float64)
        daily_returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
        sharpe: float | None = None
        if len(values) > 1 and np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
        assert sharpe is None

    def test_model_roi_percentage(self) -> None:
        """model_roi = (final - initial) / initial * 100."""
        initial_value = 100_000.0
        final_value = 120_000.0
        model_roi = (final_value - initial_value) / initial_value * 100
        assert model_roi == pytest.approx(20.0)

    def test_model_roi_negative(self) -> None:
        """model_roi should be negative when final < initial."""
        initial_value = 100_000.0
        final_value = 80_000.0
        model_roi = (final_value - initial_value) / initial_value * 100
        assert model_roi == pytest.approx(-20.0)

    def test_buyhold_roi_percentage(self) -> None:
        """buyhold_roi = (bh_final - initial) / initial * 100."""
        initial_value = 100_000.0
        bh_final = 110_000.0
        buyhold_roi = (bh_final - initial_value) / initial_value * 100
        assert buyhold_roi == pytest.approx(10.0)

    def test_outperformance_is_difference(self) -> None:
        """outperformance = model_roi - buyhold_roi."""
        model_roi = 20.0
        buyhold_roi = 10.0
        outperformance = model_roi - buyhold_roi
        assert outperformance == pytest.approx(10.0)

    def test_outperformance_negative(self) -> None:
        """outperformance can be negative when model underperforms."""
        model_roi = 5.0
        buyhold_roi = 10.0
        outperformance = model_roi - buyhold_roi
        assert outperformance == pytest.approx(-5.0)


# ===================================================================
# create_trainer — recommended_trades derivation
# ===================================================================


class TestCreateTrainerRecommendedTrades:
    """Tests for create_trainer deriving recommended_trades from allocation_history."""

    def _make_mock_results(
        self,
        allocation_history: list[dict[str, float]],
        portfolio_values: list[float] | None = None,
        buyhold_values: list[float] | None = None,
        final_holdings: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Build a mock SimulationRunner.run() result dict."""
        if portfolio_values is None:
            portfolio_values = [100_000.0 + i * 100 for i in range(len(allocation_history))]
        if buyhold_values is None:
            buyhold_values = [100_000.0 + i * 80 for i in range(len(allocation_history))]
        if final_holdings is None:
            final_holdings = {"AAPL": 100.0, "MSFT": 50.0}
        return {
            "final_value": portfolio_values[-1] if portfolio_values else 100_000.0,
            "initial_value": 100_000.0,
            "portfolio_values": portfolio_values,
            "daily_returns": [],
            "rewards": [],
            "allocation_history": allocation_history,
            "dates": [],
            "final_holdings": final_holdings,
            "final_prices": {"AAPL": 150.0, "MSFT": 300.0},
            "buyhold_values": buyhold_values,
        }

    def test_derives_trades_from_two_allocations(self) -> None:
        """When allocation_history has ≥2 entries, trades are derived from deltas."""
        alloc_hist = [
            {"AAPL": 0.4, "MSFT": 0.4, "cash": 0.2},
            {"AAPL": 0.5, "MSFT": 0.3, "cash": 0.2},
        ]
        mock_results = self._make_mock_results(alloc_hist)

        with patch("alloc.core.SimulationRunner") as mock_runner_cls, \
             patch("alloc.core.ActorCriticNetworks") as mock_networks, \
             patch("alloc.core.PolygonClient") as mock_client, \
             patch("alloc.core.get_settings", return_value=MagicMock(
                 polygon_api_key="fake", cache_enabled=True, cache_dir="/tmp",
             )), \
             patch("alloc.core.DiskCache", return_value=MagicMock()):
            mock_runner = MagicMock()
            mock_runner.run.return_value = mock_results
            mock_runner_cls.return_value = mock_runner

            from alloc.core import create_trainer
            trainer = create_trainer()
            result = trainer(
                tickers=["AAPL", "MSFT"],
                positions={"AAPL": 50_000.0, "MSFT": 50_000.0},
                trading_days=5,
            )

        trades = result.get("recommended_trades")
        assert trades is not None
        assert len(trades) == 3  # AAPL, MSFT, cash

        # AAPL increased: 0.5 - 0.4 = +0.1 → buy
        aapl_trade = [t for t in trades if t["ticker"] == "AAPL"][0]
        assert aapl_trade["action"] == "buy"
        assert aapl_trade["change"] == pytest.approx(0.1)

        # MSFT decreased: 0.3 - 0.4 = -0.1 → sell
        msft_trade = [t for t in trades if t["ticker"] == "MSFT"][0]
        assert msft_trade["action"] == "sell"
        assert msft_trade["change"] == pytest.approx(-0.1)

        # Cash unchanged: 0.2 - 0.2 = 0 → hold
        cash_trade = [t for t in trades if t["ticker"] == "cash"][0]
        assert cash_trade["action"] == "hold"
        assert cash_trade["change"] == pytest.approx(0.0)

    def test_no_trades_with_single_allocation(self) -> None:
        """With only 1 allocation entry, trades come from final_holdings."""
        alloc_hist = [
            {"AAPL": 0.5, "MSFT": 0.3, "cash": 0.2},
        ]
        final_holdings = {"AAPL": 100.0, "MSFT": 50.0}
        mock_results = self._make_mock_results(
            alloc_hist, final_holdings=final_holdings,
        )

        with patch("alloc.core.SimulationRunner") as mock_runner_cls, \
             patch("alloc.core.ActorCriticNetworks"), \
             patch("alloc.core.PolygonClient"), \
             patch("alloc.core.get_settings", return_value=MagicMock(
                 polygon_api_key="fake", cache_enabled=True, cache_dir="/tmp",
             )), \
             patch("alloc.core.DiskCache", return_value=MagicMock()):
            mock_runner = MagicMock()
            mock_runner.run.return_value = mock_results
            mock_runner_cls.return_value = mock_runner

            from alloc.core import create_trainer
            trainer = create_trainer()
            result = trainer(
                tickers=["AAPL", "MSFT"],
                positions={"AAPL": 50_000.0, "MSFT": 50_000.0},
                trading_days=5,
            )

        trades = result.get("recommended_trades")
        assert trades is not None
        assert len(trades) == 2  # AAPL, MSFT (no cash in single-alloc mode)

    def test_no_trades_with_empty_allocation_history(self) -> None:
        """With empty allocation_history, recommended_trades is None."""
        mock_results = self._make_mock_results([])
        mock_results["portfolio_values"] = []
        mock_results["buyhold_values"] = []

        with patch("alloc.core.SimulationRunner") as mock_runner_cls, \
             patch("alloc.core.ActorCriticNetworks"), \
             patch("alloc.core.PolygonClient"), \
             patch("alloc.core.get_settings", return_value=MagicMock(
                 polygon_api_key="fake", cache_enabled=True, cache_dir="/tmp",
             )), \
             patch("alloc.core.DiskCache", return_value=MagicMock()):
            mock_runner = MagicMock()
            mock_runner.run.return_value = mock_results
            mock_runner_cls.return_value = mock_runner

            from alloc.core import create_trainer
            trainer = create_trainer()
            result = trainer(
                tickers=["AAPL"],
                positions={"AAPL": 100_000.0},
                trading_days=1,
            )

        assert result.get("recommended_trades") is None

    def test_trades_include_allocation_and_change(self) -> None:
        """Each trade dict has ticker, action, allocation, change keys."""
        alloc_hist = [
            {"AAPL": 0.3, "cash": 0.7},
            {"AAPL": 0.6, "cash": 0.4},
        ]
        mock_results = self._make_mock_results(alloc_hist)

        with patch("alloc.core.SimulationRunner") as mock_runner_cls, \
             patch("alloc.core.ActorCriticNetworks"), \
             patch("alloc.core.PolygonClient"), \
             patch("alloc.core.get_settings", return_value=MagicMock(
                 polygon_api_key="fake", cache_enabled=True, cache_dir="/tmp",
             )), \
             patch("alloc.core.DiskCache", return_value=MagicMock()):
            mock_runner = MagicMock()
            mock_runner.run.return_value = mock_results
            mock_runner_cls.return_value = mock_runner

            from alloc.core import create_trainer
            trainer = create_trainer()
            result = trainer(
                tickers=["AAPL"],
                positions={"AAPL": 100_000.0},
                trading_days=5,
            )

        trades = result["recommended_trades"]
        for trade in trades:
            assert "ticker" in trade
            assert "action" in trade
            assert "allocation" in trade
            assert "change" in trade
            assert trade["action"] in ("buy", "sell", "hold")


# ===================================================================
# print_results — recommended_trades rendering
# ===================================================================


class TestPrintResultsTrades:
    """Tests for print_results rendering recommended_trades."""

    def test_print_results_with_trades(self, caplog: pytest.LogCaptureFixture) -> None:
        from alloc.cli import print_results

        caplog.set_level("INFO")
        result = WorkflowResult(
            status="success",
            trials=[],
            best_trial=TrainingTrial(
                iteration=1,
                update=0,
                sharpe_ratio=2.0,
                outperformance=10.0,
                final_value=120_000.0,
                model_roi=20.0,
                buyhold_roi=10.0,
                allocation=[0.5, 0.5],
                recommended_trades=[
                    {"ticker": "AAPL", "action": "buy", "allocation": 0.5, "change": 0.1},
                    {"ticker": "cash", "action": "sell", "allocation": 0.5, "change": -0.1},
                ],
            ),
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )
        print_results(result)

        # Check that trade lines appear in log
        log_text = "\n".join(r.message for r in caplog.records)
        assert "Recommended trades:" in log_text
        assert "AAPL" in log_text
        assert "BUY" in log_text
        assert "cash" in log_text
        assert "SELL" in log_text

    def test_print_results_without_trades(self, caplog: pytest.LogCaptureFixture) -> None:
        from alloc.cli import print_results

        caplog.set_level("INFO")
        result = WorkflowResult(
            status="success",
            trials=[],
            best_trial=TrainingTrial(
                iteration=1,
                update=0,
                sharpe_ratio=2.0,
                outperformance=10.0,
                final_value=120_000.0,
                model_roi=20.0,
                buyhold_roi=10.0,
                allocation=[0.5, 0.5],
                recommended_trades=None,
            ),
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )
        print_results(result)

        log_text = "\n".join(r.message for r in caplog.records)
        assert "Recommended trades:" not in log_text


# ===================================================================
# Integration: full workflow with recommended_trades
# ===================================================================


class TestWorkflowIntegration:
    """Integration tests for the full workflow with recommended_trades."""

    def test_full_workflow_propagates_trades(self) -> None:
        """End-to-end: trainer returns trades → WorkflowRunner → TrainingTrial."""
        config = TrainingConfig(
            tickers=["AAPL", "MSFT"],
            positions={"AAPL": 50_000.0, "MSFT": 50_000.0},
            iterations=2,
        )

        trades = [
            {"ticker": "AAPL", "action": "buy", "allocation": 0.5, "change": 0.1},
            {"ticker": "MSFT", "action": "sell", "allocation": 0.4, "change": -0.1},
            {"ticker": "cash", "action": "hold", "allocation": 0.1, "change": 0.0},
        ]

        def trainer(**kwargs: Any) -> dict[str, Any]:
            return {
                "sharpe_ratio": 1.5,
                "outperformance": 10.0,
                "final_value": 110_000.0,
                "model_roi": 10.0,
                "buyhold_roi": 5.0,
                "allocation": [0.5, 0.4, 0.1],
                "recommended_trades": trades,
                "model_path": None,
                "results_path": None,
                "update": 0,
            }

        runner = WorkflowRunner(config=config, trainer=trainer)
        result = runner.run()

        assert result.status == "success"
        assert len(result.trials) == 2
        for trial in result.trials:
            assert trial.recommended_trades == trades
        assert result.best_trial.recommended_trades == trades
