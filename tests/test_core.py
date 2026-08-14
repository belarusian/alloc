"""Tests for alloc.core — SimulationRunner, CLI, and serialization helpers.

All tests mock external dependencies (PolygonClient, ActorCriticNetworks) to
avoid real API calls and GPU requirements.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Serialization helpers (TICKET-012)
# ---------------------------------------------------------------------------


class TestSerializeResults:
    """Tests for serialize_results."""

    def test_numpy_arrays_converted_to_lists(self) -> None:
        from alloc.core import serialize_results

        results = {
            "portfolio_values": np.array([100.0, 101.0, 102.0]),
            "daily_returns": np.array([0.01, 0.009]),
            "final_value": 102.0,
        }
        serialized = serialize_results(results)
        assert isinstance(serialized["portfolio_values"], list)
        assert isinstance(serialized["daily_returns"], list)
        assert serialized["portfolio_values"] == [100.0, 101.0, 102.0]
        assert serialized["daily_returns"] == [0.01, 0.009]

    def test_nested_dict_handled_recursively(self) -> None:
        from alloc.core import serialize_results

        results = {
            "data": {
                "values": np.array([1.0, 2.0]),
                "nested": {
                    "inner": np.array([3.0, 4.0]),
                },
            },
            "scalar": 42,
        }
        serialized = serialize_results(results)
        assert isinstance(serialized["data"]["values"], list)
        assert isinstance(serialized["data"]["nested"]["inner"], list)
        assert serialized["data"]["nested"]["inner"] == [3.0, 4.0]

    def test_non_numpy_values_unchanged(self) -> None:
        from alloc.core import serialize_results

        results = {
            "final_value": 102.0,
            "dates": ["2024-01-01", "2024-01-02"],
            "tickers": ["AAPL", "MSFT"],
        }
        serialized = serialize_results(results)
        assert serialized["final_value"] == 102.0
        assert serialized["dates"] == ["2024-01-01", "2024-01-02"]
        assert serialized["tickers"] == ["AAPL", "MSFT"]

    def test_empty_dict(self) -> None:
        from alloc.core import serialize_results

        assert serialize_results({}) == {}

    def test_list_of_numpy_arrays(self) -> None:
        from alloc.core import serialize_results

        results = {
            "allocation_history": [
                np.array([0.5, 0.5]),
                np.array([0.6, 0.4]),
            ],
        }
        serialized = serialize_results(results)
        assert isinstance(serialized["allocation_history"], list)
        assert isinstance(serialized["allocation_history"][0], list)
        assert serialized["allocation_history"][0] == [0.5, 0.5]

    def test_np_floating_converted(self) -> None:
        from alloc.core import serialize_results

        results = {"value": np.float64(3.14)}
        serialized = serialize_results(results)
        assert isinstance(serialized["value"], float)
        assert serialized["value"] == 3.14

    def test_np_integer_converted(self) -> None:
        from alloc.core import serialize_results

        results = {"count": np.int64(42)}
        serialized = serialize_results(results)
        assert isinstance(serialized["count"], float)
        assert serialized["count"] == 42.0


class TestSaveResults:
    """Tests for save_results."""

    def test_save_results_creates_file(self, tmp_path: Path) -> None:
        from alloc.core import save_results

        results = {"final_value": 102.0, "portfolio_values": [100.0, 101.0]}
        save_results(results, str(tmp_path), mode="backtest")
        assert (tmp_path / "backtest_results.json").exists()

    def test_save_results_backtest_mode(self, tmp_path: Path) -> None:
        from alloc.core import save_results

        results = {"final_value": 102.0}
        save_results(results, str(tmp_path), mode="backtest")
        with open(tmp_path / "backtest_results.json") as f:
            data = json.load(f)
        assert data["final_value"] == 102.0

    def test_save_results_predict_mode(self, tmp_path: Path) -> None:
        from alloc.core import save_results

        results = {"final_value": 102.0}
        save_results(results, str(tmp_path), mode="predict")
        assert (tmp_path / "prediction_results.json").exists()
        with open(tmp_path / "prediction_results.json") as f:
            data = json.load(f)
        assert data["final_value"] == 102.0

    def test_save_results_creates_parent_dirs(self, tmp_path: Path) -> None:
        from alloc.core import save_results

        nested = tmp_path / "sub" / "dir"
        results = {"final_value": 102.0}
        save_results(results, str(nested), mode="backtest")
        assert (nested / "backtest_results.json").exists()

    def test_save_results_serializes_numpy(self, tmp_path: Path) -> None:
        from alloc.core import save_results

        results = {"values": np.array([1.0, 2.0, 3.0])}
        save_results(results, str(tmp_path), mode="backtest")
        with open(tmp_path / "backtest_results.json") as f:
            data = json.load(f)
        assert data["values"] == [1.0, 2.0, 3.0]


class TestLoadResults:
    """Tests for load_results."""

    def test_load_results_roundtrip(self, tmp_path: Path) -> None:
        from alloc.core import load_results, save_results

        results = {
            "final_value": 102.0,
            "portfolio_values": [100.0, 101.0, 102.0],
            "dates": ["2024-01-01", "2024-01-02"],
        }
        save_results(results, str(tmp_path), mode="backtest")
        loaded = load_results(str(tmp_path))
        assert loaded["final_value"] == 102.0
        assert loaded["portfolio_values"] == [100.0, 101.0, 102.0]
        assert loaded["dates"] == ["2024-01-01", "2024-01-02"]

    def test_load_results_predict_mode(self, tmp_path: Path) -> None:
        from alloc.core import load_results, save_results

        results = {"prediction": [1.0, 2.0]}
        save_results(results, str(tmp_path), mode="predict")
        loaded = load_results(str(tmp_path), mode="predict")
        assert loaded["prediction"] == [1.0, 2.0]

    def test_load_results_missing_file_raises(self, tmp_path: Path) -> None:
        from alloc.core import load_results

        with pytest.raises(FileNotFoundError):
            load_results(str(tmp_path))


# ---------------------------------------------------------------------------
# CLI argument parsing (TICKET-010)
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for parse_args."""

    def test_backtest_mode(self) -> None:
        from alloc.core import parse_args

        args = parse_args(["--backtest", "--tickers", "AAPL", "MSFT"])
        assert args.backtest is True
        assert args.predict is False
        assert args.tickers == ["AAPL", "MSFT"]

    def test_predict_mode(self) -> None:
        from alloc.core import parse_args

        args = parse_args(["--predict", "--tickers", "AAPL"])
        assert args.predict is True
        assert args.backtest is False

    def test_mutually_exclusive_modes(self) -> None:
        from alloc.core import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--backtest", "--predict", "--tickers", "AAPL"])

    def test_neither_mode_raises(self) -> None:
        from alloc.core import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--tickers", "AAPL"])

    def test_default_values(self) -> None:
        from alloc.core import parse_args

        args = parse_args(["--backtest", "--tickers", "AAPL"])
        assert args.initial_value == 100_000.0
        assert args.trading_days == 242
        assert args.model_path == "saved_models/portfolio_model"
        assert args.actor_lr == 0.0001
        assert args.critic_lr == 0.0005
        assert args.gamma == 0.95
        assert args.tau == 0.01
        assert args.risk_aversion == 0.5
        assert args.transaction_cost == 0.001
        assert args.diversification_weight == 0.05
        assert args.concentration_penalty == 0.02
        assert args.min_cash == 0.05
        assert args.batch_size == 32
        assert args.replay_capacity == 50_000
        assert args.verbose is False

    def test_custom_values(self) -> None:
        from alloc.core import parse_args

        args = parse_args([
            "--backtest",
            "--tickers", "AAPL", "GOOGL",
            "--initial-value", "50000",
            "--trading-days", "100",
            "--model-path", "/tmp/model",
            "--actor-lr", "0.001",
            "--critic-lr", "0.001",
            "--gamma", "0.99",
            "--tau", "0.05",
            "--risk-aversion", "1.0",
            "--transaction-cost", "0.005",
            "--diversification-weight", "0.1",
            "--concentration-penalty", "0.05",
            "--min-cash", "0.1",
            "--batch-size", "64",
            "--replay-capacity", "100000",
            "--verbose",
        ])
        assert args.initial_value == 50_000.0
        assert args.trading_days == 100
        assert args.model_path == "/tmp/model"
        assert args.actor_lr == 0.001
        assert args.critic_lr == 0.001
        assert args.gamma == 0.99
        assert args.tau == 0.05
        assert args.risk_aversion == 1.0
        assert args.transaction_cost == 0.005
        assert args.diversification_weight == 0.1
        assert args.concentration_penalty == 0.05
        assert args.min_cash == 0.1
        assert args.batch_size == 64
        assert args.replay_capacity == 100_000
        assert args.verbose is True


# ---------------------------------------------------------------------------
# SimulationRunner (TICKET-009)
# ---------------------------------------------------------------------------


class TestSimulationRunnerInit:
    """Tests for SimulationRunner.__init__."""

    def test_constructor_stores_parameters(self) -> None:
        from alloc.core import SimulationRunner

        mock_networks = MagicMock()
        mock_client = MagicMock()
        mock_data_pipeline = MagicMock()

        runner = SimulationRunner(
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
            transaction_cost=0.002,
            risk_aversion=0.7,
            gamma=0.99,
            tau=0.02,
            diversification_weight=0.1,
            concentration_penalty=0.05,
            min_cash=0.1,
            batch_size=64,
            verbose=True,
        )
        assert runner.tickers == ["AAPL", "MSFT"]
        assert runner.initial_value == 100_000.0
        assert runner.networks is mock_networks
        assert runner.data_pipeline is mock_data_pipeline
        assert runner.client is mock_client
        assert runner.transaction_cost == 0.002
        assert runner.risk_aversion == 0.7
        assert runner.gamma == 0.99
        assert runner.tau == 0.02
        assert runner.diversification_weight == 0.1
        assert runner.concentration_penalty == 0.05
        assert runner.min_cash == 0.1
        assert runner.batch_size == 64
        assert runner.verbose is True

    def test_default_parameters(self) -> None:
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL"],
            initial_value=50_000.0,
            networks=MagicMock(),
            data_pipeline=MagicMock(),
            client=MagicMock(),
        )
        assert runner.transaction_cost == 0.001
        assert runner.risk_aversion == 0.5
        assert runner.gamma == 0.95
        assert runner.tau == 0.01
        assert runner.diversification_weight == 0.05
        assert runner.concentration_penalty == 0.02
        assert runner.min_cash == 0.05
        assert runner.batch_size == 32
        assert runner.verbose is False


class TestSimulationRunnerRun:
    """Tests for SimulationRunner.run()."""

    @pytest.fixture()
    def mock_client(self) -> MagicMock:
        client = MagicMock()
        client.get_last_trade.return_value = SimpleNamespace(price=150.0)
        client.get_aggs.return_value = [
            SimpleNamespace(close=100.0 + i * 0.1) for i in range(10)
        ]
        return client

    @pytest.fixture()
    def mock_networks(self) -> MagicMock:
        networks = MagicMock()
        networks.get_allocation.return_value = np.array([0.4, 0.4, 0.2])
        networks.actor = MagicMock()
        networks.critic = MagicMock()
        networks.actor_target = MagicMock()
        networks.critic_target = MagicMock()
        networks.replay_buffer = MagicMock()
        networks.input_dim = 10
        networks.num_assets = 3
        return networks

    @pytest.fixture()
    def mock_data_pipeline(self) -> MagicMock:
        dp = MagicMock()
        dp.get_multi_asset_data.return_value = {
            "AAPL": {"hourly": [100.0] * 5, "daily": [100.0] * 5, "weekly": [100.0] * 5},
            "MSFT": {"hourly": [200.0] * 5, "daily": [200.0] * 5, "weekly": [200.0] * 5},
        }
        dp.build_state_vector.return_value = np.random.randn(10).astype(np.float32)
        dp.fetch_latest_prices.return_value = {"AAPL": 150.0, "MSFT": 300.0}
        return dp

    def test_run_returns_dict_with_expected_keys(
        self, mock_client, mock_networks, mock_data_pipeline
    ) -> None:
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
        )
        results = runner.run(trading_days=3)
        expected_keys = {
            "final_value",
            "initial_value",
            "portfolio_values",
            "daily_returns",
            "rewards",
            "allocation_history",
            "dates",
            "final_holdings",
            "final_prices",
        }
        assert expected_keys.issubset(set(results.keys()))

    def test_run_tracks_portfolio_values(
        self, mock_client, mock_networks, mock_data_pipeline
    ) -> None:
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
        )
        results = runner.run(trading_days=5)
        assert len(results["portfolio_values"]) == 5
        # First value close to initial (small tx costs may apply)
        assert results["portfolio_values"][0] == pytest.approx(
            100_000.0, abs=100.0
        )

    def test_run_tracks_daily_returns(
        self, mock_client, mock_networks, mock_data_pipeline
    ) -> None:
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
        )
        results = runner.run(trading_days=5)
        assert len(results["daily_returns"]) == 5

    def test_run_tracks_rewards(
        self, mock_client, mock_networks, mock_data_pipeline
    ) -> None:
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
        )
        results = runner.run(trading_days=5)
        assert len(results["rewards"]) == 5

    def test_run_tracks_allocation_history(
        self, mock_client, mock_networks, mock_data_pipeline
    ) -> None:
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
        )
        results = runner.run(trading_days=5)
        assert len(results["allocation_history"]) == 5

    def test_run_tracks_dates(
        self, mock_client, mock_networks, mock_data_pipeline
    ) -> None:
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
        )
        results = runner.run(trading_days=5)
        assert len(results["dates"]) == 5

    def test_run_returns_final_holdings(
        self, mock_client, mock_networks, mock_data_pipeline
    ) -> None:
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
        )
        results = runner.run(trading_days=3)
        assert "AAPL" in results["final_holdings"]
        assert "MSFT" in results["final_holdings"]

    def test_run_returns_final_prices(
        self, mock_client, mock_networks, mock_data_pipeline
    ) -> None:
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
        )
        results = runner.run(trading_days=3)
        assert "AAPL" in results["final_prices"]
        assert "MSFT" in results["final_prices"]

    def test_run_initial_value_stored(
        self, mock_client, mock_networks, mock_data_pipeline
    ) -> None:
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
        )
        results = runner.run(trading_days=3)
        assert results["initial_value"] == 100_000.0

    def test_run_uses_networks_get_allocation(
        self, mock_client, mock_networks, mock_data_pipeline
    ) -> None:
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
        )
        runner.run(trading_days=3)
        assert mock_networks.get_allocation.called

    def test_run_adds_noise_during_exploration(
        self, mock_client, mock_networks, mock_data_pipeline
    ) -> None:
        """First half of trading days should use exploration noise."""
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
        )
        runner.run(trading_days=10)
        # Should have called with add_noise=True at least once
        calls_with_noise = [
            c for c in mock_networks.get_allocation.call_args_list
            if c[1].get("add_noise", False)
        ]
        assert len(calls_with_noise) > 0

    def test_run_no_noise_during_exploitation(
        self, mock_client, mock_networks, mock_data_pipeline
    ) -> None:
        """Second half of trading days should not use exploration noise."""
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
        )
        runner.run(trading_days=10)
        # Last 5 calls should have add_noise=False
        last_five = mock_networks.get_allocation.call_args_list[-5:]
        for call in last_five:
            assert call[1].get("add_noise", False) is False

    def test_run_updates_replay_buffer(
        self, mock_client, mock_networks, mock_data_pipeline
    ) -> None:
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
        )
        runner.run(trading_days=5)
        # replay_buffer.add should have been called
        assert mock_networks.replay_buffer.add.called

    def test_run_single_day(self, mock_client, mock_networks, mock_data_pipeline) -> None:
        from alloc.core import SimulationRunner

        runner = SimulationRunner(
            tickers=["AAPL"],
            initial_value=50_000.0,
            networks=mock_networks,
            data_pipeline=mock_data_pipeline,
            client=mock_client,
        )
        mock_networks.get_allocation.return_value = np.array([0.6, 0.4])
        mock_networks.num_assets = 2
        mock_networks.input_dim = 10
        mock_data_pipeline.build_state_vector.return_value = np.random.randn(10).astype(np.float32)
        mock_data_pipeline.get_multi_asset_data.return_value = {
            "AAPL": {"hourly": [100.0] * 5, "daily": [100.0] * 5, "weekly": [100.0] * 5},
        }
        mock_data_pipeline.fetch_latest_prices.return_value = {"AAPL": 150.0}
        results = runner.run(trading_days=1)
        assert len(results["portfolio_values"]) == 1
        assert len(results["dates"]) == 1


# ---------------------------------------------------------------------------
# main() integration (TICKET-010)
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main() — integration level."""

    def test_main_calls_runner_and_saves_results(self, tmp_path: Path) -> None:
        from alloc.core import main

        model_path = tmp_path / "model"

        mock_networks = MagicMock()
        mock_networks.get_allocation.return_value = np.array([0.4, 0.4, 0.2])
        mock_networks.actor = MagicMock()
        mock_networks.critic = MagicMock()
        mock_networks.actor_target = MagicMock()
        mock_networks.critic_target = MagicMock()
        mock_networks.replay_buffer = MagicMock()
        mock_networks.input_dim = 10
        mock_networks.num_assets = 3

        mock_client = MagicMock()
        mock_client.get_last_trade.return_value = SimpleNamespace(price=150.0)
        mock_client.get_aggs.return_value = [
            SimpleNamespace(close=100.0 + i * 0.1) for i in range(10)
        ]

        mock_data_pipeline = MagicMock()
        mock_data_pipeline.get_multi_asset_data.return_value = {
            "AAPL": {"hourly": [100.0] * 5, "daily": [100.0] * 5, "weekly": [100.0] * 5},
            "MSFT": {"hourly": [200.0] * 5, "daily": [200.0] * 5, "weekly": [200.0] * 5},
        }
        mock_data_pipeline.build_state_vector.return_value = np.random.randn(10).astype(np.float32)
        mock_data_pipeline.fetch_latest_prices.return_value = {"AAPL": 150.0, "MSFT": 300.0}

        with patch("alloc.core.parse_args", return_value=argparse.Namespace(
            backtest=True,
            predict=False,
            tickers=["AAPL", "MSFT"],
            initial_value=100_000.0,
            trading_days=2,
            model_path=str(model_path),
            actor_lr=0.0001,
            critic_lr=0.0005,
            gamma=0.95,
            tau=0.01,
            risk_aversion=0.5,
            transaction_cost=0.001,
            diversification_weight=0.05,
            concentration_penalty=0.02,
            min_cash=0.05,
            batch_size=32,
            replay_capacity=50_000,
            verbose=False,
        )), patch("alloc.core.ActorCriticNetworks", return_value=mock_networks), \
             patch("alloc.core.PolygonClient", return_value=mock_client), \
             patch("alloc.core.get_settings", return_value=MagicMock(
                 polygon_api_key="fake-key",
                 cache_enabled=True,
                 cache_dir=tmp_path,
             )), patch("alloc.core.DiskCache", return_value=MagicMock()):
            main()

        # Results file should be created inside model_path
        assert (model_path / "backtest_results.json").exists()
