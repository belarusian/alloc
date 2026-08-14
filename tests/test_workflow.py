"""Tests for alloc.utils.workflow — TrainingConfig, TrainingTrial, WorkflowRunner."""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from alloc.utils.workflow import (
    TrainingConfig,
    TrainingTrial,
    WorkflowResult,
    WorkflowRunner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config() -> TrainingConfig:
    return TrainingConfig(
        tickers=["AAPL", "GOOGL"],
        positions={"AAPL": 50_000.0, "GOOGL": 50_000.0},
        iterations=3,
        update_iterations=2,
        trading_days=222,
        batch_size=22,
        min_allocation=0.001,
        concentration_penalty=0.001,
        transaction_cost=0.0,
        risk_aversion=0.001,
        min_cash_alloc=0.05,
        target_sharpe=2.1,
        target_value=220_000.0,
        target_outperformance=15.0,
    )


@pytest.fixture
def mock_trainer() -> Any:
    """Return a callable that simulates a trainer returning deterministic results."""
    call_count = 0

    def trainer(**kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {
            "sharpe_ratio": 1.5 + call_count * 0.1,
            "outperformance": 12.0 + call_count * 0.5,
            "final_value": 200_000.0 + call_count * 1_000,
            "model_roi": 10.0 + call_count * 0.5,
            "buyhold_roi": 8.0 + call_count * 0.2,
            "allocation": [0.45, 0.45, 0.10],  # AAPL, GOOGL, cash
            "model_path": f"/tmp/model_{call_count}.pt",
            "results_path": f"/tmp/results_{call_count}.json",
            "update": call_count,
        }

    trainer.call_count = call_count  # type: ignore[attr-defined]
    return trainer


# ---------------------------------------------------------------------------
# TrainingConfig tests
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_default_values(self) -> None:
        cfg = TrainingConfig(tickers=["SPY"], positions={"SPY": 100_000.0})
        assert cfg.tickers == ["SPY"]
        assert cfg.positions == {"SPY": 100_000.0}
        assert cfg.iterations == 1
        assert cfg.update_iterations == 1
        assert cfg.trading_days == 222
        assert cfg.batch_size == 22
        assert cfg.min_allocation == 0.001
        assert cfg.concentration_penalty == 0.001
        assert cfg.transaction_cost == 0.0
        assert cfg.risk_aversion == 0.001
        assert cfg.min_cash_alloc == 0.05
        assert cfg.target_sharpe == 2.1
        assert cfg.target_value == 220_000.0
        assert cfg.target_outperformance == 15.0

    def test_custom_values(self) -> None:
        cfg = TrainingConfig(
            tickers=["TSLA", "NVDA"],
            positions={"TSLA": 30_000.0, "NVDA": 70_000.0},
            iterations=10,
            update_iterations=5,
            trading_days=252,
            batch_size=32,
            min_allocation=0.01,
            concentration_penalty=0.01,
            transaction_cost=0.001,
            risk_aversion=0.01,
            min_cash_alloc=0.1,
            target_sharpe=3.0,
            target_value=300_000.0,
            target_outperformance=20.0,
        )
        assert cfg.iterations == 10
        assert cfg.update_iterations == 5
        assert cfg.trading_days == 252
        assert cfg.batch_size == 32
        assert cfg.min_allocation == 0.01
        assert cfg.concentration_penalty == 0.01
        assert cfg.transaction_cost == 0.001
        assert cfg.risk_aversion == 0.01
        assert cfg.min_cash_alloc == 0.1
        assert cfg.target_sharpe == 3.0
        assert cfg.target_value == 300_000.0
        assert cfg.target_outperformance == 20.0

    def test_is_dataclass(self) -> None:
        cfg = TrainingConfig(tickers=["X"], positions={"X": 1.0})
        d = asdict(cfg)
        assert isinstance(d, dict)
        assert "tickers" in d
        assert "positions" in d


# ---------------------------------------------------------------------------
# TrainingTrial tests
# ---------------------------------------------------------------------------

class TestTrainingTrial:
    def test_minimal_creation(self) -> None:
        trial = TrainingTrial(iteration=1, update=0)
        assert trial.iteration == 1
        assert trial.update == 0
        assert trial.sharpe_ratio is None
        assert trial.outperformance is None
        assert trial.final_value is None
        assert trial.model_roi is None
        assert trial.buyhold_roi is None
        assert trial.allocation == []
        assert trial.model_path is None
        assert trial.results_path is None

    def test_full_creation(self) -> None:
        trial = TrainingTrial(
            iteration=5,
            update=3,
            sharpe_ratio=2.5,
            outperformance=18.0,
            final_value=250_000.0,
            model_roi=15.0,
            buyhold_roi=10.0,
            allocation=[0.5, 0.4, 0.1],
            model_path="/tmp/model.pt",
            results_path="/tmp/results.json",
        )
        assert trial.sharpe_ratio == 2.5
        assert trial.outperformance == 18.0
        assert trial.final_value == 250_000.0
        assert trial.model_roi == 15.0
        assert trial.buyhold_roi == 10.0
        assert trial.allocation == [0.5, 0.4, 0.1]
        assert trial.model_path == "/tmp/model.pt"
        assert trial.results_path == "/tmp/results.json"


# ---------------------------------------------------------------------------
# WorkflowResult tests
# ---------------------------------------------------------------------------

class TestWorkflowResult:
    def test_creation(self) -> None:
        trial = TrainingTrial(iteration=1, update=0)
        result = WorkflowResult(
            status="success",
            trials=[trial],
            best_trial=trial,
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )
        assert result.status == "success"
        assert len(result.trials) == 1
        assert result.best_trial is trial
        assert result.allocation_stats == {}
        assert result.concentration == {}
        assert result.metrics_progression == []


# ---------------------------------------------------------------------------
# WorkflowRunner tests
# ---------------------------------------------------------------------------

class TestWorkflowRunner:
    def test_init(self, sample_config: TrainingConfig, mock_trainer: Any) -> None:
        runner = WorkflowRunner(config=sample_config, trainer=mock_trainer)
        assert runner.config is sample_config
        assert runner.trainer is mock_trainer

    def test_run_single_trial(self, sample_config: TrainingConfig, mock_trainer: Any) -> None:
        sample_config.iterations = 1
        runner = WorkflowRunner(config=sample_config, trainer=mock_trainer)
        result = runner.run()

        assert result.status == "success"
        assert len(result.trials) == 1
        assert result.best_trial.iteration == 1
        assert len(result.metrics_progression) == 1

    def test_run_multiple_trials(self, sample_config: TrainingConfig, mock_trainer: Any) -> None:
        sample_config.iterations = 3
        runner = WorkflowRunner(config=sample_config, trainer=mock_trainer)
        result = runner.run()

        assert result.status == "success"
        assert len(result.trials) == 3
        assert len(result.metrics_progression) == 3
        assert result.trials[0].iteration == 1
        assert result.trials[1].iteration == 2
        assert result.trials[2].iteration == 3

    def test_run_zero_iterations(self, sample_config: TrainingConfig, mock_trainer: Any) -> None:
        sample_config.iterations = 0
        runner = WorkflowRunner(config=sample_config, trainer=mock_trainer)
        result = runner.run()

        assert result.status == "error"
        assert result.trials == []
        assert result.best_trial.iteration == 0
        assert result.allocation_stats == {}
        assert result.concentration == {}
        assert result.metrics_progression == []

    def test_best_trial_selection(self, sample_config: TrainingConfig) -> None:
        """The trial with the highest combined score should be selected."""
        call_count = 0

        def trainer(**kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            # Trial 2 has the best combined score
            if call_count == 2:
                return {
                    "sharpe_ratio": 3.0,
                    "outperformance": 20.0,
                    "final_value": 300_000.0,
                    "model_roi": 20.0,
                    "buyhold_roi": 10.0,
                    "allocation": [0.5, 0.4, 0.1],
                    "model_path": "/tmp/m2.pt",
                    "results_path": "/tmp/r2.json",
                    "update": 2,
                }
            return {
                "sharpe_ratio": 1.0,
                "outperformance": 5.0,
                "final_value": 100_000.0,
                "model_roi": 5.0,
                "buyhold_roi": 4.0,
                "allocation": [0.3, 0.3, 0.4],
                "model_path": f"/tmp/m{call_count}.pt",
                "results_path": f"/tmp/r{call_count}.json",
                "update": call_count,
            }

        sample_config.iterations = 3
        runner = WorkflowRunner(config=sample_config, trainer=trainer)
        result = runner.run()

        assert result.best_trial.iteration == 2
        assert result.best_trial.sharpe_ratio == 3.0

    def test_trainer_receives_config_kwargs(self, sample_config: TrainingConfig) -> None:
        received_kwargs: dict[str, Any] | None = None

        def capturing_trainer(**kwargs: Any) -> dict[str, Any]:
            nonlocal received_kwargs
            received_kwargs = copy.deepcopy(kwargs)
            return {
                "sharpe_ratio": 1.0,
                "outperformance": 10.0,
                "final_value": 100_000.0,
                "model_roi": 5.0,
                "buyhold_roi": 4.0,
                "allocation": [0.5, 0.5],
                "model_path": "/tmp/m.pt",
                "results_path": "/tmp/r.json",
                "update": 1,
            }

        sample_config.iterations = 1
        runner = WorkflowRunner(config=sample_config, trainer=capturing_trainer)
        runner.run()

        assert received_kwargs is not None
        assert received_kwargs["tickers"] == ["AAPL", "GOOGL"]
        assert received_kwargs["positions"] == {"AAPL": 50_000.0, "GOOGL": 50_000.0}
        assert received_kwargs["trading_days"] == 222
        assert received_kwargs["batch_size"] == 22
        assert received_kwargs["min_allocation"] == 0.001
        assert received_kwargs["concentration_penalty"] == 0.001
        assert received_kwargs["transaction_cost"] == 0.0
        assert received_kwargs["risk_aversion"] == 0.001
        assert received_kwargs["min_cash_alloc"] == 0.05
        assert received_kwargs["target_sharpe"] == 2.1
        assert received_kwargs["target_value"] == 220_000.0
        assert received_kwargs["target_outperformance"] == 15.0

    def test_trainer_missing_optional_fields(self, sample_config: TrainingConfig) -> None:
        """Trainer that returns minimal dict should not crash."""
        def sparse_trainer(**kwargs: Any) -> dict[str, Any]:
            return {"sharpe_ratio": 1.0}

        sample_config.iterations = 1
        runner = WorkflowRunner(config=sample_config, trainer=sparse_trainer)
        result = runner.run()

        assert result.status == "success"
        assert result.trials[0].sharpe_ratio == 1.0
        assert result.trials[0].outperformance is None
        assert result.trials[0].allocation == []

    def test_metrics_progression_content(self, sample_config: TrainingConfig, mock_trainer: Any) -> None:
        sample_config.iterations = 2
        runner = WorkflowRunner(config=sample_config, trainer=mock_trainer)
        result = runner.run()

        assert len(result.metrics_progression) == 2
        assert result.metrics_progression[0]["iteration"] == 1
        assert result.metrics_progression[1]["iteration"] == 2
        assert "sharpe_ratio" in result.metrics_progression[0]
        assert "outperformance" in result.metrics_progression[0]
        assert "final_value" in result.metrics_progression[0]
        assert "model_roi" in result.metrics_progression[0]
        assert "buyhold_roi" in result.metrics_progression[0]


# ---------------------------------------------------------------------------
# _combined_score tests
# ---------------------------------------------------------------------------

class TestCombinedScore:
    def test_basic_score(self) -> None:
        trial = TrainingTrial(
            iteration=1, update=0, sharpe_ratio=2.0, outperformance=10.0
        )
        score = WorkflowRunner._combined_score(trial)
        expected = 0.5 * 2.0 + 0.5 * (10.0 / 10.0)
        assert score == pytest.approx(expected)

    def test_none_values_default_to_zero(self) -> None:
        trial = TrainingTrial(iteration=1, update=0)
        score = WorkflowRunner._combined_score(trial)
        assert score == pytest.approx(0.0)

    def test_sharpe_only(self) -> None:
        trial = TrainingTrial(iteration=1, update=0, sharpe_ratio=3.0)
        score = WorkflowRunner._combined_score(trial)
        expected = 0.5 * 3.0 + 0.5 * 0.0
        assert score == pytest.approx(expected)

    def test_outperformance_only(self) -> None:
        trial = TrainingTrial(iteration=1, update=0, outperformance=20.0)
        score = WorkflowRunner._combined_score(trial)
        expected = 0.5 * 0.0 + 0.5 * (20.0 / 10.0)
        assert score == pytest.approx(expected)

    def test_negative_values(self) -> None:
        trial = TrainingTrial(
            iteration=1, update=0, sharpe_ratio=-1.0, outperformance=-5.0
        )
        score = WorkflowRunner._combined_score(trial)
        expected = 0.5 * (-1.0) + 0.5 * (-5.0 / 10.0)
        assert score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _allocation_stats tests
# ---------------------------------------------------------------------------

class TestAllocationStats:
    def test_single_trial(self) -> None:
        trials = [
            TrainingTrial(iteration=1, update=0, allocation=[0.5, 0.3, 0.2]),
        ]
        stats = WorkflowRunner._allocation_stats(trials)
        assert "position_0" in stats
        assert stats["position_0"]["mean"] == pytest.approx(0.5)
        assert stats["position_0"]["std"] == pytest.approx(0.0)
        assert stats["position_0"]["min"] == pytest.approx(0.5)
        assert stats["position_0"]["max"] == pytest.approx(0.5)

    def test_multiple_trials(self) -> None:
        trials = [
            TrainingTrial(iteration=1, update=0, allocation=[0.5, 0.3, 0.2]),
            TrainingTrial(iteration=2, update=0, allocation=[0.6, 0.2, 0.2]),
        ]
        stats = WorkflowRunner._allocation_stats(trials)
        assert stats["position_0"]["mean"] == pytest.approx(0.55)
        assert stats["position_0"]["min"] == pytest.approx(0.5)
        assert stats["position_0"]["max"] == pytest.approx(0.6)

    def test_empty_trials(self) -> None:
        stats = WorkflowRunner._allocation_stats([])
        assert stats == {}

    def test_trials_without_allocations(self) -> None:
        trials = [
            TrainingTrial(iteration=1, update=0, allocation=[]),
            TrainingTrial(iteration=2, update=0, allocation=[]),
        ]
        stats = WorkflowRunner._allocation_stats(trials)
        assert stats == {}

    def test_varying_allocation_lengths(self) -> None:
        """Handle trials with different allocation vector lengths."""
        trials = [
            TrainingTrial(iteration=1, update=0, allocation=[0.5, 0.3, 0.2]),
            TrainingTrial(iteration=2, update=0, allocation=[0.6, 0.4]),
        ]
        stats = WorkflowRunner._allocation_stats(trials)
        # position_0 has both values
        assert stats["position_0"]["mean"] == pytest.approx(0.55)
        # position_1 has both values
        assert stats["position_1"]["mean"] == pytest.approx(0.35)
        # position_2 only has one value (from trial 1)
        assert stats["position_2"]["mean"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# _concentration_metrics tests
# ---------------------------------------------------------------------------

class TestConcentrationMetrics:
    def test_single_trial_uniform(self) -> None:
        trials = [
            TrainingTrial(iteration=1, update=0, allocation=[0.25, 0.25, 0.25, 0.25]),
        ]
        metrics = WorkflowRunner._concentration_metrics(trials)
        assert metrics["max_weight"] == pytest.approx(0.25)
        # HHI = 0.25^2 * 4 = 0.25
        assert metrics["herfindahl"] == pytest.approx(0.25)

    def test_concentrated_portfolio(self) -> None:
        trials = [
            TrainingTrial(iteration=1, update=0, allocation=[0.9, 0.05, 0.05]),
        ]
        metrics = WorkflowRunner._concentration_metrics(trials)
        assert metrics["max_weight"] == pytest.approx(0.9)
        # HHI = 0.9^2 + 0.05^2 + 0.05^2 = 0.81 + 0.0025 + 0.0025 = 0.815
        assert metrics["herfindahl"] == pytest.approx(0.815)

    def test_multiple_trials(self) -> None:
        trials = [
            TrainingTrial(iteration=1, update=0, allocation=[0.5, 0.5]),
            TrainingTrial(iteration=2, update=0, allocation=[0.7, 0.3]),
        ]
        metrics = WorkflowRunner._concentration_metrics(trials)
        assert metrics["max_weight"] == pytest.approx(0.7)
        # HHI trial1 = 0.25 + 0.25 = 0.5
        # HHI trial2 = 0.49 + 0.09 = 0.58
        # Mean HHI = (0.5 + 0.58) / 2 = 0.54
        assert metrics["herfindahl"] == pytest.approx(0.54)

    def test_empty_trials(self) -> None:
        metrics = WorkflowRunner._concentration_metrics([])
        assert metrics["max_weight"] == 0.0
        assert metrics["herfindahl"] == 0.0

    def test_trials_without_allocations(self) -> None:
        trials = [
            TrainingTrial(iteration=1, update=0, allocation=[]),
        ]
        metrics = WorkflowRunner._concentration_metrics(trials)
        assert metrics["max_weight"] == 0.0
        assert metrics["herfindahl"] == 0.0

    def test_all_zero_allocation(self) -> None:
        trials = [
            TrainingTrial(iteration=1, update=0, allocation=[0.0, 0.0, 0.0]),
        ]
        metrics = WorkflowRunner._concentration_metrics(trials)
        assert metrics["max_weight"] == 0.0
        assert metrics["herfindahl"] == pytest.approx(0.0)

    def test_hhi_formula_correctness(self) -> None:
        """Verify HHI is sum of squared weights."""
        alloc = [0.6, 0.3, 0.1]
        trials = [TrainingTrial(iteration=1, update=0, allocation=alloc)]
        metrics = WorkflowRunner._concentration_metrics(trials)
        expected_hhi = 0.6**2 + 0.3**2 + 0.1**2  # 0.36 + 0.09 + 0.01 = 0.46
        assert metrics["herfindahl"] == pytest.approx(expected_hhi)


# ---------------------------------------------------------------------------
# TrainingConfig validation tests (edge cases)
# ---------------------------------------------------------------------------

class TestTrainingConfigValidation:
    """Tests for TrainingConfig __post_init__ validation."""

    def test_empty_tickers_raises_value_error(self) -> None:
        """Empty tickers list raises ValueError with meaningful message."""
        with pytest.raises(ValueError, match="Tickers list is empty"):
            TrainingConfig(tickers=[], positions={"AAPL": 50000.0})

    def test_empty_positions_raises_value_error(self) -> None:
        """Empty positions dict raises ValueError with meaningful message."""
        with pytest.raises(ValueError, match="Positions dictionary is empty"):
            TrainingConfig(tickers=["AAPL"], positions={})

    def test_zero_position_value_raises_value_error(self) -> None:
        """Zero position value raises ValueError with meaningful message."""
        with pytest.raises(ValueError, match="Position value for 'AAPL' is 0"):
            TrainingConfig(tickers=["AAPL"], positions={"AAPL": 0.0})

    def test_negative_position_value_raises_value_error(self) -> None:
        """Negative position value raises ValueError with meaningful message."""
        with pytest.raises(ValueError, match="Position value for 'AAPL' is -100"):
            TrainingConfig(tickers=["AAPL"], positions={"AAPL": -100.0})

    def test_valid_config_does_not_raise(self) -> None:
        """Valid config with positive positions does not raise."""
        cfg = TrainingConfig(
            tickers=["AAPL", "GOOGL"],
            positions={"AAPL": 50000.0, "GOOGL": 50000.0},
        )
        assert cfg.tickers == ["AAPL", "GOOGL"]
        assert cfg.positions == {"AAPL": 50000.0, "GOOGL": 50000.0}

    def test_mixed_valid_invalid_positions_raises(self) -> None:
        """When one position is invalid, ValueError identifies the bad ticker."""
        with pytest.raises(ValueError, match="Position value for 'META' is -50"):
            TrainingConfig(
                tickers=["AAPL", "META"],
                positions={"AAPL": 50000.0, "META": -50.0},
            )
