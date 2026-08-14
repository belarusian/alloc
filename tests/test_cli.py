"""Tests for alloc.cli — CLI entry point, type converters, parser, config builder.

Covers:
- Type converters (_positive_int, _non_negative_int, _non_negative_float,
  _positive_float, _fraction, _json_string)
- build_parser() argument definitions
- parse_args() post-processing (ticker splitting, uppercasing, fresh-only)
- build_config() mapping to TrainingConfig
- print_results() output
- main() exit codes and orchestration
"""

from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alloc.cli import (
    EXIT_SUCCESS,
    EXIT_UNEXPECTED,
    EXIT_USER_ERROR,
    EXIT_WORKFLOW_FAIL,
    _fraction,
    _json_string,
    _non_negative_float,
    _non_negative_int,
    _positive_float,
    _positive_int,
    build_config,
    build_parser,
    main,
    parse_args,
    print_results,
)
from alloc.utils.workflow import TrainingConfig, TrainingTrial, WorkflowResult


# ===================================================================
# Type converter tests
# ===================================================================


class TestPositiveInt:
    """Tests for _positive_int type converter."""

    def test_valid_values(self) -> None:
        assert _positive_int("1") == 1
        assert _positive_int("42") == 42
        assert _positive_int("999999") == 999999

    def test_zero_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be > 0"):
            _positive_int("0")

    def test_negative_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be > 0"):
            _positive_int("-1")

    def test_non_integer_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="not a valid integer"):
            _positive_int("abc")

    def test_float_string_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="not a valid integer"):
            _positive_int("3.14")


class TestNonNegativeInt:
    """Tests for _non_negative_int type converter."""

    def test_zero_accepted(self) -> None:
        assert _non_negative_int("0") == 0

    def test_positive_accepted(self) -> None:
        assert _non_negative_int("1") == 1
        assert _non_negative_int("100") == 100

    def test_negative_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be >= 0"):
            _non_negative_int("-1")

    def test_non_integer_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="not a valid integer"):
            _non_negative_int("abc")


class TestNonNegativeFloat:
    """Tests for _non_negative_float type converter."""

    def test_zero_accepted(self) -> None:
        assert _non_negative_float("0") == 0.0

    def test_positive_accepted(self) -> None:
        assert _non_negative_float("0.5") == 0.5
        assert _non_negative_float("100.0") == 100.0

    def test_negative_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be >= 0"):
            _non_negative_float("-0.1")

    def test_non_number_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="not a valid number"):
            _non_negative_float("abc")


class TestPositiveFloat:
    """Tests for _positive_float type converter."""

    def test_positive_accepted(self) -> None:
        assert _positive_float("0.001") == 0.001
        assert _positive_float("1.0") == 1.0

    def test_zero_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be > 0"):
            _positive_float("0")

    def test_negative_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be > 0"):
            _positive_float("-1.0")

    def test_non_number_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="not a valid number"):
            _positive_float("abc")


class TestFraction:
    """Tests for _fraction type converter."""

    def test_zero_accepted(self) -> None:
        assert _fraction("0") == 0.0

    def test_one_accepted(self) -> None:
        assert _fraction("1") == 1.0

    def test_mid_range_accepted(self) -> None:
        assert _fraction("0.5") == 0.5
        assert _fraction("0.001") == 0.001

    def test_above_one_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be in \\[0, 1\\]"):
            _fraction("1.1")

    def test_negative_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be in \\[0, 1\\]"):
            _fraction("-0.1")

    def test_non_number_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="not a valid number"):
            _fraction("abc")


class TestJsonString:
    """Tests for _json_string type converter."""

    def test_valid_dict(self) -> None:
        result = _json_string('{"AAPL": 50000, "META": 50000}')
        assert result == {"AAPL": 50000.0, "META": 50000.0}

    def test_integer_values_coerced_to_float(self) -> None:
        result = _json_string('{"X": 1}')
        assert result == {"X": 1.0}

    def test_empty_dict_accepted(self) -> None:
        result = _json_string('{}')
        assert result == {}

    def test_list_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be a JSON object"):
            _json_string('[1, 2, 3]')

    def test_string_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="not valid JSON"):
            _json_string('not json')

    def test_non_string_key_rejected(self) -> None:
        """JSON always has string keys, so {1: ...} is invalid JSON."""
        with pytest.raises(argparse.ArgumentTypeError, match="not valid JSON"):
            _json_string('{1: "a"}')

    def test_non_numeric_value_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be numeric"):
            _json_string('{"AAPL": "not_a_number"}')


# ===================================================================
# build_parser tests
# ===================================================================


class TestBuildParser:
    """Tests for build_parser()."""

    def test_returns_argument_parser(self) -> None:
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_required_tickers_arg(self) -> None:
        parser = build_parser()
        # Parse without --tickers should fail
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_required_positions_values_arg(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--tickers", "AAPL"])

    def test_all_training_config_args_present(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--tickers", "AAPL,META",
            "--positions-values", '{"AAPL": 50000}',
        ])
        # Check all TrainingConfig fields are present as attributes
        assert hasattr(args, "tickers")
        assert hasattr(args, "positions_values")
        assert hasattr(args, "iterations")
        assert hasattr(args, "update_iterations")
        assert hasattr(args, "trading_days")
        assert hasattr(args, "batch_size")
        assert hasattr(args, "min_allocation")
        assert hasattr(args, "concentration_penalty")
        assert hasattr(args, "transaction_cost")
        assert hasattr(args, "risk_aversion")
        assert hasattr(args, "min_cash_allocation")
        assert hasattr(args, "target_sharpe")
        assert hasattr(args, "target_value")
        assert hasattr(args, "target_outperformance")

    def test_flag_args_present(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--tickers", "AAPL",
            "--positions-values", '{"AAPL": 1000}',
            "--fresh-only",
            "--conservative",
            "--verbose",
        ])
        assert args.fresh_only is True
        assert args.conservative is True
        assert args.verbose is True

    def test_default_values(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--tickers", "AAPL",
            "--positions-values", '{"AAPL": 1000}',
        ])
        assert args.iterations == 1
        assert args.update_iterations == 1
        assert args.trading_days == 222
        assert args.batch_size == 22
        assert args.min_allocation == 0.001
        assert args.concentration_penalty == 0.001
        assert args.transaction_cost == 0.0
        assert args.risk_aversion == 0.001
        assert args.min_cash_allocation == 0.05
        assert args.target_sharpe == 2.1
        assert args.target_value == 220_000.0
        assert args.target_outperformance == 15.0
        assert args.fresh_only is False
        assert args.conservative is False
        assert args.verbose is False

    def test_custom_values(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--tickers", "AAPL,META",
            "--positions-values", '{"AAPL": 50000}',
            "--iterations", "10",
            "--update-iterations", "5",
            "--trading-days", "252",
            "--batch-size", "32",
            "--min-allocation", "0.01",
            "--concentration-penalty", "0.01",
            "--transaction-cost", "0.001",
            "--risk-aversion", "0.01",
            "--min-cash-allocation", "0.1",
            "--target-sharpe", "3.0",
            "--target-value", "300000",
            "--target-outperformance", "20.0",
        ])
        assert args.iterations == 10
        assert args.update_iterations == 5
        assert args.trading_days == 252
        assert args.batch_size == 32
        assert args.min_allocation == 0.01
        assert args.concentration_penalty == 0.01
        assert args.transaction_cost == 0.001
        assert args.risk_aversion == 0.01
        assert args.min_cash_allocation == 0.1
        assert args.target_sharpe == 3.0
        assert args.target_value == 300_000.0
        assert args.target_outperformance == 20.0


# ===================================================================
# parse_args tests
# ===================================================================


class TestParseArgs:
    """Tests for parse_args() post-processing."""

    def test_ticker_splitting(self) -> None:
        args = parse_args([
            "--tickers", "AAPL,META,GOOG",
            "--positions-values", '{"AAPL": 50000}',
        ])
        assert args.ticker_list == ["AAPL", "META", "GOOG"]

    def test_ticker_uppercasing(self) -> None:
        args = parse_args([
            "--tickers", "aapl,meta,goog",
            "--positions-values", '{"AAPL": 50000}',
        ])
        assert args.ticker_list == ["AAPL", "META", "GOOG"]

    def test_ticker_whitespace_stripping(self) -> None:
        args = parse_args([
            "--tickers", " AAPL , META , GOOG ",
            "--positions-values", '{"AAPL": 50000}',
        ])
        assert args.ticker_list == ["AAPL", "META", "GOOG"]

    def test_single_ticker(self) -> None:
        args = parse_args([
            "--tickers", "AAPL",
            "--positions-values", '{"AAPL": 100000}',
        ])
        assert args.ticker_list == ["AAPL"]

    def test_fresh_only_sets_update_iterations_zero(self) -> None:
        args = parse_args([
            "--tickers", "AAPL",
            "--positions-values", '{"AAPL": 100000}',
            "--fresh-only",
            "--update-iterations", "5",
        ])
        assert args.update_iterations == 0

    def test_fresh_only_false_keeps_update_iterations(self) -> None:
        args = parse_args([
            "--tickers", "AAPL",
            "--positions-values", '{"AAPL": 100000}',
            "--update-iterations", "3",
        ])
        assert args.update_iterations == 3

    def test_empty_tickers_rejected(self) -> None:
        with pytest.raises(SystemExit):
            parse_args([
                "--tickers", ",,,",
                "--positions-values", '{}',
            ])

    def test_positions_values_parsed_as_dict(self) -> None:
        args = parse_args([
            "--tickers", "AAPL,META",
            "--positions-values", '{"AAPL": 50000, "META": 50000}',
        ])
        assert args.positions_values == {"AAPL": 50000.0, "META": 50000.0}


# ===================================================================
# build_config tests
# ===================================================================


class TestBuildConfig:
    """Tests for build_config()."""

    def test_basic_mapping(self) -> None:
        args = parse_args([
            "--tickers", "AAPL,META",
            "--positions-values", '{"AAPL": 50000, "META": 50000}',
        ])
        config = build_config(args)
        assert isinstance(config, TrainingConfig)
        assert config.tickers == ["AAPL", "META"]
        assert config.positions == {"AAPL": 50000.0, "META": 50000.0}

    def test_all_fields_mapped(self) -> None:
        args = parse_args([
            "--tickers", "AAPL,META",
            "--positions-values", '{"AAPL": 50000, "META": 50000}',
            "--iterations", "10",
            "--update-iterations", "5",
            "--trading-days", "252",
            "--batch-size", "32",
            "--min-allocation", "0.01",
            "--concentration-penalty", "0.01",
            "--transaction-cost", "0.001",
            "--risk-aversion", "0.01",
            "--min-cash-allocation", "0.1",
            "--target-sharpe", "3.0",
            "--target-value", "300000",
            "--target-outperformance", "20.0",
        ])
        config = build_config(args)
        assert config.iterations == 10
        assert config.update_iterations == 5
        assert config.trading_days == 252
        assert config.batch_size == 32
        assert config.min_allocation == 0.01
        assert config.concentration_penalty == 0.01
        assert config.transaction_cost == 0.001
        assert config.risk_aversion == 0.01
        assert config.min_cash_alloc == 0.1
        assert config.target_sharpe == 3.0
        assert config.target_value == 300_000.0
        assert config.target_outperformance == 20.0

    def test_default_values_preserved(self) -> None:
        args = parse_args([
            "--tickers", "AAPL",
            "--positions-values", '{"AAPL": 100000}',
        ])
        config = build_config(args)
        assert config.iterations == 1
        assert config.update_iterations == 1
        assert config.trading_days == 222
        assert config.batch_size == 22
        assert config.min_allocation == 0.001
        assert config.concentration_penalty == 0.001
        assert config.transaction_cost == 0.0
        assert config.risk_aversion == 0.001
        assert config.min_cash_alloc == 0.05
        assert config.target_sharpe == 2.1
        assert config.target_value == 220_000.0
        assert config.target_outperformance == 15.0


# ===================================================================
# print_results tests
# ===================================================================


class TestPrintResults:
    """Tests for print_results()."""

    def test_prints_best_trial_info(self, caplog: Any) -> None:
        import logging
        caplog.set_level(logging.INFO)

        best = TrainingTrial(
            iteration=3,
            update=2,
            sharpe_ratio=2.5,
            outperformance=15.0,
            final_value=250_000.0,
            model_roi=25.0,
            buyhold_roi=10.0,
            allocation=[0.4, 0.4, 0.2],
        )
        result = WorkflowResult(
            status="success",
            best_trial=best,
            trials=[best],
            allocation_stats={},
            concentration={"max_weight": 0.4, "herfindahl": 0.33},
            metrics_progression=[],
        )
        print_results(result)
        assert "WORKFLOW COMPLETE" in caplog.text
        assert "Best trial #3" in caplog.text
        assert "Sharpe=2.500" in caplog.text

    def test_prints_concentration(self, caplog: Any) -> None:
        import logging
        caplog.set_level(logging.INFO)

        best = TrainingTrial(
            iteration=1,
            update=0,
            sharpe_ratio=1.0,
            outperformance=5.0,
            final_value=110_000.0,
            model_roi=10.0,
            buyhold_roi=5.0,
            allocation=[0.5, 0.5],
        )
        result = WorkflowResult(
            status="success",
            best_trial=best,
            trials=[best],
            allocation_stats={},
            concentration={"max_weight": 0.5, "herfindahl": 0.5},
            metrics_progression=[],
        )
        print_results(result)
        assert "max_weight=0.500" in caplog.text
        assert "herfindahl=0.500" in caplog.text

    def test_prints_centroid(self, caplog: Any) -> None:
        import logging
        caplog.set_level(logging.INFO)

        best = TrainingTrial(
            iteration=1,
            update=0,
            sharpe_ratio=1.0,
            outperformance=5.0,
            final_value=110_000.0,
            model_roi=10.0,
            buyhold_roi=5.0,
            allocation=[0.5, 0.5],
        )
        result = WorkflowResult(
            status="success",
            best_trial=best,
            trials=[best],
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )
        print_results(result)

    def test_no_best_trial_warns(self, caplog: Any) -> None:
        import logging
        caplog.set_level(logging.WARNING)
        # When trials exist but best_trial is None, we get "No best trial"
        result = WorkflowResult(
            status="error",
            best_trial=None,  # type: ignore[arg-type]
            trials=[TrainingTrial(iteration=1, update=0)],
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )
        print_results(result)
        assert "No best trial" in caplog.text

    def test_prints_trial_count(self, caplog: Any) -> None:
        import logging
        caplog.set_level(logging.INFO)

        trials = [
            TrainingTrial(iteration=i, update=0, allocation=[0.5, 0.5])
            for i in range(1, 6)
        ]
        result = WorkflowResult(
            status="success",
            best_trial=trials[-1],
            trials=trials,
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )
        print_results(result)
        assert "Total trials completed: 5" in caplog.text


# ===================================================================
# main() tests
# ===================================================================


class TestMain:
    """Tests for main() exit codes and orchestration."""

    def test_main_success_exit_code(self) -> None:
        """main() returns EXIT_SUCCESS on successful workflow."""
        mock_result = WorkflowResult(
            status="success",
            best_trial=TrainingTrial(
                iteration=1,
                update=0,
                sharpe_ratio=2.0,
                outperformance=10.0,
                final_value=200_000.0,
                model_roi=100.0,
                buyhold_roi=50.0,
                allocation=[0.5, 0.5],
            ),
            trials=[],
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )

        with patch("alloc.cli.parse_args") as mock_parse, \
             patch("alloc.cli.WorkflowRunner") as mock_runner_cls, \
             patch("alloc.core.create_trainer") as mock_create_trainer:
            mock_parse.return_value = MagicMock(
                ticker_list=["AAPL"],
                positions_values={"AAPL": 50000.0},
                iterations=1,
                update_iterations=1,
                conservative=False,
                verbose=False,
            )
            mock_runner = MagicMock()
            mock_runner.run.return_value = mock_result
            mock_runner_cls.return_value = mock_runner
            mock_create_trainer.return_value = MagicMock()

            exit_code = main([])
            assert exit_code == EXIT_SUCCESS

    def test_main_workflow_fail_exit_code(self) -> None:
        """main() returns EXIT_WORKFLOW_FAIL on workflow failure."""
        mock_result = WorkflowResult(
            status="error",
            best_trial=TrainingTrial(iteration=1, update=0),
            trials=[],
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )

        with patch("alloc.cli.parse_args") as mock_parse, \
             patch("alloc.cli.WorkflowRunner") as mock_runner_cls, \
             patch("alloc.core.create_trainer") as mock_create_trainer:
            mock_parse.return_value = MagicMock(
                ticker_list=["AAPL"],
                positions_values={"AAPL": 50000.0},
                iterations=1,
                update_iterations=1,
                conservative=False,
                verbose=False,
            )
            mock_runner = MagicMock()
            mock_runner.run.return_value = mock_result
            mock_runner_cls.return_value = mock_runner
            mock_create_trainer.return_value = MagicMock()

            exit_code = main([])
            assert exit_code == EXIT_WORKFLOW_FAIL

    def test_main_exception_returns_workflow_fail(self) -> None:
        """main() returns EXIT_WORKFLOW_FAIL when runner raises."""
        with patch("alloc.cli.parse_args") as mock_parse, \
             patch("alloc.cli.WorkflowRunner") as mock_runner_cls, \
             patch("alloc.core.create_trainer") as mock_create_trainer:
            mock_parse.return_value = MagicMock(
                ticker_list=["AAPL"],
                positions_values={"AAPL": 50000.0},
                iterations=1,
                update_iterations=1,
                conservative=False,
                verbose=False,
            )
            mock_runner = MagicMock()
            mock_runner.run.side_effect = RuntimeError("simulated failure")
            mock_runner_cls.return_value = mock_runner
            mock_create_trainer.return_value = MagicMock()

            exit_code = main([])
            assert exit_code == EXIT_WORKFLOW_FAIL

    def test_main_calls_workflow_runner(self) -> None:
        """main() instantiates WorkflowRunner and calls run()."""
        mock_result = WorkflowResult(
            status="success",
            best_trial=TrainingTrial(
                iteration=1, update=0, allocation=[0.5, 0.5],
            ),
            trials=[],
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )

        with patch("alloc.cli.parse_args") as mock_parse, \
             patch("alloc.cli.WorkflowRunner") as mock_runner_cls, \
             patch("alloc.core.create_trainer") as mock_create_trainer:
            mock_parse.return_value = MagicMock(
                ticker_list=["AAPL"],
                positions_values={"AAPL": 50000.0},
                iterations=1,
                update_iterations=1,
                conservative=False,
                verbose=False,
            )
            mock_runner = MagicMock()
            mock_runner.run.return_value = mock_result
            mock_runner_cls.return_value = mock_runner
            mock_create_trainer.return_value = MagicMock()

            main([])
            mock_runner.run.assert_called_once()

    def test_main_passes_conservative_to_create_trainer(self) -> None:
        """main() passes conservative flag to create_trainer."""
        mock_result = WorkflowResult(
            status="success",
            best_trial=TrainingTrial(
                iteration=1, update=0, allocation=[0.5, 0.5],
            ),
            trials=[],
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )

        with patch("alloc.cli.parse_args") as mock_parse, \
             patch("alloc.cli.WorkflowRunner") as mock_runner_cls, \
             patch("alloc.core.create_trainer") as mock_create_trainer:
            mock_parse.return_value = MagicMock(
                ticker_list=["AAPL"],
                positions_values={"AAPL": 50000.0},
                iterations=1,
                update_iterations=1,
                conservative=True,
                verbose=False,
            )
            mock_runner = MagicMock()
            mock_runner.run.return_value = mock_result
            mock_runner_cls.return_value = mock_runner
            mock_create_trainer.return_value = MagicMock()

            main([])
            mock_create_trainer.assert_called_once_with(conservative=True)

    def test_main_help_returns_success(self) -> None:
        """main() returns EXIT_SUCCESS (0) when --help is passed.

        argparse calls sys.exit(0) for --help, which main() propagates.
        """
        exit_code = main(["--help"])
        assert exit_code == EXIT_SUCCESS

    def test_main_invalid_args_returns_non_zero(self) -> None:
        """main() returns non-zero exit code on invalid arguments.

        argparse calls sys.exit(2) on argument errors, which main()
        propagates as the exit code.
        """
        # Missing required --tickers
        exit_code = main(["--positions-values", '{"AAPL": 100}'])
        assert exit_code != EXIT_SUCCESS

    def test_main_builds_config(self) -> None:
        """main() calls build_config with parsed args."""
        mock_result = WorkflowResult(
            status="success",
            best_trial=TrainingTrial(
                iteration=1, update=0, allocation=[0.5, 0.5],
            ),
            trials=[],
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )

        with patch("alloc.cli.parse_args") as mock_parse, \
             patch("alloc.cli.build_config") as mock_build_config, \
             patch("alloc.cli.WorkflowRunner") as mock_runner_cls, \
             patch("alloc.core.create_trainer") as mock_create_trainer:
            mock_args = MagicMock(
                ticker_list=["AAPL"],
                positions_values={"AAPL": 50000.0},
                iterations=1,
                update_iterations=1,
                conservative=False,
                verbose=False,
            )
            mock_parse.return_value = mock_args
            mock_build_config.return_value = MagicMock()
            mock_runner = MagicMock()
            mock_runner.run.return_value = mock_result
            mock_runner_cls.return_value = mock_runner
            mock_create_trainer.return_value = MagicMock()

            main([])
            mock_build_config.assert_called_once_with(mock_args)


# ===================================================================
# Exit code constants
# ===================================================================


class TestExitCodes:
    """Tests for exit code constants."""

    def test_exit_success_is_zero(self) -> None:
        assert EXIT_SUCCESS == 0

    def test_exit_user_error_is_one(self) -> None:
        assert EXIT_USER_ERROR == 1

    def test_exit_workflow_fail_is_two(self) -> None:
        assert EXIT_WORKFLOW_FAIL == 2

    def test_exit_unexpected_is_three(self) -> None:
        assert EXIT_UNEXPECTED == 3


# ===================================================================
# __main__ module test
# ===================================================================


class TestMainModule:
    """Tests for alloc.__main__."""

    def test_main_module_imports(self) -> None:
        """alloc.__main__ can be imported without side effects."""
        import alloc.__main__  # noqa: F401
        # Should not raise

    def test_main_module_has_main(self) -> None:
        """alloc.__main__ exposes main from alloc.cli."""
        from alloc.__main__ import main as main_entry
        assert callable(main_entry)


# ===================================================================
# Edge case validation tests (issue #58)
# ===================================================================


class TestBuildConfigEdgeCases:
    """Tests for build_config edge case validation."""

    def test_empty_tickers_raises_value_error(self) -> None:
        """build_config raises ValueError when tickers list is empty."""
        args = MagicMock(
            ticker_list=[],
            positions_values={"AAPL": 50000.0},
            iterations=1,
            update_iterations=1,
            trading_days=222,
            batch_size=22,
            min_allocation=0.001,
            concentration_penalty=0.001,
            transaction_cost=0.0,
            risk_aversion=0.001,
            min_cash_allocation=0.05,
            target_sharpe=2.1,
            target_value=220000.0,
            target_outperformance=15.0,
        )
        with pytest.raises(ValueError, match="Tickers list is empty"):
            build_config(args)

    def test_zero_position_raises_value_error(self) -> None:
        """build_config raises ValueError when a position value is zero."""
        args = MagicMock(
            ticker_list=["AAPL"],
            positions_values={"AAPL": 0.0},
            iterations=1,
            update_iterations=1,
            trading_days=222,
            batch_size=22,
            min_allocation=0.001,
            concentration_penalty=0.001,
            transaction_cost=0.0,
            risk_aversion=0.001,
            min_cash_allocation=0.05,
            target_sharpe=2.1,
            target_value=220000.0,
            target_outperformance=15.0,
        )
        with pytest.raises(ValueError, match="Position value for 'AAPL' is 0"):
            build_config(args)

    def test_negative_position_raises_value_error(self) -> None:
        """build_config raises ValueError when a position value is negative."""
        args = MagicMock(
            ticker_list=["AAPL"],
            positions_values={"AAPL": -100.0},
            iterations=1,
            update_iterations=1,
            trading_days=222,
            batch_size=22,
            min_allocation=0.001,
            concentration_penalty=0.001,
            transaction_cost=0.0,
            risk_aversion=0.001,
            min_cash_allocation=0.05,
            target_sharpe=2.1,
            target_value=220000.0,
            target_outperformance=15.0,
        )
        with pytest.raises(ValueError, match="Position value for 'AAPL' is -100"):
            build_config(args)

    def test_empty_positions_dict_raises_value_error(self) -> None:
        """build_config raises ValueError when positions dict is empty."""
        args = MagicMock(
            ticker_list=["AAPL"],
            positions_values={},
            iterations=1,
            update_iterations=1,
            trading_days=222,
            batch_size=22,
            min_allocation=0.001,
            concentration_penalty=0.001,
            transaction_cost=0.0,
            risk_aversion=0.001,
            min_cash_allocation=0.05,
            target_sharpe=2.1,
            target_value=220000.0,
            target_outperformance=15.0,
        )
        with pytest.raises(ValueError, match="Positions dictionary is empty"):
            build_config(args)

    def test_non_dict_positions_raises_value_error(self) -> None:
        """build_config raises ValueError when positions is not a dict."""
        args = MagicMock(
            ticker_list=["AAPL"],
            positions_values="not a dict",
            iterations=1,
            update_iterations=1,
            trading_days=222,
            batch_size=22,
            min_allocation=0.001,
            concentration_penalty=0.001,
            transaction_cost=0.0,
            risk_aversion=0.001,
            min_cash_allocation=0.05,
            target_sharpe=2.1,
            target_value=220000.0,
            target_outperformance=15.0,
        )
        with pytest.raises(ValueError, match="Positions must be a JSON object"):
            build_config(args)


class TestPrintResultsEmptyWorkflow:
    """Tests for print_results graceful handling of empty workflow results."""

    def test_empty_trials_logs_warning(self, caplog: Any) -> None:
        """print_results logs warning when no trials were completed and best_trial is placeholder."""
        import logging
        caplog.set_level(logging.WARNING)
        result = WorkflowResult(
            status="error",
            best_trial=TrainingTrial(iteration=0, update=0, allocation=[]),
            trials=[],
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )
        print_results(result)
        assert "no valid trial" in caplog.text.lower()

    def test_empty_trials_with_valid_best_still_shows(self, caplog: Any) -> None:
        """print_results warns about empty trials but still shows valid best_trial."""
        import logging
        caplog.set_level(logging.INFO)
        result = WorkflowResult(
            status="success",
            best_trial=TrainingTrial(
                iteration=1, update=0, sharpe_ratio=2.0,
                outperformance=10.0, final_value=120000.0,
                model_roi=20.0, buyhold_roi=10.0,
                allocation=[0.5, 0.5],
            ),
            trials=[],
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )
        print_results(result)
        # Should warn about no trials but still show results
        assert "no trials" in caplog.text.lower()
        assert "WORKFLOW COMPLETE" in caplog.text

    def test_empty_best_trial_logs_warning(self, caplog: Any) -> None:
        """print_results logs warning when best_trial is placeholder (iteration=0)."""
        import logging
        caplog.set_level(logging.WARNING)
        result = WorkflowResult(
            status="error",
            best_trial=TrainingTrial(iteration=0, update=0, allocation=[]),
            trials=[TrainingTrial(iteration=0, update=0, allocation=[])],
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )
        print_results(result)
        assert "no valid trial" in caplog.text.lower()

    def test_none_best_trial_logs_warning(self, caplog: Any) -> None:
        """print_results logs warning when best_trial is None."""
        import logging
        caplog.set_level(logging.WARNING)
        result = WorkflowResult(
            status="error",
            best_trial=None,  # type: ignore[arg-type]
            trials=[TrainingTrial(iteration=1, update=0)],
            allocation_stats={},
            concentration={},
            metrics_progression=[],
        )
        print_results(result)
        assert "No best trial" in caplog.text


class TestJsonStringInvalidInput:
    """Tests for _json_string with invalid JSON input."""

    def test_invalid_json_raises_argument_type_error(self) -> None:
        """_json_string raises ArgumentTypeError on malformed JSON."""
        with pytest.raises(argparse.ArgumentTypeError, match="not valid JSON"):
            _json_string("{invalid json}")

    def test_json_array_raises_argument_type_error(self) -> None:
        """_json_string raises ArgumentTypeError when JSON is an array."""
        with pytest.raises(argparse.ArgumentTypeError, match="must be a JSON object"):
            _json_string('[1, 2, 3]')

    def test_json_string_raises_argument_type_error(self) -> None:
        """_json_string raises ArgumentTypeError when JSON is a string."""
        with pytest.raises(argparse.ArgumentTypeError, match="must be a JSON object"):
            _json_string('"just a string"')

    def test_json_number_raises_argument_type_error(self) -> None:
        """_json_string raises ArgumentTypeError when JSON is a number."""
        with pytest.raises(argparse.ArgumentTypeError, match="must be a JSON object"):
            _json_string('42')

    def test_json_null_raises_argument_type_error(self) -> None:
        """_json_string raises ArgumentTypeError when JSON is null."""
        with pytest.raises(argparse.ArgumentTypeError, match="must be a JSON object"):
            _json_string('null')

    def test_json_with_non_numeric_value_raises(self) -> None:
        """_json_string raises ArgumentTypeError when value is non-numeric."""
        with pytest.raises(argparse.ArgumentTypeError, match="must be numeric"):
            _json_string('{"AAPL": "not_a_number"}')


# ===================================================================
# TICKET-041: Ticker-position consistency validation
# ===================================================================


class TestTickerPositionConsistency:
    """Tests for TICKET-041: ticker-position cross-validation in TrainingConfig."""

    def test_matching_tickers_and_positions(self) -> None:
        """Matching tickers and positions should pass validation."""
        config = TrainingConfig(
            tickers=["AAPL", "MSFT"],
            positions={"AAPL": 50000.0, "MSFT": 50000.0},
        )
        assert config.tickers == ["AAPL", "MSFT"]

    def test_ticker_missing_from_positions(self) -> None:
        """Ticker without a corresponding position should raise ValueError."""
        with pytest.raises(ValueError, match="Tickers without corresponding positions"):
            TrainingConfig(
                tickers=["AAPL", "MSFT", "GOOG"],
                positions={"AAPL": 50000.0, "MSFT": 50000.0},
            )

    def test_position_extra_ticker_not_in_tickers(self) -> None:
        """Position key not in tickers list should raise ValueError."""
        with pytest.raises(ValueError, match="Positions without corresponding tickers"):
            TrainingConfig(
                tickers=["AAPL", "MSFT"],
                positions={"AAPL": 50000.0, "MSFT": 50000.0, "GOOG": 30000.0},
            )

    def test_empty_tickers_raises(self) -> None:
        """Empty tickers list should raise ValueError."""
        with pytest.raises(ValueError, match="Tickers list is empty"):
            TrainingConfig(
                tickers=[],
                positions={"AAPL": 50000.0},
            )

    def test_empty_positions_raises(self) -> None:
        """Empty positions dict should raise ValueError."""
        with pytest.raises(ValueError, match="Positions dictionary is empty"):
            TrainingConfig(
                tickers=["AAPL"],
                positions={},
            )

    def test_zero_position_value_raises(self) -> None:
        """Zero position value should raise ValueError."""
        with pytest.raises(ValueError, match="Position value.*is 0"):
            TrainingConfig(
                tickers=["AAPL"],
                positions={"AAPL": 0.0},
            )

    def test_negative_position_value_raises(self) -> None:
        """Negative position value should raise ValueError."""
        with pytest.raises(ValueError, match="Position value.*is -100"):
            TrainingConfig(
                tickers=["AAPL"],
                positions={"AAPL": -100.0},
            )

    def test_invalid_json_positions_in_build_config(self) -> None:
        """build_config should raise ValueError for non-dict positions_values."""
        args = argparse.Namespace(
            ticker_list=["AAPL"],
            positions_values="not a dict",
            iterations=1,
            update_iterations=1,
            trading_days=222,
            batch_size=22,
            min_allocation=0.001,
            concentration_penalty=0.001,
            transaction_cost=0.0,
            risk_aversion=0.001,
            min_cash_allocation=0.05,
            target_sharpe=2.1,
            target_value=220000.0,
            target_outperformance=15.0,
        )
        with pytest.raises(ValueError, match="Positions must be a JSON object"):
            build_config(args)

    def test_both_mismatch_raises_first_error(self) -> None:
        """When both tickers missing from positions AND extra positions exist,
        the missing-from-positions error is raised first."""
        with pytest.raises(ValueError, match="Tickers without corresponding positions"):
            TrainingConfig(
                tickers=["AAPL", "MSFT", "GOOG"],
                positions={"AAPL": 50000.0, "TSLA": 30000.0},
            )
