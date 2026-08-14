"""alloc.cli — CLI entry point for the multi-trial training workflow.

Wraps :class:`alloc.utils.workflow.WorkflowRunner` behind an argparse-based
command-line interface.  Every field of :class:`alloc.utils.workflow.TrainingConfig`
is exposed as a CLI argument with sensible defaults matching the dataclass.

Design principles
-----------------
* **Typed argparse** — custom type converters validate input at parse time
  (positive ints, non-negative floats, JSON strings).
* **Separation of concerns** — :func:`build_parser` returns a configured
  ``ArgumentParser``; :func:`parse_args` returns a ``Namespace``;
  :func:`main` orchestrates execution.  No library code is called during
  parser construction.
* **Logging, not print** — all diagnostics go through ``logging``.
* **Meaningful exit codes** — 0 = success, 1 = user error (bad args),
  2 = workflow failure, 3 = unexpected exception.

Usage
-----
    python -m alloc.cli \\
        --tickers AAPL,META,GOOG \\
        --positions-values '{"AAPL": 50000, "META": 50000}' \\
        --iterations 5 \\
        --update-iterations 3

Or via the module entry-point::

    python -m alloc.cli --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from alloc.utils.workflow import TrainingConfig, WorkflowResult, WorkflowRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_SUCCESS = 0
EXIT_USER_ERROR = 1      # bad arguments, JSON parse failure
EXIT_WORKFLOW_FAIL = 2   # workflow returned non-success status
EXIT_UNEXPECTED = 3      # unhandled exception

# ---------------------------------------------------------------------------
# Type converters
# ---------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    """Convert *value* to a strictly positive integer.

    Raises ``argparse.ArgumentTypeError`` on failure.
    """
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid integer"
        )
    if n <= 0:
        raise argparse.ArgumentTypeError(
            f"'{n}' must be > 0"
        )
    return n


def _non_negative_int(value: str) -> int:
    """Convert *value* to a non-negative integer (≥ 0).

    Raises ``argparse.ArgumentTypeError`` on failure.
    """
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid integer"
        )
    if n < 0:
        raise argparse.ArgumentTypeError(
            f"'{n}' must be >= 0"
        )
    return n


def _non_negative_float(value: str) -> float:
    """Convert *value* to a non-negative float (≥ 0).

    Raises ``argparse.ArgumentTypeError`` on failure.
    """
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid number"
        )
    if f < 0:
        raise argparse.ArgumentTypeError(
            f"'{f}' must be >= 0"
        )
    return f


def _positive_float(value: str) -> float:
    """Convert *value* to a strictly positive float (> 0).

    Raises ``argparse.ArgumentTypeError`` on failure.
    """
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid number"
        )
    if f <= 0:
        raise argparse.ArgumentTypeError(
            f"'{f}' must be > 0"
        )
    return f


def _fraction(value: str) -> float:
    """Convert *value* to a float in the range [0, 1].

    Raises ``argparse.ArgumentTypeError`` on failure.
    """
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid number"
        )
    if not (0.0 <= f <= 1.0):
        raise argparse.ArgumentTypeError(
            f"'{f}' must be in [0, 1]"
        )
    return f


def _json_string(value: str) -> dict[str, float]:
    """Parse *value* as a JSON object mapping strings to numbers.

    Raises ``argparse.ArgumentTypeError`` on failure.
    """
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not valid JSON: {exc}"
        )
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError(
            f"positions must be a JSON object (dict), got {type(data).__name__}"
        )
    # Coerce all values to float
    result: dict[str, float] = {}
    for key, val in data.items():
        if not isinstance(key, str):
            raise argparse.ArgumentTypeError(
                f"position keys must be strings, got {type(key).__name__}"
            )
        try:
            result[key] = float(val)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(
                f"position value for '{key}' must be numeric, got {val!r}"
            )
    return result


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Fully configured parser for the training workflow.
    """
    parser = argparse.ArgumentParser(
        prog="alloc",
        description=(
            "Multi-trial portfolio allocation training workflow.\n"
            "Trains multiple DDPG actor-critic candidates, scores them by\n"
            "combined Sharpe + outperformance, and recommends the best allocation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --tickers AAPL,META --positions-values '{\"AAPL\":50000,\"META\":50000}'\n"
            "  %(prog)s --tickers NVDA,GOOG --positions-values '{\"NVDA\":30000,\"GOOG\":70000}' "
            "--iterations 10 --conservative\n"
        ),
    )

    # --- Required inputs ---
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated list of ticker symbols (e.g. AAPL,META,GOOG)",
    )
    parser.add_argument(
        "--positions-values",
        required=True,
        type=_json_string,
        help=(
            "JSON object mapping ticker → dollar value "
            "(e.g. '{\"AAPL\": 50000, \"META\": 50000}')"
        ),
    )

    # --- Training loop ---
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=1,
        help="Number of complete training iterations (default: 1)",
    )
    parser.add_argument(
        "--update-iterations",
        type=_non_negative_int,
        default=1,
        help="Max update steps per iteration (default: 1, 0 = fresh-only)",
    )

    # --- Data window ---
    parser.add_argument(
        "--trading-days",
        type=_positive_int,
        default=222,
        help="Number of trading days for model training (default: 222)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=22,
        help="Batch size for model training (default: 22)",
    )

    # --- Allocation constraints ---
    parser.add_argument(
        "--min-allocation",
        type=_fraction,
        default=0.001,
        help="Minimum allocation fraction per asset (default: 0.001)",
    )
    parser.add_argument(
        "--concentration-penalty",
        type=_non_negative_float,
        default=0.001,
        help="Penalty applied for concentrated positions (default: 0.001)",
    )
    parser.add_argument(
        "--transaction-cost",
        type=_non_negative_float,
        default=0.0,
        help="Transaction cost factor (default: 0)",
    )
    parser.add_argument(
        "--risk-aversion",
        type=_non_negative_float,
        default=0.001,
        help="Risk aversion parameter (default: 0.001)",
    )
    parser.add_argument(
        "--min-cash-allocation",
        type=_fraction,
        default=0.05,
        help="Minimum cash allocation fraction (default: 0.05)",
    )

    # --- Target metrics ---
    parser.add_argument(
        "--target-sharpe",
        type=_positive_float,
        default=2.1,
        help="Target Sharpe ratio (default: 2.1)",
    )
    parser.add_argument(
        "--target-value",
        type=_positive_float,
        default=220_000.0,
        help="Target final portfolio value (default: 220000)",
    )
    parser.add_argument(
        "--target-outperformance",
        type=_positive_float,
        default=15.0,
        help="Target outperformance percentage (default: 15.0)",
    )

    # --- Flags ---
    parser.add_argument(
        "--fresh-only",
        action="store_true",
        help=(
            "Train fresh models each run (skip continue-training updates). "
            "Equivalent to setting --update-iterations 0."
        ),
    )
    parser.add_argument(
        "--conservative",
        action="store_true",
        help="Use conservative (non-aggressive) training settings",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging",
    )

    return parser


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with all TrainingConfig fields populated.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Post-parse: split tickers
    args.ticker_list = [
        t.strip().upper() for t in args.tickers.split(",") if t.strip()
    ]
    if not args.ticker_list:
        parser.error("--tickers must contain at least one ticker")

    # Post-parse: fresh-only mode
    if args.fresh_only:
        args.update_iterations = 0
        logger.info(
            "fresh-only mode: update_iterations set to 0"
        )

    return args


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------


def build_config(args: argparse.Namespace) -> TrainingConfig:
    """Construct a :class:`TrainingConfig` from parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Output of :func:`parse_args`.

    Returns
    -------
    TrainingConfig
        Fully populated configuration object.

    Raises
    ------
    ValueError
        If tickers list is empty, positions contain zero/negative values,
        or positions JSON is invalid.
    """
    # Validate tickers list is not empty
    if not args.ticker_list:
        raise ValueError(
            "Tickers list is empty. Provide at least one ticker via --tickers."
        )

    # Validate positions is a non-empty dict
    if not isinstance(args.positions_values, dict):
        type_name = type(args.positions_values).__name__
        raise ValueError(
            f"Positions must be a JSON object (dict), got {type_name}. "
            "Use --positions-values with valid JSON."
        )

    if not args.positions_values:
        raise ValueError(
            "Positions dictionary is empty. "
            "Provide at least one position via --positions-values."
        )

    # Validate no zero or negative position values
    for ticker, value in args.positions_values.items():
        if value <= 0:
            raise ValueError(
                f"Position value for '{ticker}' is {value}. "
                "All position values must be strictly positive."
            )

    return TrainingConfig(
        tickers=args.ticker_list,
        positions=args.positions_values,
        iterations=args.iterations,
        update_iterations=args.update_iterations,
        trading_days=args.trading_days,
        batch_size=args.batch_size,
        min_allocation=args.min_allocation,
        concentration_penalty=args.concentration_penalty,
        transaction_cost=args.transaction_cost,
        risk_aversion=args.risk_aversion,
        min_cash_alloc=args.min_cash_allocation,
        target_sharpe=args.target_sharpe,
        target_value=args.target_value,
        target_outperformance=args.target_outperformance,
    )


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------


def print_results(result: "WorkflowResult") -> None:
    """Print workflow results in a structured, human-readable format.

    Handles empty workflow results gracefully by logging an informative
    message instead of crashing.

    Parameters
    ----------
    result : WorkflowResult
        The result returned by :meth:`WorkflowRunner.run`.
    """
    best = result.best_trial

    # Handle missing best_trial
    if best is None:
        logger.warning("No best trial in results")
        return

    # Handle placeholder/empty best_trial (iteration=0 means no real trial ran)
    if best.iteration == 0 and not best.allocation:
        logger.warning(
            "Workflow completed but no valid trial results were produced. "
            "Status: %s",
            result.status,
        )
        return

    # Handle empty trials list (log warning but still show best_trial if valid)
    if not result.trials:
        logger.warning(
            "Workflow completed with no trials recorded. "
            "Status: %s. Showing best_trial from result.",
            result.status,
        )

    logger.info("=" * 60)
    logger.info("WORKFLOW COMPLETE")
    logger.info("=" * 60)

    # Best trial metrics
    logger.info(
        "Best trial #%d: Sharpe=%.3f  Outperformance=%.1f%%  "
        "FinalValue=$%.0f  ModelROI=%.1f%%  BuyHoldROI=%.1f%%",
        best.iteration,
        best.sharpe_ratio or 0.0,
        best.outperformance or 0.0,
        best.final_value or 0.0,
        best.model_roi or 0.0,
        best.buyhold_roi or 0.0,
    )

    # Allocation
    alloc = best.allocation
    if alloc:
        logger.info("Recommended allocation: %s", alloc)

    # Recommended trades
    trades = best.recommended_trades
    if trades:
        logger.info("Recommended trades:")
        for trade in trades:
            ticker = trade.get("ticker", "?")
            action = trade.get("action", "hold").upper()
            alloc_w = trade.get("allocation", 0.0)
            change = trade.get("change", 0.0)
            sign = "+" if change >= 0 else ""
            logger.info(
                "  %-8s %s  alloc=%.4f  change=%s%.4f",
                ticker, action, alloc_w, sign, change,
            )

    # Concentration
    conc = result.concentration
    if conc:
        logger.info(
            "Concentration — max_weight=%.3f  herfindahl=%.3f",
            conc.get("max_weight", 0.0),
            conc.get("herfindahl", 0.0),
        )

    # Trial summary
    trials = result.trials
    logger.info("Total trials completed: %d", len(trials))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the multi-trial training workflow.

    Parses arguments, builds configuration, instantiates a
    :class:`WorkflowRunner`, executes the workflow, prints results,
    and returns an appropriate exit code.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code: 0 = success, 1 = user error, 2 = workflow failure,
        3 = unexpected exception.
    """
    # --- Parse ---
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        # argparse calls sys.exit on --help or error
        code = exc.code if exc.code is not None else EXIT_USER_ERROR
        return code if isinstance(code, int) else EXIT_USER_ERROR

    # --- Logging ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    logger.info(
        "alloc training workflow — tickers=%s, iterations=%d, "
        "update_iterations=%d, conservative=%s",
        args.ticker_list,
        args.iterations,
        args.update_iterations,
        args.conservative,
    )

    # --- Build config ---
    config = build_config(args)

    # --- Resolve trainer ---
    # The trainer is the SimulationRunner-based callable that accepts
    # TrainingConfig kwargs and returns a trial result dict.
    # We import here to avoid circular dependencies at module load time.
    try:
        from alloc.core import create_trainer
    except ImportError:
        logger.error(
            "Cannot import alloc.core.create_trainer. "
            "Ensure alloc.core is installed and up to date."
        )
        return EXIT_USER_ERROR

    trainer = create_trainer(conservative=args.conservative)

    # --- Run workflow ---
    runner = WorkflowRunner(config=config, trainer=trainer)

    try:
        result = runner.run()
    except Exception as exc:
        logger.error("Workflow failed: %s", exc, exc_info=True)
        return EXIT_WORKFLOW_FAIL

    # --- Print results ---
    print_results(result)

    # --- Exit code ---
    if result.status != "success":
        logger.error("Workflow status: %s", result.status)
        return EXIT_WORKFLOW_FAIL

    return EXIT_SUCCESS


# ---------------------------------------------------------------------------
# Module execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
