"""alloc.core — Simulation runner, CLI entry point, and results serialization.

Provides :class:`SimulationRunner` (DDPG portfolio simulation loop),
:func:`parse_args` / :func:`main` (CLI entry point), and
:func:`serialize_results` / :func:`save_results` / :func:`load_results`
(results persistence helpers).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

from alloc.config.settings import get_settings
from alloc.lib.cache import DiskCache
from alloc.lib.client import PolygonClient
from alloc.models import data as data_module
from alloc.models.data import StateBuilder
from alloc.models.networks import ActorCriticNetworks
from alloc.models.portfolio import Portfolio, calculate_portfolio_reward

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TICKET-012 — Results serialization helpers
# ---------------------------------------------------------------------------


def serialize_results(results: Any) -> Any:
    """Recursively convert numpy arrays to Python lists for JSON serialisation.

    Parameters
    ----------
    results : dict
        Results dictionary potentially containing numpy arrays, floats,
        strings, nested dicts, and lists.

    Returns
    -------
    dict
        Same structure with all numpy arrays replaced by plain lists.
    """
    if isinstance(results, dict):
        return {k: serialize_results(v) for k, v in results.items()}
    if isinstance(results, list):
        return [serialize_results(item) for item in results]
    if isinstance(results, np.ndarray):
        return results.tolist()
    if isinstance(results, (np.floating, np.integer)):
        return float(results)
    return results


def save_results(
    results: dict[str, Any],
    path: str,
    mode: str = "backtest",
) -> None:
    """Write serialised results to a JSON file.

    Parameters
    ----------
    results : dict
        Results dictionary (will be serialised via :func:`serialize_results`).
    path : str
        Directory path where the JSON file will be written.
    mode : str
        ``"backtest"`` → ``{path}/backtest_results.json``
        ``"predict"``  → ``{path}/prediction_results.json``
    """
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "predict":
        filename = "prediction_results.json"
    else:
        filename = "backtest_results.json"

    filepath = output_dir / filename
    serialised = serialize_results(results)

    with open(filepath, "w") as fh:
        json.dump(serialised, fh, indent=2, default=str)

    logger.info("Results saved to %s", filepath)


def load_results(path: str, mode: str = "backtest") -> dict[str, Any]:
    """Load results from a JSON file.

    Parameters
    ----------
    path : str
        Directory path containing the results file.
    mode : str
        ``"backtest"`` → reads ``{path}/backtest_results.json``
        ``"predict"``  → reads ``{path}/prediction_results.json``

    Returns
    -------
    dict
        Deserialised results dictionary.

    Raises
    ------
    FileNotFoundError
        If the expected results file does not exist.
    """
    output_dir = Path(path)

    if mode == "predict":
        filename = "prediction_results.json"
    else:
        filename = "backtest_results.json"

    filepath = output_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    with open(filepath) as fh:
        return cast(dict[str, Any], json.load(fh))


# ---------------------------------------------------------------------------
# TICKET-009 — SimulationRunner
# ---------------------------------------------------------------------------


class SimulationRunner:
    """DDPG portfolio allocation simulation runner.

    Encapsulates the per-day simulation loop: fetch prices, build state,
    actor proposes allocation, portfolio executes trades, calculate reward,
    update networks, and track history.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to trade.
    initial_value : float
        Starting capital.
    networks : ActorCriticNetworks
        The DDPG actor-critic model.
    data_pipeline : Any
        Module providing data fetching functions (alloc.models.data).
    client : Any
        PolygonClient (or compatible) instance.
    transaction_cost : float
        Fractional cost per traded value.
    risk_aversion : float
        Weight on volatility penalty in reward.
    gamma : float
        Discount factor for TD learning.
    tau : float
        Soft-update coefficient for target networks.
    diversification_weight : float
        Multiplier on diversification bonus.
    concentration_penalty : float
        Multiplier on concentration penalty.
    min_cash : float
        Minimum cash fraction enforced by the actor.
    batch_size : int
        Mini-batch size for replay buffer sampling.
    verbose : bool
        If ``True``, log per-day diagnostics.
    """

    def __init__(
        self,
        tickers: list[str],
        initial_value: float,
        networks: ActorCriticNetworks,
        data_pipeline: Any,
        client: Any,
        transaction_cost: float = 0.001,
        risk_aversion: float = 0.5,
        gamma: float = 0.95,
        tau: float = 0.01,
        diversification_weight: float = 0.05,
        concentration_penalty: float = 0.02,
        min_cash: float = 0.05,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        self.tickers = list(tickers)
        self.initial_value = float(initial_value)
        self.networks = networks
        self.data_pipeline = data_pipeline
        self.client = client
        self.transaction_cost = float(transaction_cost)
        self.risk_aversion = float(risk_aversion)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.diversification_weight = float(diversification_weight)
        self.concentration_penalty = float(concentration_penalty)
        self.min_cash = float(min_cash)
        self.batch_size = int(batch_size)
        self.verbose = verbose

        # Portfolio instance — created fresh each run
        self.portfolio: Portfolio | None = None

        logger.info(
            "SimulationRunner initialised: tickers=%s, initial_value=%.2f",
            self.tickers,
            self.initial_value,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_state(
        self,
        multi_freq: dict[str, dict[str, list[float]]],
        alloc_list: list[float],
    ) -> np.ndarray:
        """Build state vector using StateBuilder (TICKET-043).

        Replaces the legacy ``data_pipeline.build_state_vector`` call with
        the OOP ``StateBuilder`` for consistent, testable state construction.

        Parameters
        ----------
        multi_freq : dict
            Multi-frequency price data from ``data_pipeline.get_multi_asset_data``.
        alloc_list : list[float]
            Current allocation weights per ticker.

        Returns
        -------
        np.ndarray
            1-D float64 state vector.
        """
        builder = StateBuilder(
            hourly_window=5,
            daily_window=5,
            weekly_window=5,
        )
        return builder.build_state(multi_freq, alloc_list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, trading_days: int) -> dict[str, Any]:
        """Execute the simulation for *trading_days* steps.

        Parameters
        ----------
        trading_days : int
            Number of trading days to simulate.

        Returns
        -------
        dict
            Simulation results with keys: ``final_value``,
            ``initial_value``, ``portfolio_values``, ``daily_returns``,
            ``rewards``, ``allocation_history``, ``dates``,
            ``final_holdings``, ``final_prices``.

            ``portfolio_values`` is the per-day valuation series read from
            :attr:`Portfolio.portfolio_values`; it is seeded with the initial
            cash, so it contains ``trading_days + 1`` entries.
        """
        self.portfolio = Portfolio(
            tickers=self.tickers,
            initial_cash=self.initial_value,
            transaction_cost=self.transaction_cost,
        )

        daily_returns: list[float] = []
        rewards: list[float] = []
        allocation_history: list[dict[str, float]] = []
        dates: list[str] = []

        previous_allocation: np.ndarray | None = None
        previous_state: np.ndarray | None = None
        previous_value: float | None = None

        exploration_days = max(1, trading_days // 2)

        current_date = datetime.today()

        for day in range(trading_days):
            date_str = current_date.strftime("%Y-%m-%d")

            # 1. Fetch current prices
            prices = self.data_pipeline.fetch_latest_prices(
                self.tickers, self.client
            )

            # 2. Build state vector
            multi_freq = self.data_pipeline.get_multi_asset_data(
                self.tickers, self.client, end_date=current_date
            )

            current_alloc_dict = self.portfolio.get_allocation(prices)
            alloc_list = [
                current_alloc_dict.get(t, 0.0) for t in self.tickers
            ]

            state = self._build_state(multi_freq, alloc_list)

            # Ensure state matches network input_dim
            if state.shape[0] != self.networks.input_dim:
                if state.shape[0] < self.networks.input_dim:
                    state = np.pad(
                        state,
                        (0, self.networks.input_dim - state.shape[0]),
                        mode="constant",
                    )
                else:
                    state = state[: self.networks.input_dim]

            # 3. Actor proposes allocation
            is_exploration = day < exploration_days
            allocation = self.networks.get_allocation(
                state, add_noise=is_exploration
            )

            # Build target allocation dict (tickers + cash)
            target_alloc: dict[str, float] = {}
            for i, t in enumerate(self.tickers):
                target_alloc[t] = float(allocation[i])
            target_alloc["cash"] = float(allocation[-1])

            # 4. Portfolio executes trades
            self.portfolio.execute_trades(target_alloc, prices)

            # 5. Calculate reward
            current_value = self.portfolio.get_portfolio_value(prices)

            # Record the per-day valuation so portfolio.portfolio_values
            # mirrors the per-day series (issue #111).
            self.portfolio.record_value(prices)

            if previous_value is not None and previous_value > 0:
                day_return = (current_value - previous_value) / previous_value
            else:
                day_return = 0.0

            # Build allocation vectors for reward calculation
            current_alloc_vec = np.array(
                [current_alloc_dict.get(t, 0.0) for t in self.tickers]
                + [current_alloc_dict.get("cash", 0.0)]
            )

            if previous_allocation is not None:
                # Compute returns for reward
                returns_vec = np.zeros(len(self.tickers), dtype=np.float64)
                for i, t in enumerate(self.tickers):
                    returns_vec[i] = day_return / max(len(self.tickers), 1)

                reward = calculate_portfolio_reward(
                    current=current_alloc_vec,
                    previous=previous_allocation,
                    returns=returns_vec,
                    risk_aversion=self.risk_aversion,
                    transaction_cost=self.transaction_cost,
                    diversification_weight=self.diversification_weight,
                    concentration_penalty=self.concentration_penalty,
                )
            else:
                reward = 0.0

            # 6. Build next state
            next_multi_freq = self.data_pipeline.get_multi_asset_data(
                self.tickers, self.client, end_date=current_date
            )
            next_alloc_dict = self.portfolio.get_allocation(prices)
            next_alloc_list = [
                next_alloc_dict.get(t, 0.0) for t in self.tickers
            ]
            next_state = self._build_state(next_multi_freq, next_alloc_list)
            if next_state.shape[0] != self.networks.input_dim:
                if next_state.shape[0] < self.networks.input_dim:
                    next_state = np.pad(
                        next_state,
                        (0, self.networks.input_dim - next_state.shape[0]),
                        mode="constant",
                    )
                else:
                    next_state = next_state[: self.networks.input_dim]

            # 7. Update networks via replay buffer (except last day)
            if previous_state is not None and day < trading_days - 1:
                self.networks.replay_buffer.add(
                    state=previous_state,
                    action=allocation,
                    reward=reward,
                    next_state=next_state,
                )

                # Sample and train if enough data
                if len(self.networks.replay_buffer) >= self.batch_size:
                    batch_states, batch_actions, batch_rewards, \
                        batch_next_states = self.networks.replay_buffer.sample(  # noqa: E501
                        self.batch_size
                    )

                    # TD target via critic target
                    with np.errstate(invalid="ignore"):
                        target_q = self.networks.critic_target.predict(
                            [
                                batch_next_states,
                                self.networks.actor_target.predict(
                                    batch_next_states, verbose=0,
                                ),
                            ],
                            verbose=0,
                        ).flatten()

                    td_targets = batch_rewards + self.gamma * target_q

                    # Update critic
                    self.networks.critic.train_on_batch(
                        [batch_states, batch_actions], td_targets,
                    )

                    # Update actor
                    action_preds = self.networks.actor.predict(
                        batch_states, verbose=0,
                    )
                    self.networks.actor.train_on_batch(
                        batch_states,
                        self.networks.critic.predict(
                            [batch_states, action_preds], verbose=0,
                        ),
                    )

                    # Soft update targets
                    self.networks._soft_update_targets()

            # 8. Track history
            daily_returns.append(day_return)
            rewards.append(reward)
            allocation_history.append(
                {t: float(target_alloc.get(t, 0.0)) for t in self.tickers}
                | {"cash": float(target_alloc.get("cash", 0.0))}
            )
            dates.append(date_str)

            if self.verbose:
                logger.info(
                    "Day %d (%s): value=%.2f, return=%.4f, reward=%.4f",
                    day + 1,
                    date_str,
                    current_value,
                    day_return,
                    reward,
                )

            # Advance state for next iteration
            previous_allocation = current_alloc_vec.copy()
            previous_state = state.copy()
            previous_value = current_value
            current_date -= timedelta(days=1)

        # Final prices
        final_prices = self.data_pipeline.fetch_latest_prices(
            self.tickers, self.client,
        )

        # Final holdings
        final_holdings = dict(self.portfolio.shares_held)

        # Per-day valuation series (issue #111): seeded with initial cash,
        # so it has trading_days + 1 entries.
        portfolio_values = list(self.portfolio.portfolio_values)

        results: dict[str, Any] = {
            "final_value": portfolio_values[-1]
            if portfolio_values else self.initial_value,
            "initial_value": self.initial_value,
            "portfolio_values": portfolio_values,
            "daily_returns": daily_returns,
            "rewards": rewards,
            "allocation_history": allocation_history,
            "dates": dates,
            "final_holdings": final_holdings,
            "final_prices": final_prices,
        }

        logger.info(
            "Simulation complete: initial=%.2f, final=%.2f, days=%d",
            self.initial_value,
            results["final_value"],
            trading_days,
        )

        return results


# ---------------------------------------------------------------------------
# TICKET-010 — CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="alloc — DDPG portfolio allocation engine",
    )

    # Mutually exclusive mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest simulation",
    )
    mode_group.add_argument(
        "--predict",
        action="store_true",
        help="Run prediction (forward) simulation",
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Ticker symbols to trade",
    )
    parser.add_argument(
        "--initial-value",
        type=float,
        default=100_000.0,
        help="Starting capital (default: 100000)",
    )
    parser.add_argument(
        "--trading-days",
        type=int,
        default=242,
        help="Number of trading days (default: 242)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="saved_models/portfolio_model",
        help="Path to save/load model weights",
    )
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=0.0001,
        help="Actor learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=0.0005,
        help="Critic learning rate (default: 0.0005)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor (default: 0.95)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.01,
        help="Soft-update coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=0.5,
        help="Risk aversion weight (default: 0.5)",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Transaction cost fraction (default: 0.001)",
    )
    parser.add_argument(
        "--diversification-weight",
        type=float,
        default=0.05,
        help="Diversification bonus weight (default: 0.05)",
    )
    parser.add_argument(
        "--concentration-penalty",
        type=float,
        default=0.02,
        help="Concentration penalty weight (default: 0.02)",
    )
    parser.add_argument(
        "--min-cash",
        type=float,
        default=0.05,
        help="Minimum cash fraction (default: 0.05)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Replay batch size (default: 32)",
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=50_000,
        help="Replay buffer capacity (default: 50000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Parses arguments, initialises dependencies, runs the simulation,
    serialises results, and optionally saves the model.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to ``sys.argv[1:]``).
    """
    args = parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    mode_str = "backtest" if args.backtest else "predict"
    logger.info(
        "alloc starting — mode=%s, tickers=%s", mode_str, args.tickers,
    )

    # Load settings
    settings = get_settings()

    # Initialize cache and client
    cache = DiskCache(
        cache_dir=settings.cache_dir,
        enabled=settings.cache_enabled,
    )
    client = PolygonClient(
        api_key=settings.polygon_api_key,
        cache=cache,
    )

    # Compute input dimension from tickers
    n_hourly = 5
    n_daily = 5
    n_weekly = 5
    input_dim = len(args.tickers) * (
        n_hourly + n_daily + n_weekly
    ) + len(args.tickers)

    # Initialize networks
    networks = ActorCriticNetworks(
        input_dim=input_dim,
        num_assets=len(args.tickers) + 1,  # +1 for cash
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        min_cash_allocation=args.min_cash,
        buffer_capacity=args.replay_capacity,
    )

    # Initialize runner
    runner = SimulationRunner(
        tickers=args.tickers,
        initial_value=args.initial_value,
        networks=networks,
        data_pipeline=data_module,
        client=client,
        transaction_cost=args.transaction_cost,
        risk_aversion=args.risk_aversion,
        gamma=args.gamma,
        tau=args.tau,
        diversification_weight=args.diversification_weight,
        concentration_penalty=args.concentration_penalty,
        min_cash=args.min_cash,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    # Run simulation
    results = runner.run(trading_days=args.trading_days)

    # Log metrics
    if results["portfolio_values"]:
        total_return = (
            (results["final_value"] - results["initial_value"])
            / results["initial_value"]
            * 100
        )
        logger.info(
            "Final value: %.2f | Total return: %.2f%%",
            results["final_value"],
            total_return,
        )

    # Serialize results
    mode = "backtest" if args.backtest else "predict"
    save_results(results, args.model_path, mode=mode)

    # Save model weights (backtest mode)
    if args.backtest:
        model_dir = Path(args.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        networks.actor.save_weights(model_dir / "actor_weights.h5")
        networks.critic.save_weights(model_dir / "critic_weights.h5")
        logger.info("Model weights saved to %s", model_dir)

    logger.info("alloc finished")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# TICKET-034 — create_trainer: bridge WorkflowRunner → SimulationRunner
# ---------------------------------------------------------------------------


def create_trainer(conservative: bool = False) -> Callable[..., dict[str, Any]]:
    """Factory that returns a callable trainer bridging WorkflowRunner to SimulationRunner.

    The returned callable accepts keyword arguments matching the subset of
    :class:`TrainingConfig` fields that ``WorkflowRunner._run_trial`` passes
    through, and returns a dict with trial-level metrics suitable for
    :class:`TrainingTrial` construction.

    Parameters
    ----------
    conservative : bool
        If ``True``, use conservative hyper-parameters (lower learning
        rates, higher risk aversion).  Defaults to ``False`` (aggressive).

    Returns
    -------
    Callable[..., dict]
        A trainer function accepting tickers, positions, update_iterations,
        trading_days, batch_size, min_allocation, concentration_penalty,
        transaction_cost, risk_aversion, min_cash_alloc, target_sharpe,
        target_value, target_outperformance — and returning a dict with
        keys: sharpe_ratio, outperformance, final_value, model_roi,
        buyhold_roi, allocation, model_path, results_path, update.
    """

    def _trainer(
        tickers: list[str],
        positions: dict[str, float],
        update_iterations: int = 1,
        trading_days: int = 222,
        batch_size: int = 22,
        min_allocation: float = 0.001,
        concentration_penalty: float = 0.001,
        transaction_cost: float = 0.0,
        risk_aversion: float = 0.001,
        min_cash_alloc: float = 0.05,
        target_sharpe: float = 2.1,
        target_value: float = 220_000.0,
        target_outperformance: float = 15.0,
    ) -> dict[str, Any]:
        """Run a single training trial via SimulationRunner.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols to trade.
        positions : dict[str, float]
            Initial dollar-value positions keyed by ticker.
        update_iterations : int
            Number of update steps (0 = fresh model each time).
        trading_days : int
            Trading days for the simulation window.
        batch_size : int
            Replay batch size.
        min_allocation : float
            Minimum allocation per asset.
        concentration_penalty : float
            Concentration penalty weight.
        transaction_cost : float
            Transaction cost fraction.
        risk_aversion : float
            Risk aversion weight.
        min_cash_alloc : float
            Minimum cash allocation fraction.
        target_sharpe : float
            Target Sharpe ratio.
        target_value : float
            Target final portfolio value.
        target_outperformance : float
            Target outperformance percentage.

        Returns
        -------
        dict
            Trial result dict with keys suitable for TrainingTrial.
        """
        # Compute initial value from positions
        initial_value = sum(positions.values()) if positions else 100_000.0

        # Conservative vs aggressive hyper-parameters
        if conservative:
            actor_lr = 0.00005
            critic_lr = 0.0002
            gamma = 0.99
            diversification_weight = 0.1
        else:
            actor_lr = 0.0001
            critic_lr = 0.0005
            gamma = 0.95
            diversification_weight = 0.05

        # Compute input dimension
        n_hourly = 5
        n_daily = 5
        n_weekly = 5
        input_dim = len(tickers) * (n_hourly + n_daily + n_weekly) + len(tickers)
        num_assets = len(tickers) + 1  # +1 for cash

        # Load settings and build dependencies
        settings = get_settings()
        cache = DiskCache(
            cache_dir=settings.cache_dir,
            enabled=settings.cache_enabled,
        )
        client = PolygonClient(
            api_key=settings.polygon_api_key,
            cache=cache,
        )

        # Build networks
        networks = ActorCriticNetworks(
            input_dim=input_dim,
            num_assets=num_assets,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            tau=0.01,
            min_cash_allocation=min_cash_alloc,
            buffer_capacity=50_000,
        )

        # Build runner
        runner = SimulationRunner(
            tickers=tickers,
            initial_value=initial_value,
            networks=networks,
            data_pipeline=data_module,
            client=client,
            transaction_cost=transaction_cost,
            risk_aversion=risk_aversion,
            gamma=gamma,
            tau=0.01,
            diversification_weight=diversification_weight,
            concentration_penalty=concentration_penalty,
            min_cash=min_cash_alloc,
            batch_size=batch_size,
            verbose=False,
        )

        # Run simulation
        results = runner.run(trading_days=trading_days)

        # Extract metrics
        final_value = results.get("final_value", initial_value)

        # Return metrics come from the portfolio's own valuation series
        # (issue #111).  Portfolio.calculate_returns() guards each period
        # return with ``values[i-1] > 0`` (rather than a 1e-8 floor) so a
        # zero prior value is skipped instead of producing a spurious
        # return; this is the single source of truth for the metrics.
        if runner.portfolio is None:
            raise RuntimeError(
                "SimulationRunner.portfolio is not initialised after run()"
            )
        returns_metrics = runner.portfolio.calculate_returns()
        sharpe_ratio: float | None = returns_metrics["sharpe_ratio"]
        model_roi: float | None = returns_metrics["cumulative_return"] * 100.0
        max_drawdown: float = returns_metrics["max_drawdown"]
        annualized_return: float = returns_metrics["annualized_return"]

        buyhold_roi: float | None = None

        # Buy-and-hold ROI (from results if available)
        buyhold_values = results.get("buyhold_values", [])
        if buyhold_values and len(buyhold_values) > 0:
            bh_final = buyhold_values[-1]
            buyhold_roi = float((bh_final - initial_value) / initial_value * 100)

        # Outperformance
        outperformance: float | None = None
        if model_roi is not None and buyhold_roi is not None:
            outperformance = float(model_roi - buyhold_roi)

        # Final allocation
        allocation: list[float] = []
        allocation_history = results.get("allocation_history", [])
        if allocation_history:
            last_alloc = allocation_history[-1]
            if isinstance(last_alloc, dict):
                allocation = (
                    [last_alloc.get(t, 0.0) for t in tickers]
                    + [last_alloc.get("cash", 0.0)]
                )
            else:
                if hasattr(last_alloc, "tolist"):
                    allocation = last_alloc.tolist()
                else:
                    allocation = list(last_alloc)

        # Derive recommended_trades from allocation_history
        recommended_trades: list[dict] | None = None
        if len(allocation_history) >= 2:
            prev_alloc = allocation_history[-2]
            curr_alloc = allocation_history[-1]
            if isinstance(prev_alloc, dict) and isinstance(curr_alloc, dict):
                recommended_trades = []
                for t in tickers:
                    prev_w = prev_alloc.get(t, 0.0)
                    curr_w = curr_alloc.get(t, 0.0)
                    change = curr_w - prev_w
                    if abs(change) < 1e-6:
                        action = "hold"
                    elif change > 0:
                        action = "buy"
                    else:
                        action = "sell"
                    recommended_trades.append({
                        "ticker": t,
                        "action": action,
                        "allocation": round(curr_w, 6),
                        "change": round(change, 6),
                    })
                # Include cash
                prev_cash = prev_alloc.get("cash", 0.0)
                curr_cash = curr_alloc.get("cash", 0.0)
                cash_change = curr_cash - prev_cash
                if abs(cash_change) < 1e-6:
                    cash_action = "hold"
                elif cash_change > 0:
                    cash_action = "buy"
                else:
                    cash_action = "sell"
                recommended_trades.append({
                    "ticker": "cash",
                    "action": cash_action,
                    "allocation": round(curr_cash, 6),
                    "change": round(cash_change, 6),
                })
        elif len(allocation_history) == 1:
            # Only one allocation — derive from final_holdings
            final_holdings = results.get("final_holdings", {})
            if final_holdings:
                recommended_trades = []
                for t in tickers:
                    shares = final_holdings.get(t, 0)
                    action = "buy" if shares > 0 else "hold"
                    recommended_trades.append({
                        "ticker": t,
                        "action": action,
                        "allocation": 0.0,
                        "change": 0.0,
                    })

        return {
            "sharpe_ratio": sharpe_ratio,
            "outperformance": outperformance,
            "final_value": final_value,
            "model_roi": model_roi,
            "buyhold_roi": buyhold_roi,
            "max_drawdown": max_drawdown,
            "annualized_return": annualized_return,
            "allocation": allocation,
            "recommended_trades": recommended_trades,
            "model_path": None,
            "results_path": None,
            "update": update_iterations,
        }

    return _trainer
