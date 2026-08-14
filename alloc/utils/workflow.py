"""alloc.utils.workflow — Multi-trial training orchestration.

Provides :class:`TrainingConfig`, :class:`TrainingTrial`,
:class:`WorkflowResult`, and :class:`WorkflowRunner` for running
multiple training trials, scoring them, and analysing allocation
statistics and concentration metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for a multi-trial training workflow.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols to include in the portfolio.
    positions : dict[str, float]
        Initial dollar-value positions keyed by ticker.
    iterations : int
        Number of complete training iterations.
    update_iterations : int
        Number of update steps per iteration.
    trading_days : int
        Number of trading days used for model training.
    batch_size : int
        Batch size for model training.
    min_allocation : float
        Minimum allocation fraction per asset.
    concentration_penalty : float
        Penalty applied for concentrated positions.
    transaction_cost : float
        Transaction cost factor.
    risk_aversion : float
        Risk aversion parameter.
    min_cash_alloc : float
        Minimum cash allocation fraction.
    target_sharpe : float
        Target Sharpe ratio.
    target_value : float
        Target final portfolio value.
    target_outperformance : float
        Target outperformance percentage.

    Raises
    ------
    ValueError
        If tickers list is empty, positions dict is empty, or any
        position value is zero or negative.
    """

    tickers: list[str]
    positions: dict[str, float]
    iterations: int = 1
    update_iterations: int = 1
    trading_days: int = 222
    batch_size: int = 22
    min_allocation: float = 0.001
    concentration_penalty: float = 0.001
    transaction_cost: float = 0.0
    risk_aversion: float = 0.001
    min_cash_alloc: float = 0.05
    target_sharpe: float = 2.1
    target_value: float = 220_000.0
    target_outperformance: float = 15.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises
        ------
        ValueError
            If tickers list is empty, positions dict is empty, any
            position value is zero or negative, or there is a mismatch
            between the set of tickers and the set of position keys.
        """
        if not self.tickers:
            raise ValueError(
                "Tickers list is empty. Provide at least one ticker."
            )

        if not self.positions:
            raise ValueError(
                "Positions dictionary is empty. Provide at least one position."
            )

        for ticker, value in self.positions.items():
            if value <= 0:
                raise ValueError(
                    f"Position value for '{ticker}' is {value}. "
                    "All position values must be strictly positive."
                )

        # TICKET-041: Cross-validate tickers and positions
        ticker_set = set(self.tickers)
        position_set = set(self.positions.keys())

        missing_in_positions = ticker_set - position_set
        if missing_in_positions:
            raise ValueError(
                f"Tickers without corresponding positions: "
                f"{sorted(missing_in_positions)}. "
                "Every ticker must have a matching entry in positions."
            )

        extra_in_positions = position_set - ticker_set
        if extra_in_positions:
            raise ValueError(
                f"Positions without corresponding tickers: "
                f"{sorted(extra_in_positions)}. "
                "Every position key must appear in the tickers list."
            )


@dataclass
class TrainingTrial:
    """Result of a single training trial.

    Parameters
    ----------
    iteration : int
        Trial / iteration number.
    update : int
        Update step within the iteration.
    sharpe_ratio : float | None
        Sharpe ratio achieved by the model.
    outperformance : float | None
        Outperformance percentage vs. buy-and-hold.
    final_value : float | None
        Final portfolio value.
    model_roi : float | None
        Model return on investment.
    buyhold_roi : float | None
        Buy-and-hold return on investment.
    allocation : list[float]
        Final allocation weights (one per ticker, plus cash).
    recommended_trades : list[dict] | None
        List of recommended trade actions derived from the final
        allocation step.  Each dict has keys ``ticker``, ``action``
        (``"buy"`` / ``"sell"`` / ``"hold"``), ``allocation`` (target
        weight), and ``change`` (delta vs. previous allocation).
    model_path : str | None
        Path to the saved model file.
    results_path : str | None
        Path to the saved results JSON file.
    """

    iteration: int
    update: int
    sharpe_ratio: float | None = None
    outperformance: float | None = None
    final_value: float | None = None
    model_roi: float | None = None
    buyhold_roi: float | None = None
    allocation: list[float] = field(default_factory=list)
    recommended_trades: list[dict] | None = None
    model_path: str | None = None
    results_path: str | None = None


@dataclass
class WorkflowResult:
    """Aggregated result of a full workflow run.

    Parameters
    ----------
    status : str
        Overall status (e.g. ``"success"`` or ``"error"``).
    trials : list[TrainingTrial]
        All individual trial results.
    best_trial : TrainingTrial
        The trial with the highest combined score.
    allocation_stats : dict
        Per-ticker allocation statistics (mean, std, min, max).
    concentration : dict
        Concentration metrics (max_weight, herfindahl).
    metrics_progression : list[dict]
        Ordered list of metric snapshots per trial.
    """

    status: str
    trials: list[TrainingTrial]
    best_trial: TrainingTrial
    allocation_stats: dict
    concentration: dict
    metrics_progression: list[dict]


# ---------------------------------------------------------------------------
# Workflow runner
# ---------------------------------------------------------------------------

class WorkflowRunner:
    """Orchestrates multiple training trials and aggregates results.

    Parameters
    ----------
    config : TrainingConfig
        Configuration for the workflow.
    trainer : Callable[..., dict]
        A callable that accepts keyword arguments derived from
        :class:`TrainingConfig` and returns a dict with trial results
        (e.g. ``sharpe_ratio``, ``outperformance``, ``final_value``,
        ``model_roi``, ``buyhold_roi``, ``allocation``,
        ``model_path``, ``results_path``).
    """

    def __init__(
        self,
        config: TrainingConfig,
        trainer: Callable[..., dict],
    ) -> None:
        self.config = config
        self.trainer = trainer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> WorkflowResult:
        """Execute all training trials and return aggregated results.

        Returns
        -------
        WorkflowResult
            Aggregated workflow result containing all trials,
            the best trial, allocation statistics, concentration
            metrics, and metrics progression.
        """
        trials: list[TrainingTrial] = []
        metrics_progression: list[dict] = []

        for trial_num in range(1, self.config.iterations + 1):
            logger.info("Running trial %d / %d", trial_num, self.config.iterations)
            trial = self._run_trial(trial_num)
            trials.append(trial)
            metrics_progression.append({
                "iteration": trial.iteration,
                "update": trial.update,
                "sharpe_ratio": trial.sharpe_ratio,
                "outperformance": trial.outperformance,
                "final_value": trial.final_value,
                "model_roi": trial.model_roi,
                "buyhold_roi": trial.buyhold_roi,
            })

        if not trials:
            empty_trial = TrainingTrial(iteration=0, update=0)
            return WorkflowResult(
                status="error",
                trials=[],
                best_trial=empty_trial,
                allocation_stats={},
                concentration={},
                metrics_progression=[],
            )

        best_trial = max(trials, key=self._combined_score)
        allocation_stats = self._allocation_stats(trials)
        concentration = self._concentration_metrics(trials)

        return WorkflowResult(
            status="success",
            trials=trials,
            best_trial=best_trial,
            allocation_stats=allocation_stats,
            concentration=concentration,
            metrics_progression=metrics_progression,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_trial(self, trial_num: int) -> TrainingTrial:
        """Run a single training trial.

        Parameters
        ----------
        trial_num : int
            The trial number (1-based).

        Returns
        -------
        TrainingTrial
            The result of the trial.
        """
        kwargs = {
            "tickers": self.config.tickers,
            "positions": self.config.positions,
            "update_iterations": self.config.update_iterations,
            "trading_days": self.config.trading_days,
            "batch_size": self.config.batch_size,
            "min_allocation": self.config.min_allocation,
            "concentration_penalty": self.config.concentration_penalty,
            "transaction_cost": self.config.transaction_cost,
            "risk_aversion": self.config.risk_aversion,
            "min_cash_alloc": self.config.min_cash_alloc,
            "target_sharpe": self.config.target_sharpe,
            "target_value": self.config.target_value,
            "target_outperformance": self.config.target_outperformance,
        }

        result = self.trainer(**kwargs)

        return TrainingTrial(
            iteration=trial_num,
            update=result.get("update", 0),
            sharpe_ratio=result.get("sharpe_ratio"),
            outperformance=result.get("outperformance"),
            final_value=result.get("final_value"),
            model_roi=result.get("model_roi"),
            buyhold_roi=result.get("buyhold_roi"),
            allocation=result.get("allocation", []),
            recommended_trades=result.get("recommended_trades"),
            model_path=result.get("model_path"),
            results_path=result.get("results_path"),
        )

    @staticmethod
    def _combined_score(trial: TrainingTrial) -> float:
        """Compute a combined score for ranking trials.

        Score = 0.5 × Sharpe + 0.5 × (outperformance / 10).

        Missing values default to 0.

        Parameters
        ----------
        trial : TrainingTrial
            The trial to score.

        Returns
        -------
        float
            The combined score.
        """
        sharpe = trial.sharpe_ratio if trial.sharpe_ratio is not None else 0.0
        outperf = trial.outperformance if trial.outperformance is not None else 0.0
        return 0.5 * sharpe + 0.5 * (outperf / 10.0)

    @staticmethod
    def _allocation_stats(trials: list[TrainingTrial]) -> dict:
        """Compute per-ticker allocation statistics across trials.

        Returns a dict keyed by ticker with ``mean``, ``std``, ``min``,
        and ``max`` allocation values.

        Parameters
        ----------
        trials : list[TrainingTrial]
            All completed trials.

        Returns
        -------
        dict
            Per-ticker statistics.
        """
        # Collect allocations per position index across trials
        allocations: list[list[float]] = [
            t.allocation for t in trials if t.allocation
        ]

        if not allocations:
            return {}

        # Determine number of positions (tickers + cash)
        num_positions = max(len(a) for a in allocations)

        stats: dict[str, dict[str, float]] = {}
        for idx in range(num_positions):
            values = [a[idx] for a in allocations if idx < len(a)]
            if not values:
                continue
            arr = np.array(values, dtype=np.float64)
            key = f"position_{idx}"
            stats[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        return stats

    @staticmethod
    def _concentration_metrics(trials: list[TrainingTrial]) -> dict:
        """Compute concentration metrics across trials.

        Returns ``max_weight`` (maximum single-trial allocation to any
        position) and ``herfindahl`` (mean Herfindahl-Hirschman Index
        across trials).

        Parameters
        ----------
        trials : list[TrainingTrial]
            All completed trials.

        Returns
        -------
        dict
            Concentration metrics.
        """
        allocations: list[list[float]] = [
            t.allocation for t in trials if t.allocation
        ]

        if not allocations:
            return {"max_weight": 0.0, "herfindahl": 0.0}

        all_weights: list[float] = []
        hhi_values: list[float] = []

        for alloc in allocations:
            weights = [w for w in alloc if w > 0]
            if weights:
                all_weights.extend(weights)
                hhi_values.append(float(sum(w * w for w in alloc)))

        return {
            "max_weight": float(max(all_weights)) if all_weights else 0.0,
            "herfindahl": float(np.mean(hhi_values)) if hhi_values else 0.0,
        }
