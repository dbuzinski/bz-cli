# mypy: disable-error-code="import,import-not-found"
"""
Optuna plugin for hyperparameter optimization.

This plugin integrates Optuna with the training loop to automatically
optimize hyperparameters during training.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import optuna
    from optuna import Trial
    from optuna.samplers import TPESampler
    from optuna.study import Study

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    # Type aliases for when optuna is not available
    Trial = Any  # type: ignore
    Study = Any  # type: ignore

from .plugin import Plugin, PluginContext


@dataclass
class OptunaConfig:
    """Configuration for Optuna hyperparameter optimization."""

    # Study configuration
    study_name: str = "bz_optimization"
    storage: Optional[str] = None  # SQLite database path or other storage
    sampler: str = "tpe"  # "tpe", "random", "cmaes", etc.
    pruner: Optional[str] = None  # "median", "hyperband", etc.

    # Optimization parameters
    n_trials: int = 10
    timeout: Optional[int] = None  # seconds
    direction: str = "minimize"  # "minimize" or "maximize"

    # Hyperparameter search space
    hyperparameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001

    # Reporting
    report_interval: int = 1  # Report every N trials
    save_best_params: bool = True
    save_study: bool = True


class OptunaPlugin(Plugin):
    """
    Optuna plugin for hyperparameter optimization.

    This plugin integrates Optuna with the training loop to automatically
    optimize hyperparameters. It can suggest hyperparameters for each trial
    and report results back to Optuna for optimization.
    """

    def __init__(self, config: Optional[OptunaConfig] = None, enable_gpu_monitoring: bool = True):
        """Initialize the Optuna plugin."""
        super().__init__()

        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install it with: pip install optuna")

        self.config: OptunaConfig = config or OptunaConfig()
        self.logger = logging.getLogger(__name__)

        # Optuna study
        self.study: Optional[Study] = None
        self.current_trial: Optional[Trial] = None
        self.trial_number: int = 0

        # Trial tracking
        self.best_score: Optional[float] = None
        self.trial_scores: List[float] = []
        self.trial_params: List[Dict[str, Any]] = []

        # Early stopping
        self.no_improvement_count: int = 0
        self.last_best_score: Optional[float] = None

        # Output paths
        self.output_dir = Path("optuna_results")
        self.output_dir.mkdir(exist_ok=True)

    def start_training_session(self, context: PluginContext) -> None:
        """Initialize Optuna study and start hyperparameter optimization."""
        try:
            # Create or load study
            sampler = self._create_sampler()
            pruner = self._create_pruner()

            self.study = optuna.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage,
                sampler=sampler,
                pruner=pruner,
                direction=self.config.direction,
                load_if_exists=True,
            )

            self.logger.info(
                f"Optuna study '{self.config.study_name}' initialized. "
                f"Direction: {self.config.direction}, "
                f"Trials: {self.config.n_trials}"
            )

            # Start first trial
            self._start_new_trial()

        except Exception as e:
            self.logger.error(f"Failed to initialize Optuna study: {e}")
            raise

    def start_epoch(self, context: PluginContext) -> None:
        """Handle epoch start for hyperparameter optimization."""
        if self.current_trial is None:
            return

        try:
            # Report intermediate value to Optuna
            if hasattr(context, "validation_loss") and context.validation_loss is not None:
                self.current_trial.report(context.validation_loss, step=context.epoch)

                # Check for pruning
                if self.current_trial.should_prune():
                    raise optuna.TrialPruned()

        except Exception as e:
            self.logger.warning(f"Error reporting to Optuna: {e}")

    def end_training_session(self, context: PluginContext) -> None:
        """Complete the current trial and start next one if needed."""
        try:
            if self.current_trial is None:
                return

            # Get final score (validation loss or custom metric)
            final_score = self._get_final_score(context)

            # Complete trial
            if self.study is not None:
                self.study.tell(self.current_trial, final_score)

            # Track trial results
            self.trial_scores.append(final_score)
            self.trial_params.append(self.current_trial.params)

            # Update best score
            if (
                self.best_score is None
                or (self.config.direction == "minimize" and final_score < self.best_score)
                or (self.config.direction == "maximize" and final_score > self.best_score)
            ):

                self.best_score = final_score
                self.no_improvement_count = 0

                # Save best parameters
                if self.config.save_best_params:
                    self._save_best_params()
            else:
                self.no_improvement_count += 1

            self.logger.info(
                f"Trial {self.trial_number} completed. " f"Score: {final_score:.4f}, " f"Best: {self.best_score:.4f}"
            )

            # Check if we should continue
            if self._should_continue_optimization():
                self._start_new_trial()
            else:
                self._finalize_optimization()

        except Exception as e:
            self.logger.error(f"Error completing Optuna trial: {e}")

    def _start_new_trial(self) -> None:
        """Start a new Optuna trial."""
        if self.study is None:
            return

        try:
            self.current_trial = self.study.ask()
            self.trial_number += 1

            # Suggest hyperparameters
            suggested_params = self._suggest_hyperparameters()

            self.logger.info(
                f"Starting trial {self.trial_number} with parameters: " f"{json.dumps(suggested_params, indent=2)}"
            )

            # Store suggested parameters for use in training
            self.current_trial.set_user_attr("suggested_params", suggested_params)

        except Exception as e:
            self.logger.error(f"Error starting new trial: {e}")
            raise

    def _suggest_hyperparameters(self) -> Dict[str, Any]:
        """Suggest hyperparameters for the current trial."""
        if self.current_trial is None:
            return {}

        suggested_params = {}

        for param_name, param_config in self.config.hyperparameters.items():
            param_type = param_config.get("type", "float")

            if param_type == "float":
                suggested_params[param_name] = self.current_trial.suggest_float(
                    param_name, param_config["low"], param_config["high"], log=param_config.get("log", False)
                )
            elif param_type == "int":
                suggested_params[param_name] = self.current_trial.suggest_int(
                    param_name, param_config["low"], param_config["high"], log=param_config.get("log", False)
                )
            elif param_type == "categorical":
                suggested_params[param_name] = self.current_trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            elif param_type == "loguniform":
                suggested_params[param_name] = self.current_trial.suggest_loguniform(
                    param_name, param_config["low"], param_config["high"]
                )

        return suggested_params

    def _get_final_score(self, context: PluginContext) -> float:
        """Get the final score for the current trial."""
        # Priority: validation_loss > custom metric > training_loss
        if hasattr(context, "validation_loss") and context.validation_loss is not None:
            return context.validation_loss
        elif hasattr(context, "metrics") and context.metrics:
            # Use the first metric as score
            for metric_name, metric_value in context.metrics.items():
                if isinstance(metric_value, (int, float)):
                    return float(metric_value)
        elif hasattr(context, "training_loss") and context.training_loss is not None:
            return context.training_loss

        # Fallback
        return 0.0

    def _should_continue_optimization(self) -> bool:
        """Check if optimization should continue."""
        # Check trial limit
        if self.trial_number >= self.config.n_trials:
            return False

        # Check timeout
        if self.config.timeout is not None:
            # This would need to be implemented with timing logic
            pass

        # Check early stopping
        if self.config.early_stopping_patience > 0 and self.no_improvement_count >= self.config.early_stopping_patience:
            self.logger.info(
                f"Early stopping triggered after {self.no_improvement_count} " f"trials without improvement"
            )
            return False

        return True

    def _finalize_optimization(self) -> None:
        """Finalize the optimization process."""
        if self.study is None:
            return

        try:
            # Print optimization summary
            self.logger.info("=" * 50)
            self.logger.info("OPTUNA OPTIMIZATION COMPLETED")
            self.logger.info("=" * 50)
            self.logger.info(f"Best trial: {self.study.best_trial.number}")
            self.logger.info(f"Best score: {self.study.best_value:.4f}")
            self.logger.info(f"Best parameters: {self.study.best_params}")
            self.logger.info(f"Total trials: {len(self.study.trials)}")

            # Save study
            if self.config.save_study:
                study_path = self.output_dir / f"{self.config.study_name}.pkl"
                with open(study_path, "wb") as f:
                    import pickle

                    pickle.dump(self.study, f)
                self.logger.info(f"Study saved to: {study_path}")

            # Save optimization history
            history_path = self.output_dir / f"{self.config.study_name}_history.json"
            history = {
                "trial_scores": self.trial_scores,
                "trial_params": self.trial_params,
                "best_score": self.best_score,
                "best_params": self.study.best_params,
                "total_trials": len(self.study.trials),
            }
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            self.logger.info(f"Optimization history saved to: {history_path}")

        except Exception as e:
            self.logger.error(f"Error finalizing optimization: {e}")

    def _save_best_params(self) -> None:
        """Save the best parameters found so far."""
        if self.study is None:
            return

        try:
            best_params_path = self.output_dir / f"{self.config.study_name}_best_params.json"
            with open(best_params_path, "w") as f:
                json.dump(self.study.best_params, f, indent=2)

            self.logger.info(f"Best parameters saved to: {best_params_path}")

        except ValueError as e:
            if "No trials are completed yet" in str(e):
                self.logger.info("No trials completed yet, skipping best params save")
            else:
                self.logger.error(f"Error saving best parameters: {e}")
        except Exception as e:
            self.logger.error(f"Error saving best parameters: {e}")

    def _create_sampler(self) -> Any:
        """Create the Optuna sampler."""
        if self.config.sampler == "tpe":
            return TPESampler()
        elif self.config.sampler == "random":
            return optuna.samplers.RandomSampler()
        elif self.config.sampler == "cmaes":
            return optuna.samplers.CmaEsSampler()
        else:
            self.logger.warning(f"Unknown sampler: {self.config.sampler}, using TPE")
            return TPESampler()

    def _create_pruner(self) -> Optional[Any]:
        """Create the Optuna pruner."""
        if self.config.pruner == "median":
            return optuna.pruners.MedianPruner()
        elif self.config.pruner == "hyperband":
            return optuna.pruners.HyperbandPruner()
        elif self.config.pruner == "percentile":
            return optuna.pruners.PercentilePruner(percentile=50.0)
        else:
            return None

    def get_suggested_hyperparameters(self) -> Dict[str, Any]:
        """Get the suggested hyperparameters for the current trial."""
        if self.current_trial is None:
            return {}

        return self.current_trial.user_attrs.get("suggested_params", {})

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of the optimization process."""
        if self.study is None:
            return {}

        summary = {
            "study_name": self.config.study_name,
            "total_trials": len(self.study.trials),
            "direction": self.config.direction,
            "current_trial": self.trial_number,
        }

        # Add best score and params if trials are completed
        try:
            summary["best_score"] = self.study.best_value
            summary["best_params"] = self.study.best_params
        except ValueError:
            # No trials completed yet
            summary["best_score"] = None
            summary["best_params"] = {}

        return summary
