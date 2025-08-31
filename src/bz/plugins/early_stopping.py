"""
Early stopping plugin for the bz training framework.

This plugin implements early stopping strategies to prevent overfitting
and reduce unnecessary training time.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .plugin import Plugin, PluginContext


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping plugin."""

    # Basic settings
    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.001
    monitor: str = "validation_loss"
    mode: str = "min"  # "min" or "max"
    restore_best_weights: bool = True

    # Advanced settings
    verbose: bool = True
    baseline: Optional[float] = None
    min_epochs: int = 0  # Minimum epochs before early stopping can trigger

    # Strategy-specific settings
    strategy: str = "patience"  # "patience", "plateau", "custom"

    # Plateau strategy settings
    plateau_factor: float = 0.1
    plateau_patience: int = 10
    plateau_threshold: float = 0.0001

    # Custom strategy settings
    custom_conditions: Dict[str, Any] = field(default_factory=dict)


class EarlyStoppingPlugin(Plugin):
    """
    Early stopping plugin for training optimization.

    This plugin monitors training metrics and stops training when
    the monitored metric stops improving according to the configured strategy.
    """

    name = "early_stopping"  # Plugin name for discovery

    def __init__(self, config: Optional[EarlyStoppingConfig] = None):
        """Initialize the early stopping plugin."""
        super().__init__()
        self.config: EarlyStoppingConfig = config or EarlyStoppingConfig()
        self.logger = logging.getLogger(__name__)

        # Early stopping state
        self.best_score: Optional[float] = None
        self.patience_counter: int = 0
        self.best_epoch: int = 0
        self.best_weights: Optional[Dict[str, Any]] = None

        # Plateau detection
        self.plateau_counter: int = 0
        self.last_score: Optional[float] = None

        # Validation
        if self.config.mode not in ["min", "max"]:
            raise ValueError(f"Mode must be 'min' or 'max', got {self.config.mode}")

        if self.config.strategy not in ["patience", "plateau", "custom"]:
            raise ValueError(f"Strategy must be 'patience', 'plateau', or 'custom', got {self.config.strategy}")

    def start_training_session(self, context: PluginContext) -> None:
        """Initialize early stopping at the start of training."""
        if not self.config.enabled:
            return

        self.logger.info(
            f"Early stopping enabled: {self.config.strategy} strategy, "
            f"patience={self.config.patience}, monitor={self.config.monitor}, "
            f"mode={self.config.mode}"
        )

        # Reset state
        self.best_score = None
        self.patience_counter = 0
        self.best_epoch = 0
        self.best_weights = None
        self.plateau_counter = 0
        self.last_score = None

    def end_epoch(self, context: PluginContext) -> None:
        """Check early stopping conditions at the end of each epoch."""
        if not self.config.enabled:
            return

        # Don't check before minimum epochs
        if context.epoch < self.config.min_epochs:
            return

        # Get the monitored metric
        current_score = self._get_monitored_metric(context)
        if current_score is None:
            self.logger.warning(f"Could not find monitored metric: {self.config.monitor}")
            return

        # Update best score if improved
        if self._is_improvement(current_score):
            self.best_score = current_score
            self.best_epoch = context.epoch
            self.patience_counter = 0
            self.plateau_counter = 0

            # Save best weights if requested
            if self.config.restore_best_weights:
                self._save_best_weights(context)

            if self.config.verbose:
                self.logger.info(f"New best {self.config.monitor}: {self.best_score:.6f} " f"at epoch {context.epoch}")
        else:
            self.patience_counter += 1
            if self.config.verbose and self.patience_counter > 0:
                self.logger.info(f"No improvement for {self.patience_counter} epochs. " f"Best: {self.best_score:.6f}")

        # Check if we should stop
        should_stop = self._should_stop(current_score, context.epoch)

        if should_stop:
            context.should_stop_training = True
            self.logger.info(
                f"Early stopping triggered at epoch {context.epoch}. "
                f"Best {self.config.monitor}: {self.best_score:.6f} at epoch {self.best_epoch}"
            )

            # Restore best weights if requested
            if self.config.restore_best_weights and self.best_weights is not None:
                self._restore_best_weights(context)

    def _get_monitored_metric(self, context: PluginContext) -> Optional[float]:
        """Get the value of the monitored metric from context."""
        # Check validation loss first
        if self.config.monitor == "validation_loss" and context.validation_batch_count > 0:
            return context.validation_loss_total / context.validation_batch_count

        # Check training loss
        if self.config.monitor == "training_loss" and context.training_batch_count > 0:
            return context.training_loss_total / context.training_batch_count

        # Check custom metrics
        if hasattr(context, "metrics") and context.metrics:
            return context.metrics.get(self.config.monitor)

        return None

    def _should_stop(self, current_score: float, epoch: int) -> bool:
        """Determine if training should stop based on the strategy."""
        if self.config.strategy == "patience":
            return self._patience_strategy(current_score)
        elif self.config.strategy == "plateau":
            return self._plateau_strategy(current_score)
        elif self.config.strategy == "custom":
            return self._custom_strategy(current_score, epoch)

        return False

    def _patience_strategy(self, current_score: float) -> bool:
        """Patience-based early stopping strategy."""
        if self.best_score is None:
            return False

        if self._is_improvement(current_score):
            return False

        # Check if we've exceeded patience
        return self.patience_counter >= self.config.patience

    def _plateau_strategy(self, current_score: float) -> bool:
        """Plateau detection early stopping strategy."""
        if self.best_score is None:
            return False

        if self._is_improvement(current_score):
            return False

        # Check if we're in a plateau
        if self.last_score is not None:
            score_change = abs(current_score - self.last_score)
            if score_change < self.config.plateau_threshold:
                self.plateau_counter += 1
            else:
                self.plateau_counter = 0

        self.last_score = current_score

        return self.plateau_counter >= self.config.plateau_patience

    def _custom_strategy(self, current_score: float, epoch: int) -> bool:
        """Custom early stopping strategy."""
        # This can be extended with custom logic
        # For now, fall back to patience strategy
        return self._patience_strategy(current_score)

    def _is_improvement(self, current_score: float) -> bool:
        """Check if the current score is an improvement over the best score."""
        if self.best_score is None:
            return True

        if self.config.mode == "min":
            return current_score < self.best_score - self.config.min_delta
        else:  # mode == "max"
            return current_score > self.best_score + self.config.min_delta

    def _save_best_weights(self, context: PluginContext) -> None:
        """Save the best model weights."""
        # This would need access to the model
        # For now, we'll store a reference that can be used later
        if hasattr(context, "model"):
            self.best_weights = context.model.state_dict()

    def _restore_best_weights(self, context: PluginContext) -> None:
        """Restore the best model weights."""
        if self.best_weights is not None and hasattr(context, "model"):
            context.model.load_state_dict(self.best_weights)
            self.logger.info("Restored best model weights")

    def get_best_score(self) -> Optional[float]:
        """Get the best score achieved during training."""
        return self.best_score

    def get_best_epoch(self) -> int:
        """Get the epoch where the best score was achieved."""
        return self.best_epoch

    def get_patience_counter(self) -> int:
        """Get the current patience counter."""
        return self.patience_counter

    def get_early_stopping_summary(self) -> Dict[str, Any]:
        """Get a summary of the early stopping state."""
        return {
            "enabled": self.config.enabled,
            "strategy": self.config.strategy,
            "monitor": self.config.monitor,
            "mode": self.config.mode,
            "patience": self.config.patience,
            "min_delta": self.config.min_delta,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "plateau_counter": self.plateau_counter,
        }

    @staticmethod
    def load_config(config_data: dict) -> Dict[str, Any]:
        """Load configuration from dict data."""
        return {
            "enabled": config_data.get("enabled", True),
            "patience": config_data.get("patience", 10),
            "min_delta": config_data.get("min_delta", 0.001),
            "monitor": config_data.get("monitor", "validation_loss"),
            "mode": config_data.get("mode", "min"),
            "strategy": config_data.get("strategy", "patience"),
            "restore_best_weights": config_data.get("restore_best_weights", True),
            "verbose": config_data.get("verbose", True),
            "baseline": config_data.get("baseline"),
            "min_epochs": config_data.get("min_epochs", 0),
            "plateau_factor": config_data.get("plateau_factor", 0.1),
            "plateau_patience": config_data.get("plateau_patience", 10),
            "plateau_threshold": config_data.get("plateau_threshold", 0.0001),
            "custom_conditions": config_data.get("custom_conditions", {}),
        }

    @staticmethod
    def create(config_data: dict, training_config) -> "EarlyStoppingPlugin":
        """Create plugin instance from config data."""
        config = EarlyStoppingPlugin.load_config(config_data)
        if not config.get("enabled", True):
            return None
        early_stopping_config = EarlyStoppingConfig(**config)
        return EarlyStoppingPlugin(early_stopping_config)
