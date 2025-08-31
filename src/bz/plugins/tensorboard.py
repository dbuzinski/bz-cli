"""
TensorBoard plugin for bz CLI.
Provides TensorBoard integration for training visualization.
"""

from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter

from .plugin import Plugin, PluginContext


class TensorBoardPlugin(Plugin):
    """Plugin for TensorBoard logging."""

    name = "tensorboard"  # Plugin name for discovery

    def __init__(self, training_loader_len: int, log_dir: str, **kwargs):
        """
        Initialize TensorBoard plugin.

        Args:
            training_loader_len: Number of training batches
            log_dir: Directory for TensorBoard logs
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(**kwargs)
        self.training_loader_len = training_loader_len
        self.writer: Optional[SummaryWriter] = None
        self.log_dir = log_dir

    def start_training_session(self, context: PluginContext) -> None:
        """Initialize TensorBoard writer."""
        try:
            self.writer = SummaryWriter(self.log_dir)
            self.logger.info(f"TensorBoard logging to {self.log_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorBoard writer: {e}")
            self.writer = None

    def end_training_batch(self, context: PluginContext) -> None:
        """Log training batch metrics to TensorBoard."""
        if self.writer is None or context.training_batch_count == 0:
            return

        try:
            avg_loss = context.training_loss_total / context.training_batch_count
            step = context.epoch * self.training_loader_len + context.training_batch_count
            self.writer.add_scalar("Loss/Train Step", avg_loss, step)
        except Exception as e:
            self.logger.error(f"Failed to log training batch metrics: {e}")

    def end_training_loop(self, context: PluginContext) -> None:
        """Log training epoch metrics to TensorBoard."""
        if self.writer is None or context.training_batch_count == 0:
            return

        try:
            avg_loss = context.training_loss_total / context.training_batch_count
            self.writer.add_scalar("Loss/Train Epoch", avg_loss, context.epoch)
            for name, value in context.training_metrics.items():
                self.writer.add_scalar(f"Metric/Train/{name} Epoch", value, context.epoch)
        except Exception as e:
            self.logger.error(f"Failed to log training epoch metrics: {e}")

    def end_validation_loop(self, context: PluginContext) -> None:
        """Log validation epoch metrics to TensorBoard."""
        if self.writer is None or context.validation_batch_count == 0:
            return

        try:
            avg_loss = context.validation_loss_total / context.validation_batch_count
            self.writer.add_scalar("Loss/Validation Epoch", avg_loss, context.epoch)
            for name, value in context.validation_metrics.items():
                self.writer.add_scalar(f"Metric/Validation/{name} Epoch", value, context.epoch)
        except Exception as e:
            self.logger.error(f"Failed to log validation epoch metrics: {e}")

    def end_epoch(self, context: PluginContext) -> None:
        """Flush TensorBoard writer."""
        if self.writer is not None:
            try:
                self.writer.flush()
            except Exception as e:
                self.logger.error(f"Failed to flush TensorBoard writer: {e}")

    def end_training_session(self, context: PluginContext) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            try:
                self.writer.close()
                self.logger.info("TensorBoard writer closed")
            except Exception as e:
                self.logger.error(f"Failed to close TensorBoard writer: {e}")
            finally:
                self.writer = None

    @staticmethod
    def init(spec, log_dir: str = "runs/experiment") -> "TensorBoardPlugin":
        """
        Create a TensorBoardPlugin instance from training specification.

        Args:
            spec: Training specification containing data loaders
            log_dir: Directory for TensorBoard logs

        Returns:
            Configured TensorBoardPlugin instance
        """
        training_loader_len = len(spec.training_loader)
        return TensorBoardPlugin(training_loader_len, log_dir)

    @staticmethod
    def load_config(config_data: dict) -> Dict[str, Any]:
        """Load configuration from dict data."""
        return {"log_dir": config_data.get("log_dir", "runs/experiment"), "enabled": config_data.get("enabled", True)}

    @staticmethod
    def create(config_data: dict, training_config) -> "TensorBoardPlugin":
        """Create plugin instance from config data."""
        config = TensorBoardPlugin.load_config(config_data)
        if not config.get("enabled", True):
            return None
        return TensorBoardPlugin.init(training_config, config["log_dir"])
