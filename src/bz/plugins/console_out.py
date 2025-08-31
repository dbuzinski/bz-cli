"""
Console output plugin for bz CLI.
Provides formatted console output during training.
"""

import time
from typing import Optional, Dict, Any
from tqdm import tqdm

from .plugin import Plugin, PluginContext


class ConsoleOutPlugin(Plugin):
    """Plugin for formatted console output during training."""

    name = "console_out"  # Plugin name for discovery

    def __init__(self, training_data_len: int, validation_data_len: int = 0, update_interval: int = 1, **kwargs):
        """
        Initialize console output plugin.

        Args:
            training_data_len: Number of training batches
            validation_data_len: Number of validation batches
            update_interval: How often to update progress bars
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(**kwargs)
        self.training_data_len = training_data_len
        self.validation_data_len = validation_data_len
        self.update_interval = update_interval
        self._training_bar: Optional[tqdm] = None
        self._validation_bar: Optional[tqdm] = None
        self.training_start_time: Optional[float] = None

    def start_training_session(self, context: PluginContext) -> None:
        """Record training start time."""
        self.training_start_time = time.time()

    def load_checkpoint(self, context: PluginContext) -> None:
        """Display checkpoint loading information."""
        print()
        start_epoch = context.extra.get("start_epoch")
        checkpoint = context.extra.get("checkpoint_path")
        print(f"✓ Epoch {start_epoch} loaded from {checkpoint}")

    def start_epoch(self, context: PluginContext) -> None:
        """Display epoch start information."""
        print()
        print(f"Epoch {context.epoch + 1}:")

    def start_training_loop(self, context: PluginContext) -> None:
        """Initialize training progress bar."""
        self._training_bar = tqdm(
            range(self.training_data_len),
            desc="Training",
            bar_format="{desc}:   {percentage:3.0f}%|{bar:40}{r_bar}",
            unit="batch",
        )

    def end_training_batch(self, context: PluginContext) -> None:
        """Update training progress bar."""
        if self._training_bar is not None:
            self._training_bar.update(1)
            if (self._training_bar.n + 1) % self.update_interval == 0:
                postfix_dict = {k: f"{v:.4f}" for k, v in context.training_metrics.items()}
                if context.training_batch_count > 0:
                    postfix_dict["loss"] = f"{context.training_loss_total / context.training_batch_count:.4f}"
                self._training_bar.set_postfix(postfix_dict)

    def end_training_loop(self, context: PluginContext) -> None:
        """Close training progress bar."""
        if self._training_bar is not None:
            self._training_bar.close()
            self._training_bar = None

    def start_validation_loop(self, context: PluginContext) -> None:
        """Initialize validation progress bar."""
        if self.validation_data_len:
            self._validation_bar = tqdm(
                range(self.validation_data_len),
                desc="Validation",
                bar_format="{desc}: {percentage:3.0f}%|{bar:40}{r_bar}",
                unit="batch",
            )

    def end_validation_batch(self, context: PluginContext) -> None:
        """Update validation progress bar."""
        if self._validation_bar is not None:
            self._validation_bar.update(1)
            if (self._validation_bar.n + 1) % self.update_interval == 0:
                postfix_dict = {k: f"{v:.4f}" for k, v in context.validation_metrics.items()}
                if context.validation_batch_count > 0:
                    postfix_dict["loss"] = f"{context.validation_loss_total / context.validation_batch_count:.4f}"
                self._validation_bar.set_postfix(postfix_dict)

    def end_validation_loop(self, context: PluginContext) -> None:
        """Close validation progress bar."""
        if self._validation_bar is not None:
            self._validation_bar.close()
            self._validation_bar = None

    def save_checkpoint(self, context: PluginContext) -> None:
        """Display checkpoint saving information."""
        checkpoint = context.extra.get("checkpoint_path")
        print(f"✓ Checkpoint saved to {checkpoint}")

    def end_training_session(self, context: PluginContext) -> None:  # type: ignore[assignment]
        """Display training completion summary."""
        if self.training_start_time is None:
            return

        # Epochs run formatting
        start_epoch = context.extra.get("start_epoch", 0)
        epochs_run = context.epoch - start_epoch
        resumed_string = f" (resumed from epoch {start_epoch})" if start_epoch else ""

        # Time formatting
        total_time = time.time() - self.training_start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"

        summary_item: Dict[str, str] = {
            "Epochs Run": f"{epochs_run}{resumed_string}",
            "Total Time": time_str,
        }
        if context.training_batch_count:
            summary_item["Training Loss"] = f"{context.training_loss_total/context.training_batch_count:.4f}"  # type: ignore[assignment]
            for name, val in context.training_metrics.items():
                summary_item[f"Training {name}"] = f"{val:.4f}"
        if context.validation_batch_count:
            summary_item["Validation Loss"] = f"{context.validation_loss_total/context.validation_batch_count:.4f}"
            for name, val in context.validation_metrics.items():
                summary_item[f"Validation {name}"] = f"{val:.4f}"

        ljust_len = max(len(label) for label in summary_item.keys()) + 1
        total_len = max(ljust_len + len(val) for val in summary_item.values()) + 4

        # Print header
        print("\n" + "=" * total_len)
        print((total_len - 18) // 2 * " " + "Training Complete")
        print("=" * total_len)
        print()

        # Print status
        for label, val in summary_item.items():  # type: ignore
            print(f" {label.ljust(ljust_len)}: {val}")

        # Print footer
        print("\n" + "=" * total_len)

    @staticmethod
    def init(spec, update_interval: int = 1) -> "ConsoleOutPlugin":
        """
        Create a ConsoleOutPlugin instance from training specification.

        Args:
            spec: Training specification containing data loaders
            update_interval: How often to update progress bars

        Returns:
            Configured ConsoleOutPlugin instance
        """
        validation_data_len = 0
        if spec.validation_loader:
            validation_data_len = len(spec.validation_loader)
        return ConsoleOutPlugin(
            len(spec.training_loader), validation_data_len=validation_data_len, update_interval=update_interval
        )

    @staticmethod
    def load_config(config_data: dict) -> Dict[str, Any]:
        """Load configuration from dict data."""
        return {"update_interval": config_data.get("update_interval", 1)}

    @staticmethod
    def create(config_data: dict, training_config) -> "ConsoleOutPlugin":
        """Create plugin instance from config data."""
        config = ConsoleOutPlugin.load_config(config_data)
        return ConsoleOutPlugin.init(training_config, update_interval=config["update_interval"])
