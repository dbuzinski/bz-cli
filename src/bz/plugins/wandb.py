"""
Weights & Biases plugin for bz CLI.
Provides WandB integration for experiment tracking and visualization.
"""

# type: ignore[import,import-not-found]
import os
from typing import Optional, Dict, Any
from bz.plugins import Plugin, PluginContext


class WandBPlugin(Plugin):
    """Plugin for Weights & Biases integration during training."""

    def __init__(
        self, project_name: str, entity: Optional[str] = None, config: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """
        Initialize WandB plugin.

        Args:
            project_name: WandB project name
            entity: WandB entity/username (optional)
            config: Plugin configuration
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(config=config, **kwargs)
        self.project_name = project_name
        self.entity = entity
        self.run = None
        self._wandb_available = self._check_wandb_available()

    def _check_wandb_available(self) -> bool:
        """Check if wandb is available."""
        try:
            import importlib.util

            return importlib.util.find_spec("wandb") is not None
        except ImportError:
            self.logger.warning("wandb not installed. Install with: pip install wandb")
            return False

    def start_training_session(self, context: PluginContext) -> None:
        """Initialize WandB run."""
        if not self._wandb_available:
            return

        try:
            import wandb # type: ignore

            # Initialize wandb run
            wandb_config = {
                "epochs": context.hyperparameters.get("epochs", 1),
                "batch_size": context.hyperparameters.get("batch_size", 32),
                "learning_rate": context.hyperparameters.get("learning_rate", 0.001),
                **context.hyperparameters,
            }

            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                config=wandb_config,
                name=f"run_{context.extra.get('training_signature', 'default')}",
            )

            if self.run is not None:
                self.logger.info(f"WandB run initialized: {self.run.name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize WandB run: {e}")
            self.run = None

    def end_training_batch(self, context: PluginContext) -> None:
        """Log training batch metrics to WandB."""
        if self.run is None or context.training_batch_count == 0:
            return

        try:
            import wandb

            # Calculate step number
            step = context.epoch * context.extra.get("training_batches_per_epoch", 1) + context.training_batch_count

            # Log training metrics
            log_dict = {"train/loss": context.training_loss_total / context.training_batch_count, "train/step": step}

            # Add training metrics
            for name, value in context.training_metrics.items():
                log_dict[f"train/{name}"] = value

            wandb.log(log_dict, step=step)

        except Exception as e:
            self.logger.error(f"Failed to log training batch metrics to WandB: {e}")

    def end_training_loop(self, context: PluginContext) -> None:
        """Log training epoch metrics to WandB."""
        if self.run is None or context.training_batch_count == 0:
            return

        try:
            import wandb

            # Log training epoch metrics
            log_dict = {
                "train/epoch_loss": context.training_loss_total / context.training_batch_count,
                "train/epoch": context.epoch,
            }

            # Add training metrics
            for name, value in context.training_metrics.items():
                log_dict[f"train/epoch_{name}"] = value

            wandb.log(log_dict, step=context.epoch)

        except Exception as e:
            self.logger.error(f"Failed to log training epoch metrics to WandB: {e}")

    def end_validation_loop(self, context: PluginContext) -> None:
        """Log validation epoch metrics to WandB."""
        if self.run is None or context.validation_batch_count == 0:
            return

        try:
            import wandb

            # Log validation epoch metrics
            log_dict = {
                "val/epoch_loss": context.validation_loss_total / context.validation_batch_count,
                "val/epoch": context.epoch,
            }

            # Add validation metrics
            for name, value in context.validation_metrics.items():
                log_dict[f"val/epoch_{name}"] = value

            wandb.log(log_dict, step=context.epoch)

        except Exception as e:
            self.logger.error(f"Failed to log validation epoch metrics to WandB: {e}")

    def save_checkpoint(self, context: PluginContext) -> None:
        """Save model checkpoint to WandB."""
        if self.run is None:
            return

        try:
            import wandb

            checkpoint_path = context.extra.get("checkpoint_path")
            if checkpoint_path and os.path.exists(checkpoint_path):
                # Save model file to WandB
                artifact = wandb.Artifact(
                    name=f"model-{self.run.name}",
                    type="model",
                    description=f"Model checkpoint from epoch {context.epoch}",
                )
                artifact.add_file(checkpoint_path)
                self.run.log_artifact(artifact)

                self.logger.info(f"Model checkpoint saved to WandB: {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to WandB: {e}")

    def end_training_session(self, context: PluginContext) -> None:
        """Finalize WandB run."""
        if self.run is None:
            return

        try:
            import wandb

            # Log final summary
            summary = {
                "final/train_loss": (
                    context.training_loss_total / context.training_batch_count
                    if context.training_batch_count > 0
                    else 0
                ),
                "final/val_loss": (
                    context.validation_loss_total / context.validation_batch_count
                    if context.validation_batch_count > 0
                    else 0
                ),
                "final/epochs_completed": context.epoch,
            }

            # Add final metrics
            for name, value in context.training_metrics.items():
                summary[f"final/train_{name}"] = value
            for name, value in context.validation_metrics.items():
                summary[f"final/val_{name}"] = value

            wandb.log(summary)

            # Finish the run
            wandb.finish()
            self.logger.info("WandB run finished")

        except Exception as e:
            self.logger.error(f"Failed to finalize WandB run: {e}")
        finally:
            self.run = None

    @staticmethod
    def init(spec, project_name: str, entity: Optional[str] = None) -> "WandBPlugin":
        """
        Create a WandBPlugin instance from training specification.

        Args:
            spec: Training specification containing data loaders
            project_name: WandB project name
            entity: WandB entity/username (optional)

        Returns:
            Configured WandBPlugin instance
        """
        return WandBPlugin(project_name=project_name, entity=entity)
