"""
Main bz package for machine learning model training.
Provides extensible training with plugins and metrics.
"""

import hashlib
import inspect
import json
import logging
import os
from typing import Dict, Any, List, Optional

import torch

from .metrics import Metric
from .plugins import Plugin, PluginContext

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default device for training
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CheckpointManager:
    """
    Manages model checkpointing and resuming.

    Handles saving and loading model checkpoints including model state,
    optimizer state, loss function state, and training data loader state.
    """

    def __init__(self, checkpoint_dir: str):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch: int, model, optimizer, loss_fn, training_loader, device) -> str:
        """Save a checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch{epoch}.pth")
        checkpoint_data = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss_fn_state": loss_fn.state_dict(),
            "epoch": epoch,
        }

        # Save generator state if available
        if hasattr(training_loader, "generator") and training_loader.generator is not None:
            checkpoint_data["generator_state"] = training_loader.generator.get_state()

        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path

    def load_latest_checkpoint(self, model, optimizer, loss_fn, training_loader, device) -> Optional[int]:
        """Load the latest checkpoint and return the epoch number."""
        latest_epoch = self._find_latest_checkpoint_epoch()
        if latest_epoch is None:
            return None

        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch{latest_epoch}.pth")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "loss_fn_state" in checkpoint:
                loss_fn.load_state_dict(checkpoint["loss_fn_state"])
            if "generator_state" in checkpoint and hasattr(training_loader, "generator"):
                training_loader.generator.set_state(checkpoint["generator_state"])
            return latest_epoch
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def _find_latest_checkpoint_epoch(self) -> Optional[int]:
        """Find the latest checkpoint epoch number."""
        if not os.path.exists(self.checkpoint_dir):
            return None
        files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("model_epoch") and f.endswith(".pth")]
        epochs = []
        for f in files:
            try:
                epoch_num = int(f.replace("model_epoch", "").replace(".pth", ""))
                epochs.append(epoch_num)
            except ValueError:
                continue
        return max(epochs) if epochs else None


class TrainingLoop:
    """
    Handles the core training loop logic.

    Manages the execution of training and validation epochs,
    including metric updates and plugin lifecycle hooks.
    """

    def __init__(self, trainer: "Trainer"):
        """
        Initialize training loop.

        Args:
            trainer: Reference to the parent trainer instance
        """
        self.trainer = trainer

    def run_training_epoch(
        self, context: PluginContext, model, optimizer, loss_fn, training_loader, device, metrics: List[Metric]
    ) -> None:
        """Run a single training epoch."""
        self.trainer._run_stage("start_training_loop", context)

        # Reset metrics
        for metric in metrics:
            metric.reset()
            context.training_metrics[metric.name] = 0.0
        context.training_loss_total = 0.0
        context.training_batch_count = 0

        model.train()

        for batch_data, batch_labels in training_loader:
            self.trainer._run_stage("start_training_batch", context)

            try:
                # Training step
                optimizer.zero_grad(set_to_none=True)
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                preds = model(batch_data)
                loss = loss_fn(preds, batch_labels)
                loss.backward()
                optimizer.step()

                # Update metrics
                with torch.no_grad():
                    for metric in metrics:
                        metric.update(preds.detach().cpu(), batch_labels.detach().cpu())
                        context.training_metrics[metric.name] = metric.compute()
                    context.training_loss_total += loss.item()
                    context.training_batch_count += 1

            except Exception as e:
                self.trainer.logger.error(f"Error in training batch: {e}")
                continue

            self.trainer._run_stage("end_training_batch", context)

        self.trainer._run_stage("end_training_loop", context)

    def run_validation_epoch(
        self, context: PluginContext, model, loss_fn, validation_loader, device, metrics: List[Metric]
    ) -> None:
        """Run a single validation epoch."""
        if validation_loader is None:
            return

        self.trainer._run_stage("start_validation_loop", context)

        # Reset metrics
        for metric in metrics:
            metric.reset()
            context.validation_metrics[metric.name] = 0.0
        context.validation_loss_total = 0.0
        context.validation_batch_count = 0

        model.eval()
        with torch.no_grad():
            for batch_inputs, batch_labels in validation_loader:
                self.trainer._run_stage("start_validation_batch", context)

                try:
                    batch_inputs = batch_inputs.to(device)
                    batch_labels = batch_labels.to(device)
                    preds = model(batch_inputs)
                    loss = loss_fn(preds, batch_labels)

                    # Update metrics
                    for metric in metrics:
                        metric.update(preds.detach().cpu(), batch_labels.detach().cpu())
                        context.validation_metrics[metric.name] = metric.compute()
                    context.validation_loss_total += loss.item()
                    context.validation_batch_count += 1

                except Exception as e:
                    self.trainer.logger.error(f"Error in validation batch: {e}")
                    continue

                self.trainer._run_stage("end_validation_batch", context)

        self.trainer._run_stage("end_validation_loop", context)


class Trainer:
    """
    Main trainer class for machine learning model training.

    Orchestrates the complete training process including:
    - Plugin lifecycle management
    - Checkpointing and resumption
    - Training and validation loops
    - Metric tracking
    - Early stopping
    """

    def __init__(self):
        """Initialize trainer with empty plugin list and training loop."""
        self.plugins: List[Plugin] = []
        self.logger = logger
        self.training_loop = TrainingLoop(self)

    def add_plugin(self, plugin: Plugin) -> None:
        """Add a plugin to the trainer."""
        self.plugins.append(plugin)

    def train(
        self,
        model,
        optimizer,
        loss_fn,
        training_loader,
        validation_loader=None,
        device=default_device,
        epochs=1,
        compile=True,
        checkpoint_interval=0,
        metrics: Optional[List[Metric]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        early_stopping_patience=None,
        early_stopping_min_delta=0.001,
    ):
        """
        Train a model with the specified configuration.

        Args:
            model: PyTorch model to train
            optimizer: PyTorch optimizer
            loss_fn: Loss function
            training_loader: Training data loader
            validation_loader: Validation data loader (optional)
            device: Device to train on
            epochs: Number of training epochs
            compile: Whether to compile the model
            checkpoint_interval: How often to save checkpoints
            metrics: List of metrics to track
            hyperparameters: Dictionary of hyperparameters
            early_stopping_patience: Early stopping patience
            early_stopping_min_delta: Minimum improvement for early stopping
        """
        if metrics is None:
            metrics = []
        if hyperparameters is None:
            hyperparameters = {}

        # Initialize context
        context = PluginContext()
        context.hyperparameters = hyperparameters

        # Compute training signature and setup checkpoint manager
        training_signature = self._compute_training_signature(model, optimizer, loss_fn, hyperparameters)
        checkpoint_dir = os.path.join(".bz", "checkpoints", training_signature)
        checkpoint_manager = CheckpointManager(checkpoint_dir)

        self._run_stage("start_training_session", context)

        # Try to resume from checkpoint
        latest_epoch = checkpoint_manager.load_latest_checkpoint(model, optimizer, loss_fn, training_loader, device)
        if latest_epoch is not None:
            context.epoch = latest_epoch
            context.extra["start_epoch"] = latest_epoch
            self._run_stage("load_checkpoint", context)
            self.logger.info(f"Resumed training from epoch {latest_epoch}")

        # Compile model if requested
        if compile:
            try:
                model.compile()
                self.logger.info("Model compiled successfully")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")

        # Move model to device
        model.to(device)

        # Training loop
        for epoch in range(context.epoch, epochs):
            self._run_stage("start_epoch", context)

            # Training epoch
            self.training_loop.run_training_epoch(context, model, optimizer, loss_fn, training_loader, device, metrics)

            # Validation epoch
            self.training_loop.run_validation_epoch(context, model, loss_fn, validation_loader, device, metrics)

            # Early stopping is now handled by the EarlyStoppingPlugin
            # Check if any plugin has requested to stop training
            if hasattr(context, "should_stop_training") and context.should_stop_training:
                self.logger.info("Training stopped by plugin")
                break

            # Save checkpoint
            if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    epoch + 1, model, optimizer, loss_fn, training_loader, device
                )
                context.extra["checkpoint_path"] = checkpoint_path
                self._run_stage("save_checkpoint", context)

            self._run_stage("end_epoch", context)
            context.epoch += 1

        self._run_stage("end_training_session", context)

    def _run_stage(self, stage_name: str, context: PluginContext) -> None:
        """Run a training stage across all plugins with error handling."""
        for plugin in self.plugins:
            try:
                plugin.run_stage(stage_name, context)
            except Exception as e:
                self.logger.error(f"Plugin {plugin.name} failed in stage {stage_name}: {e}")
                # Continue with other plugins instead of failing completely

    def _compute_training_signature(self, model, optimizer, loss_fn, config: Dict[str, Any]) -> str:
        """Compute a unique signature for the training configuration."""
        payload = config.copy()
        payload["__model"] = type(model).__name__
        payload["__optimizer"] = type(optimizer).__name__
        payload["__optimizer_params"] = optimizer.param_groups
        payload["__loss_fn"] = type(loss_fn).__name__
        payload["__loss_fn_params"] = loss_fn.__dict__ or inspect.signature(loss_fn)
        serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration file (backward compatibility).

    Args:
        path: Optional path to configuration file

    Returns:
        Configuration dictionary

    Note:
        This function searches for configuration in the following order:
        1. Provided path (if specified)
        2. BZ_CONFIG environment variable
        3. config.json in current directory
        4. Empty dictionary if no config found
    """
    # Load file if provided a path
    if path:
        with open(path, "r") as f:
            return json.load(f)

    # If BZ_CONFIG environment variable is set, load the file it points to
    env_path = os.environ.get("BZ_CONFIG")
    if env_path and os.path.isfile(env_path):
        with open(env_path, "r") as f:
            return json.load(f)

    # Otherwise default to config.json in current folder
    default_path = "config.json"
    if os.path.isfile(default_path):
        with open(default_path, "r") as f:
            return json.load(f)

    return {}


__all__ = [
    "Trainer",
    "TrainingLoop",
    "CheckpointManager",
    "load_config",
    "default_device",
    "logger",
]
