"""
Example of how a user can create a custom plugin for bz.

This demonstrates the complete process:
1. Creating a plugin class
2. Setting up entry points
3. Installing the plugin
4. Using it in configuration
"""

# Example custom plugin implementation
import logging
from typing import Dict, Any, Optional
from bz.plugins import Plugin, PluginContext


class CustomLoggerPlugin(Plugin):
    """
    Example custom plugin that logs training progress to a custom format.

    This plugin demonstrates how users can create their own plugins
    that integrate with the bz training system.
    """

    name = "custom_logger"  # This name will be used in bzconfig.json

    def __init__(self, log_file: str = "custom_training.log", log_level: str = "INFO"):
        super().__init__()
        self.log_file = log_file
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)

        # Set up custom logging
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, log_level.upper()))

    def start_training_session(self, context: PluginContext) -> None:
        """Log training session start."""
        self.logger.info(f"Training session started - Epochs: {context.extra.get('epochs', 'unknown')}")

    def end_training_batch(self, context: PluginContext) -> None:
        """Log training batch progress."""
        if context.training_batch_count % 10 == 0:  # Log every 10 batches
            loss = context.training_loss_total / context.training_batch_count if context.training_batch_count > 0 else 0
            self.logger.info(f"Epoch {context.epoch}, Batch {context.training_batch_count}, Loss: {loss:.4f}")

    def end_epoch(self, context: PluginContext) -> None:
        """Log epoch completion."""
        train_loss = (
            context.training_loss_total / context.training_batch_count if context.training_batch_count > 0 else 0
        )
        val_loss = (
            context.validation_loss_total / context.validation_batch_count if context.validation_batch_count > 0 else 0
        )
        self.logger.info(f"Epoch {context.epoch} completed - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    def end_training_session(self, context: PluginContext) -> None:
        """Log training session completion."""
        self.logger.info("Training session completed")

    @staticmethod
    def load_config(config_data: dict) -> Dict[str, Any]:
        """Load configuration from dict data."""
        return {
            "log_file": config_data.get("log_file", "custom_training.log"),
            "log_level": config_data.get("log_level", "INFO"),
            "enabled": config_data.get("enabled", True),
        }

    @staticmethod
    def create(config_data: dict, training_config) -> Optional["CustomLoggerPlugin"]:
        """Create plugin instance from config data."""
        config = CustomLoggerPlugin.load_config(config_data)
        if not config.get("enabled", True):
            return None
        return CustomLoggerPlugin(log_file=config["log_file"], log_level=config["log_level"])


# Example of how to set up entry points in a user's package
"""
To make this plugin discoverable, the user would need to:

1. Create a setup.py or pyproject.toml file in their package:

# setup.py example:
from setuptools import setup, find_packages

setup(
    name="my-bz-plugins",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "bz.plugins": [
            "custom_logger = my_plugins.custom_logger:CustomLoggerPlugin",
            "another_plugin = my_plugins.another:AnotherPlugin",
        ]
    },
    install_requires=["bz-cli"],
)

# Or pyproject.toml example:
[project]
name = "my-bz-plugins"
version = "0.1.0"
dependencies = ["bz-cli"]

[project.entry-points."bz.plugins"]
custom_logger = "my_plugins.custom_logger:CustomLoggerPlugin"
another_plugin = "my_plugins.another:AnotherPlugin"

2. Install the package:
   pip install -e .

3. Use the plugin in bzconfig.json:
{
  "epochs": 10,
  "plugins": [
    "console_out",
    {
      "custom_logger": {
        "enabled": true,
        "log_file": "my_training.log",
        "log_level": "DEBUG"
      }
    }
  ]
}
"""
