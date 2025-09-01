#!/usr/bin/env python3
"""
Demonstration of how a user can create a simple plugin for bz CLI.

This shows the minimal requirements for a plugin to be discovered via entry points.
"""

import logging
from typing import Dict, Any, Optional
from bz.plugins import Plugin, PluginContext


class SimpleUserPlugin(Plugin):
    """
    A simple example plugin that demonstrates the minimal requirements
    for a user-created plugin to work with bz CLI.
    """

    name = "simple_user_plugin"  # This name will be used in bzconfig.json

    def __init__(self, message: str = "Hello from user plugin!"):
        super().__init__()
        self.message = message
        self.logger = logging.getLogger(__name__)

    def start_training_session(self, context: PluginContext) -> None:
        """Called when training starts."""
        self.logger.info(f"🎉 {self.message}")
        self.logger.info(f"Training will run for {context.extra.get('epochs', 'unknown')} epochs")

    def end_epoch(self, context: PluginContext) -> None:
        """Called at the end of each epoch."""
        if context.epoch % 5 == 0:  # Log every 5 epochs
            self.logger.info(f"📊 Epoch {context.epoch} completed!")

    def end_training_session(self, context: PluginContext) -> None:
        """Called when training ends."""
        self.logger.info("🎯 Training completed! Great job!")

    @staticmethod
    def load_config(config_data: dict) -> Dict[str, Any]:
        """Load configuration from dict data."""
        return {
            "message": config_data.get("message", "Hello from user plugin!"),
            "enabled": config_data.get("enabled", True),
        }

    @staticmethod
    def create(config_data: dict, training_config) -> Optional["SimpleUserPlugin"]:
        """Create plugin instance from config data."""
        config = SimpleUserPlugin.load_config(config_data)
        if not config.get("enabled", True):
            return None
        return SimpleUserPlugin(message=config["message"])


# Example of how this would be set up in a user's package
"""
To make this plugin discoverable, the user would create a pyproject.toml file:

[project]
name = "my-simple-bz-plugin"
version = "0.1.0"
dependencies = ["bz-cli"]

[project.entry-points."bz.plugins"]
simple_user_plugin = "my_plugin:SimpleUserPlugin"

Then install it:
pip install -e .

And use it in bzconfig.json:
{
  "epochs": 10,
  "plugins": [
    "console_out",
    {
      "simple_user_plugin": {
        "enabled": true,
        "message": "Custom message from my plugin!"
      }
    }
  ]
}
"""


if __name__ == "__main__":
    # Demonstrate the plugin works
    print("Simple User Plugin Demo")
    print("=" * 40)

    # Test configuration loading
    config_data = {"enabled": True, "message": "Test message from config!"}

    config = SimpleUserPlugin.load_config(config_data)
    print(f"Loaded config: {config}")

    # Test plugin creation
    plugin = SimpleUserPlugin.create(config_data, None)
    print(f"Created plugin: {type(plugin).__name__}")
    print(f"Plugin message: {plugin.message}")

    print("\nThis plugin would be automatically discovered by bz CLI!")
    print("Just install it with the right entry points and it will work.")
