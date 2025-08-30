"""
Plugin system for bz CLI.
Provides extensible hooks into the training lifecycle.
"""

import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PluginContext:
    """Context passed to plugins during training lifecycle."""

    epoch: int = 0
    training_loss_total: float = 0.0
    validation_loss_total: float = 0.0
    training_batch_count: int = 0
    validation_batch_count: int = 0
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class PluginError(Exception):
    """Base exception for plugin-related errors."""

    pass


class Plugin(ABC):
    """
    Abstract base class for all plugins.

    Plugins can hook into various stages of the training lifecycle
    to perform additional actions like logging, visualization, etc.
    """

    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize plugin.

        Args:
            name: Optional custom name for the plugin
            config: Plugin-specific configuration
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.logger = logging.getLogger(f"bz.plugin.{self.name}")

    def start_training_session(self, context: PluginContext) -> None:
        """Called at the start of a training session."""
        pass

    def load_checkpoint(self, context: PluginContext) -> None:
        """Called when loading a checkpoint."""
        pass

    def start_epoch(self, context: PluginContext) -> None:
        """Called at the start of each epoch."""
        pass

    def start_training_loop(self, context: PluginContext) -> None:
        """Called at the start of the training loop."""
        pass

    def start_training_batch(self, context: PluginContext) -> None:
        """Called at the start of each training batch."""
        pass

    def end_training_batch(self, context: PluginContext) -> None:
        """Called at the end of each training batch."""
        pass

    def end_training_loop(self, context: PluginContext) -> None:
        """Called at the end of the training loop."""
        pass

    def start_validation_loop(self, context: PluginContext) -> None:
        """Called at the start of the validation loop."""
        pass

    def start_validation_batch(self, context: PluginContext) -> None:
        """Called at the start of each validation batch."""
        pass

    def end_validation_batch(self, context: PluginContext) -> None:
        """Called at the end of each validation batch."""
        pass

    def end_validation_loop(self, context: PluginContext) -> None:
        """Called at the end of the validation loop."""
        pass

    def save_checkpoint(self, context: PluginContext) -> None:
        """Called when saving a checkpoint."""
        pass

    def end_epoch(self, context: PluginContext) -> None:
        """Called at the end of each epoch."""
        pass

    def end_training_session(self, context: PluginContext) -> None:
        """Called at the end of a training session."""
        pass

    def run_stage(self, stage: str, context: PluginContext) -> None:
        """
        Run a specific training stage.

        Args:
            stage: Name of the stage to run
            context: Training context
        """
        method = getattr(self, stage, None)
        if method is not None:
            method(context)


class PluginRegistry:
    """Registry for managing plugin discovery and loading."""

    def __init__(self):
        self._plugins: Dict[str, Type[Plugin]] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, plugin_class: Type[Plugin], config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a plugin class.

        Args:
            name: Plugin name
            plugin_class: Plugin class to register
            config: Default configuration for the plugin
        """
        if not issubclass(plugin_class, Plugin):
            raise ValueError(f"Plugin class must inherit from Plugin: {plugin_class}")

        self._plugins[name] = plugin_class
        self._plugin_configs[name] = config or {}
        logger.info(f"Registered plugin: {name}")

    def get_plugin_class(self, name: str) -> Optional[Type[Plugin]]:
        """Get a plugin class by name."""
        return self._plugins.get(name)

    def create_plugin(self, name: str, config: Optional[Dict[str, Any]] = None) -> Optional[Plugin]:
        """
        Create a plugin instance.

        Args:
            name: Plugin name
            config: Plugin configuration

        Returns:
            Plugin instance or None if not found
        """
        plugin_class = self.get_plugin_class(name)
        if plugin_class is None:
            return None

        # Merge with default config
        default_config = self._plugin_configs.get(name, {})
        if config:
            merged_config = {**default_config, **config}
        else:
            merged_config = default_config

        return plugin_class(config=merged_config)

    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())

    def unregister(self, name: str) -> None:
        """Unregister a plugin."""
        if name in self._plugins:
            del self._plugins[name]
            del self._plugin_configs[name]
            logger.info(f"Unregistered plugin: {name}")


# Global plugin registry
_plugin_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _plugin_registry


def register_plugin(name: str, plugin_class: Type[Plugin], config: Optional[Dict[str, Any]] = None) -> None:
    """Register a plugin with the global registry."""
    _plugin_registry.register(name, plugin_class, config)


def create_plugin(name: str, config: Optional[Dict[str, Any]] = None) -> Optional[Plugin]:
    """Create a plugin instance from the global registry."""
    return _plugin_registry.create_plugin(name, config)


def list_plugins() -> List[str]:
    """List all registered plugins."""
    return _plugin_registry.list_plugins()
