"""
Plugin system for bz CLI.
Provides extensible hooks into the training lifecycle.
"""

import logging
import importlib.metadata
from typing import List, Type, Dict, Optional

from .plugin import Plugin, PluginContext, PluginError

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing plugin discovery and loading."""

    def __init__(self):
        self._plugins: Dict[str, Type[Plugin]] = {}
        self._discover_and_register()

    def _discover_and_register(self):
        """Discover and register all available plugins via entry points."""
        try:
            # Try the newer API first (Python 3.10+)
            entry_points = importlib.metadata.entry_points()
            if hasattr(entry_points, "select"):
                bz_plugins = entry_points.select(group="bz.plugins")
            else:
                # Fallback for older Python versions
                bz_plugins = entry_points.get("bz.plugins", [])
        except Exception:
            # Fallback for very old Python versions
            bz_plugins = importlib.metadata.entry_points().get("bz.plugins", [])

        # If no plugins found via entry points, register built-in plugins directly
        if not bz_plugins:
            logger.info("No plugins found via entry points, registering built-in plugins directly")
            self._register_builtin_plugins()
            return

        for entry_point in bz_plugins:
            try:
                plugin_class = entry_point.load()
                if hasattr(plugin_class, "name"):
                    self._plugins[plugin_class.name] = plugin_class
                    logger.info(f"Registered plugin: {plugin_class.name}")
                else:
                    logger.warning(f"Plugin {entry_point.name} missing 'name' attribute")
            except Exception as e:
                logger.warning(f"Failed to load plugin {entry_point.name}: {e}")

    def _register_builtin_plugins(self):
        """Register built-in plugins directly."""
        try:
            from .console_out import ConsoleOutPlugin
            from .early_stopping import EarlyStoppingPlugin

            self._plugins["console_out"] = ConsoleOutPlugin
            self._plugins["early_stopping"] = EarlyStoppingPlugin

            logger.info("Registered built-in plugins: console_out, early_stopping")
        except ImportError as e:
            logger.warning(f"Failed to import built-in plugins: {e}")

    def create_plugin(self, name: str, config_data: dict, training_config) -> Optional[Plugin]:
        """
        Create a plugin instance.

        Args:
            name: Plugin name
            config_data: Plugin configuration data
            training_config: Training configuration

        Returns:
            Plugin instance or None if not found
        """
        plugin_class = self._plugins.get(name)
        if plugin_class and hasattr(plugin_class, "create"):
            try:
                return plugin_class.create(config_data, training_config)
            except Exception as e:
                logger.error(f"Failed to create plugin {name}: {e}")
                return None
        elif plugin_class:
            logger.warning(f"Plugin {name} missing 'create' method")
            return None
        else:
            logger.warning(f"Unknown plugin: {name}")
            return None

    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())

    def get_plugin_class(self, name: str) -> Optional[Type[Plugin]]:
        """Get a plugin class by name."""
        return self._plugins.get(name)


# Global registry instance
_plugin_registry = None


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry instance."""
    global _plugin_registry
    if _plugin_registry is None:
        _plugin_registry = PluginRegistry()
    return _plugin_registry


def list_plugins() -> List[str]:
    """List all available plugins."""
    return get_plugin_registry().list_plugins()


def create_plugin(name: str, config_data: Optional[dict] = None, training_config=None) -> Optional[Plugin]:
    """
    Create a plugin instance.

    Args:
        name: Plugin name
        config_data: Plugin configuration data
        training_config: Training configuration

    Returns:
        Plugin instance or None if not found
    """
    if config_data is None:
        config_data = {}
    return get_plugin_registry().create_plugin(name, config_data, training_config)


__all__ = [
    "Plugin",
    "PluginContext",
    "PluginError",
    "PluginRegistry",
    "get_plugin_registry",
    "list_plugins",
    "create_plugin",
]
