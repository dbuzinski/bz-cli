"""
Configuration management for bz CLI.
Provides a unified, type-safe configuration system with validation.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
import logging

logger = logging.getLogger(__name__)


class PluginConfig(TypedDict, total=False):
    """Configuration for a plugin."""

    enabled: bool
    config: Dict[str, Any]
    dependencies: List[str]  # Plugin dependencies


class TrainingConfig(TypedDict, total=False):
    """Training-specific configuration."""

    epochs: int
    batch_size: int
    learning_rate: float
    device: str
    compile: bool
    checkpoint_interval: int
    early_stopping_patience: Optional[int]
    early_stopping_min_delta: float


class MetricsConfig(TypedDict, total=False):
    """Metrics configuration."""

    metrics: List[str]
    custom_metrics: Dict[str, Dict[str, Any]]


class BzConfig(TypedDict, total=False):
    """Main configuration schema."""

    training: TrainingConfig
    plugins: Dict[str, PluginConfig]
    metrics: MetricsConfig
    hyperparameters: Dict[str, Any]
    optuna: Optional[Dict[str, Any]]
    environment: Optional[str]  # Environment name (dev, staging, prod)


@dataclass
class ConfigManager:
    """Manages configuration loading, validation, and access."""

    config_path: Optional[str] = None
    environment: Optional[str] = None
    _config: Optional[BzConfig] = None

    def __post_init__(self):
        if self.config_path is None:
            self.config_path = self._find_config_file()
        if self.environment is None:
            self.environment = self._get_environment()

    def _get_environment(self) -> str:
        """Get the current environment from environment variable."""
        return os.environ.get("BZ_ENV", "development")

    def _find_config_file(self) -> str:
        """Find the configuration file to use."""
        # Check environment variable first
        env_path = os.environ.get("BZ_CONFIG")
        if env_path and os.path.isfile(env_path):
            return env_path

        # Check environment-specific config files
        env = self._get_environment()
        env_config_names = [
            f"bz_config.{env}.json",
            f"config.{env}.json",
            f"bz.{env}.json",
            f"bz.{env}.yaml",
            f"bz.{env}.yml",
        ]

        for name in env_config_names:
            if os.path.isfile(name):
                return name

        # Check common config file names
        config_names = ["bz_config.json", "config.json", "bz.yaml", "bz.yml"]
        for name in config_names:
            if os.path.isfile(name):
                return name

        return "bz_config.json"

    def load(self) -> BzConfig:
        """Load and validate configuration."""
        if self._config is not None:
            return self._config

        if self.config_path is None:
            self._config = self._get_default_config()
            return self._config

        try:
            with open(self.config_path, "r") as f:
                raw_config = json.load(f)

            # Validate and merge with defaults
            self._config = self._validate_and_merge(raw_config)
            return self._config

        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            self._config = self._get_default_config()
            return self._config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {self.config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config from {self.config_path}: {e}")

    def _get_default_config(self) -> BzConfig:
        """Get default configuration."""
        return {
            "training": {
                "epochs": 1,
                "batch_size": 32,
                "learning_rate": 0.001,
                "device": "auto",
                "compile": True,
                "checkpoint_interval": 0,
                "early_stopping_patience": None,
                "early_stopping_min_delta": 0.001,
            },
            "plugins": {
                "console_out": {"enabled": True, "config": {}, "dependencies": []},
                "tensorboard": {"enabled": False, "config": {"log_dir": "runs/experiment"}, "dependencies": []},
            },
            "metrics": {
                "metrics": ["accuracy"],
                "custom_metrics": {},
            },
            "hyperparameters": {},
            "optuna": None,
            "environment": self.environment,
        }

    def _validate_and_merge(self, raw_config: Dict[str, Any]) -> BzConfig:
        """Validate and merge raw config with defaults."""
        default_config = self._get_default_config()

        # Deep merge configuration
        merged_config = self._deep_merge(default_config, raw_config)  # type: ignore

        # Validate configuration
        self._validate_config(merged_config)  # type: ignore

        return merged_config  # type: ignore

    def _deep_merge(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = default.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_config(self, config: BzConfig) -> None:
        """Validate configuration values."""
        training = config.get("training", {})

        # Validate training config
        if "epochs" in training and training["epochs"] < 1:
            raise ValueError("epochs must be at least 1")

        if "batch_size" in training and training["batch_size"] < 1:
            raise ValueError("batch_size must be at least 1")

        if "learning_rate" in training and training["learning_rate"] <= 0:
            raise ValueError("learning_rate must be positive")

        # Validate device
        if "device" in training:
            device = training["device"]
            if device not in ["auto", "cpu", "cuda"]:
                raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

        # Validate plugin dependencies
        self._validate_plugin_dependencies(config.get("plugins", {}))

    def _validate_plugin_dependencies(self, plugins: Dict[str, PluginConfig]) -> None:
        """Validate plugin dependencies."""
        plugin_names = set(plugins.keys())

        for plugin_name, config in plugins.items():
            dependencies = config.get("dependencies", [])
            for dep in dependencies:
                if dep not in plugin_names:
                    raise ValueError(f"Plugin '{plugin_name}' depends on '{dep}' which is not configured")

    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        config = self.load()
        return config.get("training", {})

    def get_plugin_config(self, plugin_name: str) -> Optional[PluginConfig]:
        """Get configuration for a specific plugin."""
        config = self.load()
        plugins = config.get("plugins", {})
        return plugins.get(plugin_name)

    def get_metrics_config(self) -> MetricsConfig:
        """Get metrics configuration."""
        config = self.load()
        return config.get("metrics", {})

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters."""
        config = self.load()
        return config.get("hyperparameters", {})

    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled."""
        plugin_config = self.get_plugin_config(plugin_name)
        return plugin_config is not None and plugin_config.get("enabled", True)

    def get_enabled_plugins(self) -> List[str]:
        """Get list of enabled plugins in dependency order."""
        config = self.load()
        plugins = config.get("plugins", {})

        # Filter enabled plugins
        enabled_plugins = {name: config for name, config in plugins.items() if config.get("enabled", True)}

        # Sort by dependencies (topological sort)
        return self._sort_plugins_by_dependencies(enabled_plugins)

    def _sort_plugins_by_dependencies(self, plugins: Dict[str, PluginConfig]) -> List[str]:
        """Sort plugins by their dependencies using topological sort."""
        # Build dependency graph
        graph = {name: set(config.get("dependencies", [])) for name, config in plugins.items()}

        # Topological sort
        result = []
        visited = set()
        temp_visited = set()

        def visit(node: str) -> None:
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected involving plugin '{node}'")
            if node in visited:
                return

            temp_visited.add(node)

            for dep in graph.get(node, set()):
                if dep in plugins:  # Only visit dependencies that are enabled
                    visit(dep)

            temp_visited.remove(node)
            visited.add(node)
            result.append(node)

        for plugin_name in plugins:
            if plugin_name not in visited:
                visit(plugin_name)

        return result

    def get_environment(self) -> str:
        """Get the current environment."""
        config = self.load()
        return config.get("environment", "development") or "development"

    def save_config(self, config: BzConfig, path: Optional[str] = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = self.config_path or "bz_config.json"

        try:
            with open(path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            raise ValueError(f"Failed to save configuration to {path}: {e}")


# Backward compatibility function
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration file (backward compatibility).

    Args:
        path: Optional path to config file

    Returns:
        Dictionary containing configuration
    """
    # Load file if provided a path
    if path:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    # If BZ_CONFIG environment variable is set, load
    # the file it points to
    env_path = os.environ.get("BZ_CONFIG")
    if env_path and os.path.isfile(env_path):
        try:
            with open(env_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    # Otherwise default to config.json in current folder
    default_path = "config.json"
    if os.path.isfile(default_path):
        try:
            with open(default_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    return {}


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
