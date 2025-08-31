"""
Configuration management for bz CLI.
Provides a unified, type-safe configuration system with validation.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
import logging

logger = logging.getLogger(__name__)

# Module-level variable to store CLI-specified config path
_cli_config_path: str = None


def set_cli_config_path(config_path: str) -> None:
    """Set the config path specified by CLI arguments."""
    global _cli_config_path
    _cli_config_path = config_path


@dataclass
class TrainingConfiguration:
    """Configuration for training a machine learning model."""

    epochs: int
    model: Any = None
    loss_fn: Any = None
    optimizer: Any = None
    training_loader: Any = None
    validation_loader: Any = None
    training_set: Any = None
    validation_set: Any = None
    device: str = "auto"
    compile: bool = True
    checkpoint_interval: int = 5
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: List[Any] = field(default_factory=list)
    plugins: List[Any] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")


def get_config() -> TrainingConfiguration:
    """
    Get configuration with proper precedence order:
    1. --config argument (handled by CLI)
    2. BZ_CONFIG environment variable
    3. bzconfig.json (default)

    Returns:
        TrainingConfiguration object with user modifications applied
    """
    # Determine config file path with precedence
    global _cli_config_path
    
    if _cli_config_path:
        config_path = _cli_config_path
    else:
        config_path = os.environ.get("BZ_CONFIG", "bzconfig.json")

    if not os.path.exists(config_path):
        if config_path == "bzconfig.json":
            raise FileNotFoundError(
                f"Configuration file {config_path} not found. Please create a bzconfig.json file or set the BZ_CONFIG environment variable."
            )
        else:
            raise FileNotFoundError(
                f"Configuration file specified by BZ_CONFIG environment variable not found: {config_path}"
            )

    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading configuration file {config_path}: {e}")

    # Create TrainingConfiguration from data
    config = TrainingConfiguration(
        epochs=config_data.get("epochs", 1),
        device=config_data.get("device", "auto"),
        compile=config_data.get("compile", True),
        checkpoint_interval=config_data.get("checkpoint_interval", 5),
        hyperparameters=config_data.get("hyperparameters", {}),
        metrics=config_data.get("metrics", []),
        plugins=config_data.get("plugins", []),
    )

    # Instantiate metrics (plugins need to wait until after data loaders are set up)
    config.metrics = _instantiate_metrics(config.metrics)

    return config


def _instantiate_plugins(plugin_configs: List[Union[str, Dict[str, Any]]], config) -> List[Any]:
    """Instantiate plugins from configuration."""
    from bz.plugins import load_plugins_from_config

    if plugin_configs is None:
        logger.warning("No plugins specified in configuration")
        return []

    try:
        return load_plugins_from_config(plugin_configs, config)
    except Exception as e:
        logger.error(f"Error instantiating plugins: {e}")
        return []


def _instantiate_metrics(metric_configs: List[Union[str, Dict[str, Any]]]) -> List[Any]:
    """Instantiate metrics from configuration."""
    from bz.metrics import get_metric

    if metric_configs is None:
        logger.warning("No metrics specified in configuration")
        return []

    metrics = []
    for metric_item in metric_configs:
        if isinstance(metric_item, str):
            try:
                metric = get_metric(metric_item)
                metrics.append(metric)
            except ValueError as e:
                logger.warning(f"Warning: {e}")
        elif isinstance(metric_item, dict):
            # Handle metric with configuration
            metric_name = list(metric_item.keys())[0]
            metric_config = metric_item[metric_name]
            try:
                metric = get_metric(metric_name)
                # Apply configuration if metric supports it
                if hasattr(metric, "configure"):
                    metric.configure(metric_config)
                metrics.append(metric)
            except ValueError as e:
                logger.warning(f"Warning: {e}")

    return metrics


def instantiate_plugins(config: TrainingConfiguration) -> None:
    """Instantiate plugins after data loaders are set up."""
    config.plugins = _instantiate_plugins(config.plugins, config)
