"""
Plugin system for bz CLI.
Provides extensible hooks into the training lifecycle.
"""

from typing import List
from .plugin import Plugin, PluginContext, PluginError, PluginRegistry
from .plugin import get_plugin_registry, register_plugin, create_plugin, list_plugins
from .console_out import ConsoleOutPlugin
from .tensorboard import TensorBoardPlugin
from .wandb import WandBPlugin
from .profiler import ProfilerPlugin
from .optuna import OptunaPlugin, OptunaConfig
from .early_stopping import EarlyStoppingPlugin, EarlyStoppingConfig

# Register built-in plugins
_plugin_registry = get_plugin_registry()

# Register console output plugin
register_plugin("console_out", ConsoleOutPlugin, {"update_interval": 1})

# Register TensorBoard plugin
register_plugin("tensorboard", TensorBoardPlugin, {"log_dir": "runs/experiment"})

# Register WandB plugin
register_plugin("wandb", WandBPlugin, {"project_name": "bz-experiments", "entity": None})

# Register Profiler plugin
register_plugin("profiler", ProfilerPlugin, {"log_interval": 10, "enable_gpu_monitoring": True})

# Register Optuna plugin
register_plugin(
    "optuna",
    OptunaPlugin,
    {
        "study_name": "bz_optimization",
        "n_trials": 10,
        "direction": "minimize",
        "hyperparameters": {
            "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-1},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
        },
    },
)

# Register Early Stopping plugin
register_plugin(
    "early_stopping",
    EarlyStoppingPlugin,
    {
        "enabled": True,
        "patience": 10,
        "min_delta": 0.001,
        "monitor": "validation_loss",
        "mode": "min",
        "strategy": "patience",
    },
)


def default_plugins(spec):
    """Get default plugins for a training specification."""
    return [ConsoleOutPlugin.init(spec)]


def load_plugins_from_config(plugin_configs: dict, spec) -> List[Plugin]:
    """
    Load plugins based on configuration.

    Args:
        plugin_configs: Dictionary of plugin configurations
        spec: Training specification

    Returns:
        List of configured plugin instances
    """
    plugins: List[Plugin] = []
    """
    Load plugins based on configuration.
    
    Args:
        plugin_configs: Dictionary of plugin configurations
        spec: Training specification
        
    Returns:
        List of configured plugin instances
    """
    for plugin_name, config in plugin_configs.items():
        if not config.get("enabled", True):
            continue

        plugin_config = config.get("config", {})

        if plugin_name == "console_out":
            console_plugin: ConsoleOutPlugin = ConsoleOutPlugin.init(spec)
            if plugin_config:
                console_plugin.update_interval = plugin_config.get("update_interval", 1)
            plugins.append(console_plugin)

        elif plugin_name == "tensorboard":
            log_dir = plugin_config.get("log_dir", "runs/experiment")
            tensorboard_plugin: TensorBoardPlugin = TensorBoardPlugin.init(spec, log_dir)
            plugins.append(tensorboard_plugin)

        elif plugin_name == "wandb":
            project_name = plugin_config.get("project_name", "bz-experiments")
            entity = plugin_config.get("entity")
            wandb_plugin: WandBPlugin = WandBPlugin.init(spec, project_name, entity)
            plugins.append(wandb_plugin)

        elif plugin_name == "profiler":
            log_interval = plugin_config.get("log_interval", 10)
            enable_gpu_monitoring = plugin_config.get("enable_gpu_monitoring", True)
            profiler_plugin: ProfilerPlugin = ProfilerPlugin.init(spec, log_interval, enable_gpu_monitoring)
            plugins.append(profiler_plugin)

        elif plugin_name == "optuna":
            from .optuna import OptunaConfig

            optuna_config = OptunaConfig(**plugin_config)
            optuna_plugin: OptunaPlugin = OptunaPlugin(optuna_config)
            plugins.append(optuna_plugin)

        elif plugin_name == "early_stopping":
            from .early_stopping import EarlyStoppingConfig

            early_stopping_config = EarlyStoppingConfig(**plugin_config)
            early_stopping_plugin: EarlyStoppingPlugin = EarlyStoppingPlugin(early_stopping_config)
            plugins.append(early_stopping_plugin)

        else:
            # Try to create plugin from registry
            plugin = create_plugin(plugin_name, plugin_config)
            if plugin is not None:
                plugins.append(plugin)
            else:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Unknown plugin: {plugin_name}")

    return plugins

    return plugins


__all__ = [
    "Plugin",
    "PluginContext",
    "PluginError",
    "PluginRegistry",
    "get_plugin_registry",
    "register_plugin",
    "create_plugin",
    "list_plugins",
    "ConsoleOutPlugin",
    "TensorBoardPlugin",
    "WandBPlugin",
    "ProfilerPlugin",
    "OptunaPlugin",
    "OptunaConfig",
    "EarlyStoppingPlugin",
    "EarlyStoppingConfig",
    "default_plugins",
    "load_plugins_from_config",
]
