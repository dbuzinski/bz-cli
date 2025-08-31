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


def load_plugins_from_config(plugin_configs: list, config) -> List[Plugin]:
    """
    Load plugins based on configuration.

    Args:
        plugin_configs: List of plugin configurations (strings or objects)
        config: Training configuration

    Returns:
        List of configured plugin instances
    """
    plugins: List[Plugin] = []

    for plugin_item in plugin_configs:
        if isinstance(plugin_item, str):
            # Simple string plugin name
            plugin_name = plugin_item
            plugin_config = {}
        elif isinstance(plugin_item, dict):
            # Plugin with configuration
            plugin_name = list(plugin_item.keys())[0]
            plugin_config = plugin_item[plugin_name]
        else:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Invalid plugin configuration: {plugin_item}")
            continue

        # Check if plugin is enabled
        if isinstance(plugin_config, dict) and not plugin_config.get("enabled", True):
            continue

        # Create plugin based on name
        if plugin_name == "console_out":
            console_plugin: ConsoleOutPlugin = ConsoleOutPlugin.init(config)
            if plugin_config and isinstance(plugin_config, dict):
                console_plugin.update_interval = plugin_config.get("update_interval", 1)
            plugins.append(console_plugin)

        elif plugin_name == "tensorboard":
            log_dir = "runs/experiment"
            if isinstance(plugin_config, dict):
                log_dir = plugin_config.get("log_dir", log_dir)
            tensorboard_plugin: TensorBoardPlugin = TensorBoardPlugin.init(config, log_dir)
            plugins.append(tensorboard_plugin)

        elif plugin_name == "wandb":
            project_name = "bz-experiments"
            entity = None
            if isinstance(plugin_config, dict):
                project_name = plugin_config.get("project_name", project_name)
                entity = plugin_config.get("entity", entity)
            wandb_plugin: WandBPlugin = WandBPlugin.init(config, project_name, entity)
            plugins.append(wandb_plugin)

        elif plugin_name == "profiler":
            log_interval = 10
            enable_gpu_monitoring = True
            if isinstance(plugin_config, dict):
                log_interval = plugin_config.get("log_interval", log_interval)
                enable_gpu_monitoring = plugin_config.get("enable_gpu_monitoring", enable_gpu_monitoring)
            profiler_plugin: ProfilerPlugin = ProfilerPlugin.init(config, log_interval, enable_gpu_monitoring)
            plugins.append(profiler_plugin)

        elif plugin_name == "optuna":
            from .optuna import OptunaConfig

            optuna_config = OptunaConfig(**(plugin_config if isinstance(plugin_config, dict) else {}))
            optuna_plugin: OptunaPlugin = OptunaPlugin(optuna_config)
            plugins.append(optuna_plugin)

        elif plugin_name == "early_stopping":
            from .early_stopping import EarlyStoppingConfig

            early_stopping_config = EarlyStoppingConfig(**(plugin_config if isinstance(plugin_config, dict) else {}))
            early_stopping_plugin: EarlyStoppingPlugin = EarlyStoppingPlugin(early_stopping_config)
            plugins.append(early_stopping_plugin)

        else:
            # Try to create plugin from registry
            plugin = create_plugin(plugin_name, plugin_config if isinstance(plugin_config, dict) else {})
            if plugin is not None:
                plugins.append(plugin)
            else:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Unknown plugin: {plugin_name}")

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
