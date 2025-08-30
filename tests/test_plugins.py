"""
Tests for bz plugin system.
"""

import pytest
from unittest.mock import Mock, patch
from bz.plugins import (
    Plugin,
    PluginContext,
    PluginRegistry,
    get_plugin_registry,
    register_plugin,
    create_plugin,
    list_plugins,
)
from bz.plugins.console_out import ConsoleOutPlugin
from bz.plugins.tensorboard import TensorBoardPlugin


class TestPlugin:
    """Test base Plugin class."""

    def test_plugin_initialization(self):
        """Test plugin initialization."""
        plugin = Plugin(name="test_plugin", config={"test": "value"})
        assert plugin.name == "test_plugin"
        assert plugin.config == {"test": "value"}

    def test_plugin_default_name(self):
        """Test plugin uses class name as default name."""

        class TestPlugin(Plugin):
            pass

        plugin = TestPlugin()
        assert plugin.name == "TestPlugin"

    def test_plugin_run_stage(self):
        """Test plugin stage execution."""

        class TestPlugin(Plugin):
            def start_training_session(self, context):
                context.extra["test"] = "called"

        plugin = TestPlugin()
        context = PluginContext()

        plugin.run_stage("start_training_session", context)
        assert context.extra["test"] == "called"

    def test_plugin_run_stage_missing_method(self):
        """Test plugin stage execution with missing method."""
        plugin = Plugin()
        context = PluginContext()

        # Should not raise an error
        plugin.run_stage("nonexistent_stage", context)

    def test_plugin_context(self):
        """Test PluginContext functionality."""
        context = PluginContext()

        # Test default values
        assert context.epoch == 0
        assert context.training_loss_total == 0.0
        assert context.validation_loss_total == 0.0
        assert context.training_batch_count == 0
        assert context.validation_batch_count == 0
        assert context.training_metrics == {}
        assert context.validation_metrics == {}
        assert context.hyperparameters == {}
        assert context.extra == {}

        # Test modification
        context.epoch = 5
        context.training_metrics["accuracy"] = 0.95
        context.extra["checkpoint_path"] = "/path/to/checkpoint"

        assert context.epoch == 5
        assert context.training_metrics["accuracy"] == 0.95
        assert context.extra["checkpoint_path"] == "/path/to/checkpoint"


class TestPluginRegistry:
    """Test PluginRegistry functionality."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = PluginRegistry()
        assert registry._plugins == {}
        assert registry._plugin_configs == {}

    def test_register_plugin(self):
        """Test plugin registration."""
        registry = PluginRegistry()

        class TestPlugin(Plugin):
            pass

        registry.register("test_plugin", TestPlugin, {"default_config": "value"})

        assert "test_plugin" in registry._plugins
        assert registry._plugins["test_plugin"] == TestPlugin
        assert registry._plugin_configs["test_plugin"] == {"default_config": "value"}

    def test_register_invalid_plugin(self):
        """Test registration of invalid plugin class."""
        registry = PluginRegistry()

        class InvalidPlugin:
            pass

        with pytest.raises(ValueError, match="Plugin class must inherit from Plugin"):
            registry.register("invalid_plugin", InvalidPlugin)

    def test_create_plugin(self):
        """Test plugin creation."""
        registry = PluginRegistry()

        class TestPlugin(Plugin):
            def __init__(self, config=None):
                super().__init__(config=config)
                self.test_value = config.get("test_value", "default") if config else "default"

        registry.register("test_plugin", TestPlugin, {"test_value": "default"})

        plugin = registry.create_plugin("test_plugin")
        assert isinstance(plugin, TestPlugin)
        assert plugin.test_value == "default"

    def test_create_plugin_with_config(self):
        """Test plugin creation with custom config."""
        registry = PluginRegistry()

        class TestPlugin(Plugin):
            def __init__(self, config=None):
                super().__init__(config=config)
                self.test_value = config.get("test_value", "default") if config else "default"

        registry.register("test_plugin", TestPlugin, {"test_value": "default"})

        plugin = registry.create_plugin("test_plugin", {"test_value": "custom"})
        assert plugin.test_value == "custom"

    def test_create_nonexistent_plugin(self):
        """Test creation of nonexistent plugin."""
        registry = PluginRegistry()

        plugin = registry.create_plugin("nonexistent_plugin")
        assert plugin is None

    def test_list_plugins(self):
        """Test plugin listing."""
        registry = PluginRegistry()

        class TestPlugin1(Plugin):
            pass

        class TestPlugin2(Plugin):
            pass

        registry.register("plugin1", TestPlugin1)
        registry.register("plugin2", TestPlugin2)

        plugins = registry.list_plugins()
        assert "plugin1" in plugins
        assert "plugin2" in plugins
        assert len(plugins) == 2

    def test_unregister_plugin(self):
        """Test plugin unregistration."""
        registry = PluginRegistry()

        class TestPlugin(Plugin):
            pass

        registry.register("test_plugin", TestPlugin, {"config": "value"})
        assert "test_plugin" in registry.list_plugins()

        registry.unregister("test_plugin")
        assert "test_plugin" not in registry.list_plugins()
        assert "test_plugin" not in registry._plugin_configs


class TestConsoleOutPlugin:
    """Test ConsoleOutPlugin functionality."""

    def test_console_plugin_initialization(self):
        """Test ConsoleOutPlugin initialization."""
        plugin = ConsoleOutPlugin(10, 5, 2)
        assert plugin.training_data_len == 10
        assert plugin.validation_data_len == 5
        assert plugin.update_interval == 2

    def test_console_plugin_start_training_session(self):
        """Test ConsoleOutPlugin start_training_session."""
        plugin = ConsoleOutPlugin(10)
        context = PluginContext()

        plugin.start_training_session(context)
        assert plugin.training_start_time is not None

    def test_console_plugin_start_epoch(self):
        """Test ConsoleOutPlugin start_epoch."""
        plugin = ConsoleOutPlugin(10)
        context = PluginContext()
        context.epoch = 0

        # Mock print to capture output
        with patch("builtins.print") as mock_print:
            plugin.start_epoch(context)
            mock_print.assert_called_with("Epoch 1:")

    def test_console_plugin_load_checkpoint(self):
        """Test ConsoleOutPlugin load_checkpoint."""
        plugin = ConsoleOutPlugin(10)
        context = PluginContext()
        context.extra["start_epoch"] = 5
        context.extra["checkpoint_path"] = "/path/to/checkpoint"

        with patch("builtins.print") as mock_print:
            plugin.load_checkpoint(context)
            mock_print.assert_called_with("✓ Epoch 5 loaded from /path/to/checkpoint")

    def test_console_plugin_init(self):
        """Test ConsoleOutPlugin.init static method."""
        mock_spec = Mock()
        mock_spec.training_loader = Mock()
        mock_spec.training_loader.__len__ = Mock(return_value=10)
        mock_spec.validation_loader = Mock()
        mock_spec.validation_loader.__len__ = Mock(return_value=5)

        plugin = ConsoleOutPlugin.init(mock_spec)
        assert isinstance(plugin, ConsoleOutPlugin)
        assert plugin.training_data_len == 10
        assert plugin.validation_data_len == 5


class TestTensorBoardPlugin:
    """Test TensorBoardPlugin functionality."""

    def test_tensorboard_plugin_initialization(self):
        """Test TensorBoardPlugin initialization."""
        plugin = TensorBoardPlugin(10, "/test/log/dir")
        assert plugin.training_loader_len == 10
        assert plugin.log_dir == "/test/log/dir"
        assert plugin.writer is None

    @patch("bz.plugins.tensorboard.SummaryWriter")
    def test_tensorboard_plugin_start_training_session(self, mock_writer_class):
        """Test TensorBoardPlugin start_training_session."""
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer

        plugin = TensorBoardPlugin(10, "/test/log/dir")
        context = PluginContext()

        plugin.start_training_session(context)
        mock_writer_class.assert_called_with("/test/log/dir")
        assert plugin.writer == mock_writer

    @patch("bz.plugins.tensorboard.SummaryWriter")
    def test_tensorboard_plugin_start_training_session_error(self, mock_writer_class):
        """Test TensorBoardPlugin start_training_session with error."""
        mock_writer_class.side_effect = Exception("TensorBoard error")

        plugin = TensorBoardPlugin(10, "/test/log/dir")
        context = PluginContext()

        plugin.start_training_session(context)
        assert plugin.writer is None

    @patch("bz.plugins.tensorboard.SummaryWriter")
    def test_tensorboard_plugin_end_training_batch(self, mock_writer_class):
        """Test TensorBoardPlugin end_training_batch."""
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer

        plugin = TensorBoardPlugin(10, "/test/log/dir")
        plugin.writer = mock_writer

        context = PluginContext()
        context.epoch = 0
        context.training_batch_count = 5
        context.training_loss_total = 10.0

        plugin.end_training_batch(context)
        mock_writer.add_scalar.assert_called_with("Loss/Train Step", 2.0, 5)

    @patch("bz.plugins.tensorboard.SummaryWriter")
    def test_tensorboard_plugin_end_training_session(self, mock_writer_class):
        """Test TensorBoardPlugin end_training_session."""
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer

        plugin = TensorBoardPlugin(10, "/test/log/dir")
        plugin.writer = mock_writer

        context = PluginContext()

        plugin.end_training_session(context)
        mock_writer.close.assert_called_once()
        assert plugin.writer is None

    def test_tensorboard_plugin_init(self):
        """Test TensorBoardPlugin.init static method."""
        mock_spec = Mock()
        mock_spec.training_loader = Mock()
        mock_spec.training_loader.__len__ = Mock(return_value=10)

        plugin = TensorBoardPlugin.init(mock_spec, "/custom/log/dir")
        assert isinstance(plugin, TensorBoardPlugin)
        assert plugin.training_loader_len == 10
        assert plugin.log_dir == "/custom/log/dir"


class TestGlobalPluginRegistry:
    """Test global plugin registry functionality."""

    def test_get_plugin_registry(self):
        """Test getting global plugin registry."""
        registry = get_plugin_registry()
        assert isinstance(registry, PluginRegistry)

    def test_register_plugin_global(self):
        """Test global plugin registration."""

        class TestPlugin(Plugin):
            pass

        register_plugin("global_test_plugin", TestPlugin, {"test": "value"})

        registry = get_plugin_registry()
        assert "global_test_plugin" in registry.list_plugins()

    def test_create_plugin_global(self):
        """Test global plugin creation."""

        class TestPlugin(Plugin):
            def __init__(self, config=None):
                super().__init__(config=config)
                self.test_value = config.get("test_value", "default") if config else "default"

        register_plugin("global_create_test", TestPlugin, {"test_value": "default"})

        plugin = create_plugin("global_create_test", {"test_value": "custom"})
        assert isinstance(plugin, TestPlugin)
        assert plugin.test_value == "custom"

    def test_list_plugins_global(self):
        """Test global plugin listing."""
        plugins = list_plugins()
        assert isinstance(plugins, list)
        # Should include built-in plugins
        assert "console_out" in plugins
        assert "tensorboard" in plugins


if __name__ == "__main__":
    pytest.main([__file__])
