"""
Tests for the ConsoleOutPlugin.
"""

from unittest.mock import Mock, patch
from bz.plugins import PluginContext
from bz.plugins.console_out import ConsoleOutPlugin


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
