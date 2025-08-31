"""
Tests for the TensorBoardPlugin.
"""

from unittest.mock import Mock, patch
from bz.plugins import PluginContext
from bz.plugins.tensorboard import TensorBoardPlugin


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
        context.training_loss_total = 2.5
        context.training_metrics = {"accuracy": 0.95}

        plugin.end_training_batch(context)

        # Check that writer.add_scalar was called
        mock_writer.add_scalar.assert_called()
