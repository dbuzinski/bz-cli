"""
Tests for the base Plugin class and PluginContext.
"""

from bz.plugins import Plugin, PluginContext


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
