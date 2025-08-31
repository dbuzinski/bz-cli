"""
Tests for the Profiler plugin.
"""

from bz_profiler.profiler_plugin import ProfilerPlugin


class TestProfilerPlugin:
    """Test cases for ProfilerPlugin."""

    def test_plugin_name(self):
        """Test that the plugin has the correct name."""
        assert ProfilerPlugin.name == "profiler"

    def test_plugin_creation(self):
        """Test plugin creation."""
        plugin = ProfilerPlugin()
        assert plugin.name == "profiler"
        assert plugin.log_interval == 10
        assert plugin.enable_gpu_monitoring is True

    def test_plugin_creation_with_custom_params(self):
        """Test plugin creation with custom parameters."""
        plugin = ProfilerPlugin(log_interval=5, enable_gpu_monitoring=False)
        assert plugin.log_interval == 5
        assert plugin.enable_gpu_monitoring is False

    def test_load_config(self):
        """Test loading configuration from dict."""
        config_data = {"log_interval": 15, "enable_gpu_monitoring": False, "enabled": True}

        config = ProfilerPlugin.load_config(config_data)

        assert config["log_interval"] == 15
        assert config["enable_gpu_monitoring"] is False
        assert config["enabled"] is True

    def test_create_plugin_enabled(self):
        """Test creating plugin when enabled."""
        config_data = {"log_interval": 20, "enabled": True}

        plugin = ProfilerPlugin.create(config_data, None)
        assert plugin is not None
        assert plugin.log_interval == 20

    def test_create_plugin_disabled(self):
        """Test creating plugin when disabled."""
        config_data = {"log_interval": 20, "enabled": False}

        plugin = ProfilerPlugin.create(config_data, None)
        assert plugin is None

    def test_create_plugin_default_enabled(self):
        """Test creating plugin when enabled is not specified (defaults to True)."""
        config_data = {"log_interval": 20}

        plugin = ProfilerPlugin.create(config_data, None)
        assert plugin is not None
