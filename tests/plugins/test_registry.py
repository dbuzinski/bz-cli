"""
Tests for the PluginRegistry functionality.
"""

from unittest.mock import patch, MagicMock

from bz.plugins import (
    Plugin,
    PluginRegistry,
    get_plugin_registry,
    create_plugin,
    list_plugins,
)


class TestPluginRegistry:
    """Test PluginRegistry functionality."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        with patch("bz.plugins.PluginRegistry._discover_and_register"):
            registry = PluginRegistry()
            assert registry._plugins == {}

    def test_register_builtin_plugins(self):
        """Test builtin plugin registration."""
        # Create a registry without calling _discover_and_register
        registry = PluginRegistry()
        # Clear any existing plugins and manually call the built-in registration
        registry._plugins.clear()
        registry._register_builtin_plugins()

        # Should have registered built-in plugins
        assert "console_out" in registry._plugins
        assert "tensorboard" in registry._plugins
        assert "early_stopping" in registry._plugins

    def test_create_plugin_with_name_and_create_method(self):
        """Test plugin creation with name and create method."""
        with patch("bz.plugins.PluginRegistry._discover_and_register"):
            registry = PluginRegistry()

            class TestPlugin(Plugin):
                name = "test_plugin"

                @staticmethod
                def create(config_data, training_config):
                    return TestPlugin()

            registry._plugins["test_plugin"] = TestPlugin

            plugin = registry.create_plugin("test_plugin", {"test": "config"}, None)
            assert isinstance(plugin, TestPlugin)

    def test_create_plugin_missing_create_method(self):
        """Test plugin creation when plugin class lacks create method."""
        with patch("bz.plugins.PluginRegistry._discover_and_register"):
            registry = PluginRegistry()

            class TestPlugin(Plugin):
                name = "test_plugin"

            registry._plugins["test_plugin"] = TestPlugin

            plugin = registry.create_plugin("test_plugin", {"test": "config"}, None)
            assert plugin is None

    def test_create_nonexistent_plugin(self):
        """Test creating a plugin that doesn't exist."""
        with patch("bz.plugins.PluginRegistry._discover_and_register"):
            registry = PluginRegistry()
            plugin = registry.create_plugin("nonexistent_plugin", {}, None)
            assert plugin is None

    def test_list_plugins(self):
        """Test listing registered plugins."""
        with patch("bz.plugins.PluginRegistry._discover_and_register"):
            registry = PluginRegistry()

            class TestPlugin1(Plugin):
                name = "plugin1"

            class TestPlugin2(Plugin):
                name = "plugin2"

            registry._plugins["plugin1"] = TestPlugin1
            registry._plugins["plugin2"] = TestPlugin2

            plugins = registry.list_plugins()
            assert "plugin1" in plugins
            assert "plugin2" in plugins

    def test_get_plugin_class(self):
        """Test getting plugin class by name."""
        with patch("bz.plugins.PluginRegistry._discover_and_register"):
            registry = PluginRegistry()

            class TestPlugin(Plugin):
                name = "test_plugin"

            registry._plugins["test_plugin"] = TestPlugin

            plugin_class = registry.get_plugin_class("test_plugin")
            assert plugin_class == TestPlugin

            plugin_class = registry.get_plugin_class("nonexistent")
            assert plugin_class is None

    def test_global_registry_functions(self):
        """Test global registry functions."""
        # Clear any existing registry
        registry = get_plugin_registry()
        registry._plugins.clear()

        class TestPlugin(Plugin):
            name = "test_plugin"

            @staticmethod
            def create(config_data, training_config):
                return TestPlugin()

        # Add test plugin to registry
        registry._plugins["test_plugin"] = TestPlugin

        # Test list_plugins
        plugins = list_plugins()
        assert "test_plugin" in plugins

        # Test create_plugin
        plugin = create_plugin("test_plugin", {"test": "config"}, None)
        assert isinstance(plugin, TestPlugin)

    def test_entry_points_discovery(self):
        """Test entry points discovery."""
        mock_plugin_class = MagicMock()
        mock_plugin_class.name = "test_plugin"

        mock_entry_point = MagicMock()
        mock_entry_point.name = "test_plugin"
        mock_entry_point.load.return_value = mock_plugin_class

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value.select.return_value = [mock_entry_point]

            registry = PluginRegistry()
            assert "test_plugin" in registry._plugins

    def test_entry_points_discovery_fallback(self):
        """Test entry points discovery fallback."""
        with patch("importlib.metadata.entry_points") as mock_entry_points:
            # Simulate older Python version without select method
            mock_entry_points.return_value.select.side_effect = AttributeError()
            mock_entry_points.return_value.get.return_value = []

            registry = PluginRegistry()
            # Should fall back to built-in plugins
            assert "console_out" in registry._plugins
