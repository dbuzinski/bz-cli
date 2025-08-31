"""
Tests for the PluginRegistry functionality.
"""

from bz.plugins import (
    Plugin,
    PluginRegistry,
    get_plugin_registry,
    register_plugin,
    create_plugin,
    list_plugins,
)


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

    def test_create_plugin(self):
        """Test plugin creation."""
        registry = PluginRegistry()

        class TestPlugin(Plugin):
            pass

        registry.register("test_plugin", TestPlugin, {"default_config": "value"})
        plugin = registry.create_plugin("test_plugin", {"custom_config": "custom_value"})

        assert isinstance(plugin, TestPlugin)
        assert plugin.name == "TestPlugin"  # Default name is class name
        # Config is merged with default config
        assert plugin.config == {"default_config": "value", "custom_config": "custom_value"}

    def test_create_plugin_with_default_config(self):
        """Test plugin creation with default config."""
        registry = PluginRegistry()

        class TestPlugin(Plugin):
            pass

        registry.register("test_plugin", TestPlugin, {"default_config": "value"})
        plugin = registry.create_plugin("test_plugin")

        assert isinstance(plugin, TestPlugin)
        assert plugin.config == {"default_config": "value"}

    def test_create_nonexistent_plugin(self):
        """Test creating a plugin that doesn't exist."""
        registry = PluginRegistry()

        plugin = registry.create_plugin("nonexistent_plugin")
        assert plugin is None

    def test_list_plugins(self):
        """Test listing registered plugins."""
        registry = PluginRegistry()

        class TestPlugin1(Plugin):
            pass

        class TestPlugin2(Plugin):
            pass

        registry.register("plugin1", TestPlugin1, {})
        registry.register("plugin2", TestPlugin2, {})

        plugins = registry.list_plugins()
        assert "plugin1" in plugins
        assert "plugin2" in plugins

    def test_global_registry_functions(self):
        """Test global registry functions."""
        # Clear any existing registry
        registry = get_plugin_registry()
        registry._plugins.clear()
        registry._plugin_configs.clear()

        class TestPlugin(Plugin):
            pass

        # Test register_plugin
        register_plugin("test_plugin", TestPlugin, {"default_config": "value"})

        # Test list_plugins
        plugins = list_plugins()
        assert "test_plugin" in plugins

        # Test create_plugin
        plugin = create_plugin("test_plugin")
        assert isinstance(plugin, TestPlugin)
        assert plugin.name == "TestPlugin"  # Default name is class name
