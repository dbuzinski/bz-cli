"""
Tests for the WandB plugin.
"""

from bz_wandb.wandb_plugin import WandBPlugin


class TestWandBPlugin:
    """Test cases for WandBPlugin."""

    def test_plugin_name(self):
        """Test that the plugin has the correct name."""
        assert WandBPlugin.name == "wandb"

    def test_plugin_creation(self):
        """Test plugin creation."""
        plugin = WandBPlugin("test_project")
        assert plugin.name == "wandb"
        assert plugin.project_name == "test_project"
        assert plugin.entity is None

    def test_plugin_creation_with_entity(self):
        """Test plugin creation with entity."""
        plugin = WandBPlugin("test_project", entity="test_user")
        assert plugin.project_name == "test_project"
        assert plugin.entity == "test_user"

    def test_load_config(self):
        """Test loading configuration from dict."""
        config_data = {"project_name": "test_project", "entity": "test_user", "enabled": True}

        config = WandBPlugin.load_config(config_data)

        assert config["project_name"] == "test_project"
        assert config["entity"] == "test_user"
        assert config["enabled"] is True

    def test_create_plugin_enabled(self):
        """Test creating plugin when enabled."""
        config_data = {"project_name": "test_project", "enabled": True}

        plugin = WandBPlugin.create(config_data, None)
        assert plugin is not None
        assert plugin.project_name == "test_project"

    def test_create_plugin_disabled(self):
        """Test creating plugin when disabled."""
        config_data = {"project_name": "test_project", "enabled": False}

        plugin = WandBPlugin.create(config_data, None)
        assert plugin is None

    def test_create_plugin_default_enabled(self):
        """Test creating plugin when enabled is not specified (defaults to True)."""
        config_data = {"project_name": "test_project"}

        plugin = WandBPlugin.create(config_data, None)
        assert plugin is not None
