"""
Tests for bz CLI functionality.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import MagicMock
from bz.cli import create_parser, load_training_spec, TrainingSpecification, _load_required, _load_optional
from bz.config import ConfigManager
from bz.plugins import get_plugin_registry, Plugin


class TestCLI:
    """Test CLI functionality."""

    def test_create_parser(self):
        """Test argument parser creation."""
        parser = create_parser()
        assert parser is not None

        # Test subparsers exist
        subparsers = list(parser._subparsers._group_actions[0].choices.keys())
        expected_commands = ["train", "validate", "init", "list-plugins", "list-metrics"]
        assert all(cmd in subparsers for cmd in expected_commands)

    def test_load_required_success(self):
        """Test loading required attributes successfully."""
        mock_module = MagicMock()
        mock_module.test_attr = "test_value"

        result = _load_required(mock_module, "test_attr")
        assert result == "test_value"

    def test_load_required_missing(self):
        """Test loading required attributes that don't exist."""
        mock_module = MagicMock()
        del mock_module.test_attr

        with pytest.raises(Exception, match="test_attr must be specified in train.py"):
            _load_required(mock_module, "test_attr")

    def test_load_optional_success(self):
        """Test loading optional attributes successfully."""
        mock_module = MagicMock()
        mock_module.test_attr = "test_value"

        result = _load_optional(mock_module, "test_attr", "default")
        assert result == "test_value"

    def test_load_optional_missing(self):
        """Test loading optional attributes that don't exist."""
        mock_module = MagicMock()
        del mock_module.test_attr

        result = _load_optional(mock_module, "test_attr", "default")
        assert result == "default"

    def test_load_training_spec(self):
        """Test loading training specification."""
        mock_module = MagicMock()
        mock_module.model = "model"
        mock_module.loss_fn = "loss_fn"
        mock_module.optimizer = "optimizer"
        mock_module.training_loader = "training_loader"
        mock_module.validation_loader = "validation_loader"
        mock_module.hyperparameters = {"lr": 0.001}

        spec = load_training_spec(mock_module)

        assert isinstance(spec, TrainingSpecification)
        assert spec.model == "model"
        assert spec.loss_fn == "loss_fn"
        assert spec.optimizer == "optimizer"
        assert spec.training_loader == "training_loader"
        assert spec.validation_loader == "validation_loader"
        assert spec.hyperparameters == {"lr": 0.001}

    def test_load_training_spec_missing_required(self):
        """Test loading training specification with missing required attributes."""
        mock_module = MagicMock()
        mock_module.model = "model"
        # Missing loss_fn
        del mock_module.loss_fn

        with pytest.raises(Exception, match="loss_fn must be specified in train.py"):
            load_training_spec(mock_module)

    def test_load_training_spec_optional_defaults(self):
        """Test loading training specification with optional defaults."""
        mock_module = MagicMock()
        mock_module.model = "model"
        mock_module.loss_fn = "loss_fn"
        mock_module.optimizer = "optimizer"
        mock_module.training_loader = "training_loader"
        # Missing validation_loader and hyperparameters
        del mock_module.validation_loader
        del mock_module.hyperparameters

        spec = load_training_spec(mock_module)

        assert spec.validation_loader is None
        assert spec.hyperparameters == {}


class TestConfigManager:
    """Test configuration management."""

    def test_config_manager_defaults(self):
        """Test ConfigManager with default configuration."""
        with tempfile.TemporaryDirectory():
            config_manager = ConfigManager()
            config = config_manager.load()

            assert "training" in config
            assert "plugins" in config
            assert "metrics" in config
            assert config["environment"] == "development"

    def test_config_manager_with_file(self):
        """Test ConfigManager with configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.json")
            test_config = {
                "training": {"epochs": 20, "batch_size": 128},
                "plugins": {"console_out": {"enabled": True, "config": {}, "dependencies": []}},
            }

            with open(config_path, "w") as f:
                json.dump(test_config, f)

            config_manager = ConfigManager(config_path=config_path)
            config = config_manager.load()

            assert config["training"]["epochs"] == 20
            assert config["training"]["batch_size"] == 128

    def test_config_manager_environment_specific(self):
        """Test ConfigManager with environment-specific configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create environment-specific config
            env_config_path = os.path.join(temp_dir, "bz_config.test.json")
            test_config = {"training": {"epochs": 50, "batch_size": 256}, "environment": "test"}

            with open(env_config_path, "w") as f:
                json.dump(test_config, f)

            config_manager = ConfigManager()
            config_manager.config_path = env_config_path
            config = config_manager.load()

            assert config["training"]["epochs"] == 50
            assert config["training"]["batch_size"] == 256
            assert config["environment"] == "test"

    def test_config_manager_validation(self):
        """Test ConfigManager validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "invalid_config.json")
            invalid_config = {
                "training": {"epochs": -1, "batch_size": 0}  # Invalid: negative epochs  # Invalid: zero batch size
            }

            with open(config_path, "w") as f:
                json.dump(invalid_config, f)

            config_manager = ConfigManager(config_path=config_path)

            with pytest.raises(ValueError, match="epochs must be at least 1"):
                config_manager.load()

    def test_plugin_dependency_validation(self):
        """Test plugin dependency validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "dependency_config.json")
            config_with_deps = {
                "plugins": {
                    "plugin_a": {"enabled": True, "config": {}, "dependencies": ["plugin_b"]},
                    "plugin_b": {"enabled": True, "config": {}, "dependencies": []},
                }
            }

            with open(config_path, "w") as f:
                json.dump(config_with_deps, f)

            config_manager = ConfigManager(config_path=config_path)
            config = config_manager.load()

            # Should not raise an error
            assert "plugins" in config

    def test_plugin_dependency_validation_missing(self):
        """Test plugin dependency validation with missing dependency."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "missing_dep_config.json")
            config_with_missing_dep = {
                "plugins": {"plugin_a": {"enabled": True, "config": {}, "dependencies": ["missing_plugin"]}}
            }

            with open(config_path, "w") as f:
                json.dump(config_with_missing_dep, f)

            config_manager = ConfigManager(config_path=config_path)

            with pytest.raises(
                ValueError, match="Plugin 'plugin_a' depends on 'missing_plugin' which is not configured"
            ):
                config_manager.load()


class TestPluginRegistry:
    """Test plugin registry functionality."""

    def test_plugin_registry_register(self):
        """Test plugin registration."""
        registry = get_plugin_registry()

        class TestPlugin(Plugin):
            pass

        registry.register("test_plugin", TestPlugin)
        assert "test_plugin" in registry.list_plugins()

    def test_plugin_registry_create(self):
        """Test plugin creation."""
        registry = get_plugin_registry()

        class TestPlugin(Plugin):
            def __init__(self, config=None):
                super().__init__(config=config)
                self.test_value = config.get("test_value", "default") if config else "default"

        registry.register("test_plugin", TestPlugin, {"test_value": "custom"})

        plugin = registry.create_plugin("test_plugin")
        assert isinstance(plugin, TestPlugin)
        assert plugin.test_value == "custom"

    def test_plugin_registry_create_with_config(self):
        """Test plugin creation with custom config."""
        registry = get_plugin_registry()

        class TestPlugin(Plugin):
            def __init__(self, config=None):
                super().__init__(config=config)
                self.test_value = config.get("test_value", "default") if config else "default"

        registry.register("test_plugin", TestPlugin, {"test_value": "default"})

        plugin = registry.create_plugin("test_plugin", {"test_value": "override"})
        assert plugin is not None
        assert plugin.test_value == "override"

    def test_plugin_registry_invalid_class(self):
        """Test plugin registration with invalid class."""
        registry = get_plugin_registry()

        class InvalidPlugin:
            pass

        with pytest.raises(ValueError, match="Plugin class must inherit from Plugin"):
            registry.register("invalid_plugin", InvalidPlugin)

    def test_plugin_registry_unregister(self):
        """Test plugin unregistration."""
        registry = get_plugin_registry()

        class TestPlugin(Plugin):
            pass

        registry.register("test_plugin", TestPlugin)
        assert "test_plugin" in registry.list_plugins()

        registry.unregister("test_plugin")
        assert "test_plugin" not in registry.list_plugins()


if __name__ == "__main__":
    pytest.main([__file__])
