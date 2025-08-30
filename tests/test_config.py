"""
Tests for the configuration system.
"""

import json
import os
import tempfile
import pytest
from bz.config import ConfigManager, load_config, get_config_manager


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_default_config(self):
        """Test that default config is loaded when no file exists."""
        config_manager = ConfigManager(config_path=None)
        config = config_manager.load()
        
        # Check that default values are present
        assert "training" in config
        assert "plugins" in config
        assert "metrics" in config
        assert "hyperparameters" in config
        
        # Check training defaults
        training = config["training"]
        assert training["epochs"] == 1
        assert training["batch_size"] == 32
        assert training["learning_rate"] == 0.001
        assert training["device"] == "auto"
        assert training["compile"] is True
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "training": {
                    "epochs": 10,
                    "batch_size": 64,
                    "learning_rate": 0.01
                },
                "hyperparameters": {
                    "lr": 0.01,
                    "batch_size": 64
                }
            }
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path=config_path)
            config = config_manager.load()
            
            # Check that custom values are loaded
            assert config["training"]["epochs"] == 10
            assert config["training"]["batch_size"] == 64
            assert config["training"]["learning_rate"] == 0.01
            
            # Check that defaults are preserved for missing values
            assert config["training"]["device"] == "auto"
            assert config["training"]["compile"] is True
            
            # Check hyperparameters
            assert config["hyperparameters"]["lr"] == 0.01
            assert config["hyperparameters"]["batch_size"] == 64
            
        finally:
            os.unlink(config_path)
    
    def test_config_validation(self):
        """Test configuration validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "training": {
                    "epochs": -1,  # Invalid: negative epochs
                    "batch_size": 0,  # Invalid: zero batch size
                    "learning_rate": -0.1,  # Invalid: negative learning rate
                    "device": "invalid_device"  # Invalid device
                }
            }
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path=config_path)
            
            # Should raise validation errors
            with pytest.raises(ValueError, match="epochs must be at least 1"):
                config_manager.load()
                
        finally:
            os.unlink(config_path)
    
    def test_plugin_config(self):
        """Test plugin configuration access."""
        config_manager = ConfigManager()
        config = config_manager.load()
        
        # Test getting plugin config
        console_config = config_manager.get_plugin_config("console_out")
        assert console_config is not None
        assert console_config["enabled"] is True
        
        # Test non-existent plugin
        nonexistent_config = config_manager.get_plugin_config("nonexistent_plugin")
        assert nonexistent_config is None
    
    def test_plugin_enabled(self):
        """Test plugin enabled status."""
        config_manager = ConfigManager()
        
        # Test enabled plugin
        assert config_manager.is_plugin_enabled("console_out") is True
        
        # Test disabled plugin
        assert config_manager.is_plugin_enabled("tensorboard") is False
        
        # Test non-existent plugin
        assert config_manager.is_plugin_enabled("nonexistent_plugin") is False
    
    def test_metrics_config(self):
        """Test metrics configuration access."""
        config_manager = ConfigManager()
        metrics_config = config_manager.get_metrics_config()
        
        assert "metrics" in metrics_config
        assert "custom_metrics" in metrics_config
        assert metrics_config["metrics"] == ["accuracy"]
    
    def test_hyperparameters(self):
        """Test hyperparameters access."""
        config_manager = ConfigManager()
        hyperparams = config_manager.get_hyperparameters()
        
        assert isinstance(hyperparams, dict)
    
    def test_deep_merge(self):
        """Test deep merging of configurations."""
        config_manager = ConfigManager()
        
        default_config = {
            "training": {
                "epochs": 1,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "plugins": {
                "console_out": {"enabled": True, "config": {}}
            }
        }
        
        override_config = {
            "training": {
                "epochs": 10,
                "batch_size": 64
            },
            "plugins": {
                "tensorboard": {"enabled": True, "config": {"log_dir": "custom_dir"}}
            }
        }
        
        merged = config_manager._deep_merge(default_config, override_config)
        
        # Check that values are properly merged
        assert merged["training"]["epochs"] == 10
        assert merged["training"]["batch_size"] == 64
        assert merged["training"]["learning_rate"] == 0.001  # Preserved from default
        
        # Check that new plugins are added
        assert "tensorboard" in merged["plugins"]
        assert merged["plugins"]["tensorboard"]["enabled"] is True
        assert merged["plugins"]["console_out"]["enabled"] is True  # Preserved from default


class TestLoadConfig:
    """Test cases for backward compatibility load_config function."""
    
    def test_load_config_with_path(self):
        """Test load_config with explicit path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "lr": 0.01,
                "batch_size": 64,
                "custom_param": "value"
            }
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            result = load_config(config_path)
            assert result["lr"] == 0.01
            assert result["batch_size"] == 64
            assert result["custom_param"] == "value"
        finally:
            os.unlink(config_path)
    
    def test_load_config_env_var(self):
        """Test load_config with environment variable."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"lr": 0.02, "batch_size": 128}
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            # Set environment variable
            os.environ["BZ_CONFIG"] = config_path
            
            result = load_config()
            assert result["lr"] == 0.02
            assert result["batch_size"] == 128
            
        finally:
            # Clean up
            os.unlink(config_path)
            if "BZ_CONFIG" in os.environ:
                del os.environ["BZ_CONFIG"]
    
    def test_load_config_default_file(self):
        """Test load_config with default config.json file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"lr": 0.03, "batch_size": 256}
            json.dump(config_data, f)
            config_path = f.name
        
        # Rename to config.json in current directory
        original_cwd = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        
        try:
            os.chdir(temp_dir)
            os.rename(config_path, "config.json")
            
            result = load_config()
            assert result["lr"] == 0.03
            assert result["batch_size"] == 256
            
        finally:
            # Clean up
            os.chdir(original_cwd)
            if os.path.exists("config.json"):
                os.unlink("config.json")
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
    
    def test_load_config_no_file(self):
        """Test load_config when no file exists."""
        result = load_config()
        assert result == {}


class TestGetConfigManager:
    """Test cases for global config manager."""
    
    def test_get_config_manager_singleton(self):
        """Test that get_config_manager returns the same instance."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2
    
    def test_get_config_manager_initialization(self):
        """Test that get_config_manager properly initializes."""
        manager = get_config_manager()
        config = manager.load()
        
        assert "training" in config
        assert "plugins" in config
        assert "metrics" in config


if __name__ == "__main__":
    pytest.main([__file__])
