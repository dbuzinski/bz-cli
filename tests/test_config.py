"""
Tests for the configuration system.
"""

import json
import os
import tempfile
import pytest
from unittest.mock import patch
from bz import TrainingConfiguration, get_config, set_cli_config_path


class TestTrainingConfiguration:
    """Test cases for TrainingConfiguration."""

    def test_training_configuration_creation(self):
        """Test creating a TrainingConfiguration instance."""
        config = TrainingConfiguration(epochs=10)

        assert config.epochs == 10
        assert config.device == "auto"
        assert config.compile is True
        assert config.checkpoint_interval == 5
        assert config.hyperparameters == {}
        assert config.metrics == []
        assert config.plugins == []

    def test_training_configuration_validation(self):
        """Test TrainingConfiguration validation."""
        # Should raise error for invalid epochs
        with pytest.raises(ValueError, match="epochs must be at least 1"):
            TrainingConfiguration(epochs=0)

        with pytest.raises(ValueError, match="epochs must be at least 1"):
            TrainingConfiguration(epochs=-1)

    def test_training_configuration_with_all_fields(self):
        """Test TrainingConfiguration with all fields set."""
        config = TrainingConfiguration(
            epochs=5,
            device="cuda",
            compile=False,
            checkpoint_interval=10,
            hyperparameters={"lr": 0.001, "batch_size": 64},
            metrics=["accuracy", "precision"],
            plugins=["console_out", "tensorboard"],
        )

        assert config.epochs == 5
        assert config.device == "cuda"
        assert config.compile is False
        assert config.checkpoint_interval == 10
        assert config.hyperparameters == {"lr": 0.001, "batch_size": 64}
        assert config.metrics == ["accuracy", "precision"]
        assert config.plugins == ["console_out", "tensorboard"]


class TestGetConfig:
    """Test cases for get_config function."""

    def test_get_config_with_valid_file(self):
        """Test get_config with a valid configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "epochs": 15,
                "device": "cpu",
                "compile": False,
                "checkpoint_interval": 3,
                "hyperparameters": {"lr": 0.01, "batch_size": 128},
                "metrics": ["accuracy", "f1_score"],
                "plugins": ["console_out"],
            }
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch("os.environ.get", return_value=config_path):
                with patch("bz._cli_config_path", None):  # Clear any CLI config path
                    config = get_config()

                    assert config.epochs == 15
                    assert config.device == "cpu"
                    assert config.compile is False
                    assert config.checkpoint_interval == 3
                    assert config.hyperparameters == {"lr": 0.01, "batch_size": 128}
                    # Metrics are now instantiated objects, not strings
                    assert len(config.metrics) == 2
                    assert config.plugins == ["console_out"]

        finally:
            os.unlink(config_path)

    def test_get_config_with_missing_file(self):
        """Test get_config with a missing configuration file."""
        with patch("os.environ.get", return_value="nonexistent.json"):
            with pytest.raises(
                FileNotFoundError, match="Configuration file specified by BZ_CONFIG environment variable not found"
            ):
                get_config()

    def test_get_config_with_default_file_missing(self):
        """Test get_config when bzconfig.json is missing."""
        with patch("os.environ.get", return_value="bzconfig.json"):
            with patch("os.path.exists", return_value=False):
                with patch("bz._cli_config_path", None):  # Clear any CLI config path
                    with pytest.raises(FileNotFoundError, match="Configuration file bzconfig.json not found"):
                        get_config()

    def test_get_config_with_invalid_json(self):
        """Test get_config with invalid JSON in config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"epochs": 10, "invalid": json}')
            config_path = f.name

        try:
            with patch("os.environ.get", return_value=config_path):
                with patch("bz._cli_config_path", None):  # Clear any CLI config path
                    with pytest.raises(ValueError, match="Invalid JSON"):
                        get_config()

        finally:
            os.unlink(config_path)

    def test_get_config_precedence(self):
        """Test configuration precedence order."""
        # Test CLI config path takes precedence
        set_cli_config_path("cli_config.json")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {"epochs": 20}
            json.dump(config_data, f)
            cli_config_path = f.name

        try:
            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = '{"epochs": 20}'

                    config = get_config()
                    assert config.epochs == 20

        finally:
            os.unlink(cli_config_path)

    def test_get_config_with_environment_variable(self):
        """Test get_config using BZ_CONFIG environment variable."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {"epochs": 25}
            json.dump(config_data, f)
            env_config_path = f.name

        try:
            with patch("os.environ.get", return_value=env_config_path):
                with patch("bz._cli_config_path", None):  # No CLI config set
                    config = get_config()
                    assert config.epochs == 25

        finally:
            os.unlink(env_config_path)


class TestSetCliConfigPath:
    """Test cases for set_cli_config_path function."""

    def test_set_cli_config_path(self):
        """Test setting CLI config path."""
        set_cli_config_path("test_config.json")

        # The function should set the global variable
        # We can't easily test the global variable directly, but we can test
        # that it affects get_config behavior
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = '{"epochs": 30}'

                config = get_config()
                assert config.epochs == 30
