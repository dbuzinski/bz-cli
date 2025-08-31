"""
Tests for bz CLI functionality.
"""

import tempfile
import os
import json
from unittest.mock import patch
from bz.cli import create_parser
from bz import get_config, set_cli_config_path


class TestCLI:
    """Test CLI functionality."""

    def test_create_parser(self):
        """Test argument parser creation."""
        parser = create_parser()
        assert parser is not None

        # Test subparsers exist
        subparsers = list(parser._subparsers._group_actions[0].choices.keys())
        expected_commands = ["train", "validate", "init", "list-plugins", "list-metrics", "health"]
        assert all(cmd in subparsers for cmd in expected_commands)

    def test_train_parser_arguments(self):
        """Test train command parser arguments."""
        parser = create_parser()
        train_parser = parser._subparsers._group_actions[0].choices["train"]

        # Check that train parser has expected arguments
        train_args = [action.dest for action in train_parser._actions]
        expected_args = [
            "epochs",
            "checkpoint_interval",
            "no_compile",
            "config",
            "device",
            "optimize",
            "n_trials",
            "study_name",
        ]
        assert all(arg in train_args for arg in expected_args)

    def test_validate_parser_arguments(self):
        """Test validate command parser arguments."""
        parser = create_parser()
        validate_parser = parser._subparsers._group_actions[0].choices["validate"]

        # Check that validate parser has expected arguments
        validate_args = [action.dest for action in validate_parser._actions]
        expected_args = ["model_path", "config"]
        assert all(arg in validate_args for arg in expected_args)

    def test_init_parser_arguments(self):
        """Test init command parser arguments."""
        parser = create_parser()
        init_parser = parser._subparsers._group_actions[0].choices["init"]

        # Check that init parser has expected arguments
        init_args = [action.dest for action in init_parser._actions]
        expected_args = ["template"]
        assert all(arg in init_args for arg in expected_args)

    def test_list_plugins_parser_arguments(self):
        """Test list-plugins command parser arguments."""
        parser = create_parser()
        list_plugins_parser = parser._subparsers._group_actions[0].choices["list-plugins"]

        # Check that list-plugins parser has expected arguments
        list_plugins_args = [action.dest for action in list_plugins_parser._actions]
        expected_args = ["config"]
        assert all(arg in list_plugins_args for arg in expected_args)

    def test_health_parser_arguments(self):
        """Test health command parser arguments."""
        parser = create_parser()
        health_parser = parser._subparsers._group_actions[0].choices["health"]

        # Check that health parser has expected arguments
        health_args = [action.dest for action in health_parser._actions]
        expected_args = ["json"]
        assert all(arg in health_args for arg in expected_args)


class TestTrainingConfigurationIntegration:
    """Test CLI integration with TrainingConfiguration."""

    def test_cli_config_path_setting(self):
        """Test that CLI config path is properly set."""
        # Test setting CLI config path
        set_cli_config_path("test_config.json")

        # Verify it affects get_config behavior
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = '{"epochs": 42}'

                config = get_config()
                assert config.epochs == 42

    def test_config_precedence_in_cli(self):
        """Test configuration precedence in CLI context."""
        # Set CLI config path
        set_cli_config_path("cli_config.json")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {"epochs": 100, "device": "cpu"}
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(config_data)

                    config = get_config()
                    assert config.epochs == 100
                    assert config.device == "cpu"

        finally:
            os.unlink(config_path)


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_parser_help(self):
        """Test that parser help is properly formatted."""
        parser = create_parser()
        help_text = parser.format_help()

        # Check that help contains expected sections
        assert "A tool to help train machine learning models" in help_text
        assert "train" in help_text
        assert "validate" in help_text
        assert "init" in help_text
        assert "list-plugins" in help_text
        assert "list-metrics" in help_text
        assert "health" in help_text

    def test_examples_in_help(self):
        """Test that examples are included in help."""
        parser = create_parser()
        help_text = parser.format_help()

        # Check that examples are present
        assert "bz train" in help_text
        assert "bz train --epochs 10" in help_text
        assert "bz train --config my_config.json" in help_text
        assert "bz validate" in help_text
        assert "bz init" in help_text
        assert "bz list-plugins" in help_text
        assert "bz list-metrics" in help_text
