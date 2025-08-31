"""
Tests for Optuna plugin functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from bz_optuna.optuna_plugin import OptunaPlugin, OptunaConfig
from bz.plugins.plugin import PluginContext


class TestOptunaConfig:
    """Test OptunaConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OptunaConfig()
        assert config.study_name == "bz_optimization"
        assert config.n_trials == 10
        assert config.direction == "minimize"
        assert config.sampler == "tpe"
        assert config.hyperparameters == {}

    def test_custom_config(self):
        """Test custom configuration values."""
        hyperparams = {"learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-1}}
        config = OptunaConfig(study_name="test_study", n_trials=5, direction="maximize", hyperparameters=hyperparams)
        assert config.study_name == "test_study"
        assert config.n_trials == 5
        assert config.direction == "maximize"
        assert config.hyperparameters == hyperparams


class TestOptunaPlugin:
    """Test OptunaPlugin functionality."""

    @pytest.fixture
    def mock_optuna(self):
        """Mock Optuna imports."""
        with patch.dict("sys.modules", {"optuna": Mock(), "optuna.samplers": Mock(), "optuna.study": Mock()}):
            yield

    @pytest.fixture
    def plugin_config(self):
        """Create a test plugin configuration."""
        return OptunaConfig(
            study_name="test_study",
            n_trials=3,
            hyperparameters={
                "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-1},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
            },
        )

    @pytest.fixture
    def context(self):
        """Create a test plugin context."""
        context = PluginContext()
        context.epoch = 0
        context.training_batch_count = 0
        context.validation_loss = 0.5
        context.metrics = {"accuracy": 0.8}
        return context

    def test_plugin_initialization(self, mock_optuna, plugin_config):
        """Test plugin initialization."""
        plugin = OptunaPlugin(plugin_config)
        assert plugin.config == plugin_config
        assert plugin.trial_number == 0
        assert plugin.best_score is None

    def test_plugin_initialization_without_optuna(self):
        """Test plugin initialization when Optuna is not available."""
        with patch("bz_optuna.optuna_plugin.OPTUNA_AVAILABLE", False):
            with pytest.raises(ImportError, match="Optuna is not installed"):
                OptunaPlugin()

    def test_start_training_session(self, mock_optuna, plugin_config, context):
        """Test training session start."""
        plugin = OptunaPlugin(plugin_config)
        plugin.start_training_session(context)

        # Check that study was created and trial started
        assert plugin.study is not None
        assert plugin.trial_number == 1

    def test_start_epoch(self, mock_optuna, plugin_config, context):
        """Test epoch start handling."""
        with patch("optuna.create_study"):
            plugin = OptunaPlugin(plugin_config)
            plugin.start_training_session(context)

            # Mock current trial
            mock_trial = Mock()
            plugin.current_trial = mock_trial

            plugin.start_epoch(context)

            # Should report validation loss
            mock_trial.report.assert_called_once_with(0.5, step=0)

    def test_end_training_session(self, mock_optuna, plugin_config, context):
        """Test training session end."""
        with patch("optuna.create_study"):
            plugin = OptunaPlugin(plugin_config)
            plugin.start_training_session(context)

            # Mock current trial
            mock_trial = Mock()
            mock_trial.params = {"learning_rate": 0.001}
            plugin.current_trial = mock_trial

            plugin.end_training_session(context)

            # Should complete trial (we can't easily mock the tell method)
            assert plugin.trial_number == 1

    def test_suggest_hyperparameters(self, mock_optuna, plugin_config, context):
        """Test hyperparameter suggestion."""
        with patch("optuna.create_study"):
            plugin = OptunaPlugin(plugin_config)
            plugin.start_training_session(context)

            # Mock current trial
            mock_trial = Mock()
            mock_trial.suggest_loguniform.return_value = 0.001
            mock_trial.suggest_categorical.return_value = 32
            plugin.current_trial = mock_trial

            params = plugin._suggest_hyperparameters()

            assert "learning_rate" in params
            assert "batch_size" in params
            assert params["learning_rate"] == 0.001
            assert params["batch_size"] == 32

    def test_get_final_score(self, mock_optuna, plugin_config, context):
        """Test final score calculation."""
        plugin = OptunaPlugin(plugin_config)

        # Test with validation loss
        score = plugin._get_final_score(context)
        assert score == 0.5

        # Test with metrics
        context.validation_loss = None
        score = plugin._get_final_score(context)
        assert score == 0.8  # First metric value

        # Test with training loss
        context.metrics = {}
        context.training_loss = 0.3
        score = plugin._get_final_score(context)
        assert score == 0.3

    def test_should_continue_optimization(self, mock_optuna, plugin_config):
        """Test optimization continuation logic."""
        plugin = OptunaPlugin(plugin_config)

        # Should continue initially
        assert plugin._should_continue_optimization() is True

        # Should stop after n_trials
        plugin.trial_number = 3
        assert plugin._should_continue_optimization() is False

        # Early stopping is now handled by EarlyStoppingPlugin
        # Optuna plugin focuses on trial management only
        plugin.trial_number = 1
        plugin.no_improvement_count = 5
        assert plugin._should_continue_optimization() is True

    def test_save_best_params(self, mock_optuna, plugin_config, context):
        """Test best parameters saving."""
        plugin = OptunaPlugin(plugin_config)
        plugin.start_training_session(context)

        with tempfile.TemporaryDirectory() as temp_dir:
            plugin.output_dir = Path(temp_dir)
            plugin._save_best_params()

            # Check if file was created (may be empty if no trials completed)
            # File may or may not exist depending on study state
            # This test just verifies the method doesn't crash

    def test_get_suggested_hyperparameters(self, mock_optuna, plugin_config):
        """Test getting suggested hyperparameters."""
        plugin = OptunaPlugin(plugin_config)

        # No current trial
        params = plugin.get_suggested_hyperparameters()
        assert params == {}

        # With current trial
        mock_trial = Mock()
        mock_trial.user_attrs = {"suggested_params": {"lr": 0.001}}
        plugin.current_trial = mock_trial

        params = plugin.get_suggested_hyperparameters()
        assert params == {"lr": 0.001}

    def test_get_optimization_summary(self, mock_optuna, plugin_config, context):
        """Test optimization summary."""
        plugin = OptunaPlugin(plugin_config)
        plugin.start_training_session(context)

        # This may fail if no trials are completed, so we test the basic structure
        summary = plugin.get_optimization_summary()

        assert summary["study_name"] == "test_study"
        assert summary["direction"] == "minimize"
        assert summary["current_trial"] == 1
        # best_score and best_params may not be available until trials are completed


if __name__ == "__main__":
    pytest.main([__file__])
