"""
Tests for the early stopping plugin.
"""

import pytest

from bz.plugins.early_stopping import EarlyStoppingConfig, EarlyStoppingPlugin
from bz.plugins.plugin import PluginContext


class TestEarlyStoppingConfig:
    """Test EarlyStoppingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EarlyStoppingConfig()
        assert config.enabled is True
        assert config.patience == 10
        assert config.min_delta == 0.001
        assert config.monitor == "validation_loss"
        assert config.mode == "min"
        assert config.restore_best_weights is True
        assert config.verbose is True
        assert config.baseline is None
        assert config.min_epochs == 0
        assert config.strategy == "patience"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EarlyStoppingConfig(
            enabled=False,
            patience=5,
            min_delta=0.01,
            monitor="training_loss",
            mode="max",
            restore_best_weights=False,
            verbose=False,
            min_epochs=10,
            strategy="plateau",
        )
        assert config.enabled is False
        assert config.patience == 5
        assert config.min_delta == 0.01
        assert config.monitor == "training_loss"
        assert config.mode == "max"
        assert config.restore_best_weights is False
        assert config.verbose is False
        assert config.min_epochs == 10
        assert config.strategy == "plateau"


class TestEarlyStoppingPlugin:
    """Test EarlyStoppingPlugin functionality."""

    @pytest.fixture
    def plugin_config(self):
        """Create a test configuration."""
        return EarlyStoppingConfig(
            patience=3, min_delta=0.001, monitor="validation_loss", mode="min", strategy="patience"
        )

    @pytest.fixture
    def context(self):
        """Create a test context."""
        context = PluginContext()
        context.epoch = 0
        context.validation_loss_total = 0.0
        context.validation_batch_count = 0
        context.training_loss_total = 0.0
        context.training_batch_count = 0
        context.metrics = {}
        return context

    def test_plugin_initialization(self, plugin_config):
        """Test plugin initialization."""
        plugin = EarlyStoppingPlugin(plugin_config)
        assert plugin.config == plugin_config
        assert plugin.best_score is None
        assert plugin.patience_counter == 0
        assert plugin.best_epoch == 0
        assert plugin.best_weights is None

    def test_plugin_initialization_invalid_mode(self):
        """Test plugin initialization with invalid mode."""
        config = EarlyStoppingConfig(mode="invalid")
        with pytest.raises(ValueError, match="Mode must be 'min' or 'max'"):
            EarlyStoppingPlugin(config)

    def test_plugin_initialization_invalid_strategy(self):
        """Test plugin initialization with invalid strategy."""
        config = EarlyStoppingConfig(strategy="invalid")
        with pytest.raises(ValueError, match="Strategy must be 'patience', 'plateau', or 'custom'"):
            EarlyStoppingPlugin(config)

    def test_start_training_session(self, plugin_config, context):
        """Test training session start."""
        plugin = EarlyStoppingPlugin(plugin_config)
        plugin.start_training_session(context)

        # State should be reset
        assert plugin.best_score is None
        assert plugin.patience_counter == 0
        assert plugin.best_epoch == 0
        assert plugin.best_weights is None

    def test_start_training_session_disabled(self, context):
        """Test training session start when disabled."""
        config = EarlyStoppingConfig(enabled=False)
        plugin = EarlyStoppingPlugin(config)
        plugin.start_training_session(context)

        # Should not log anything when disabled
        # (we can't easily test logging, but we can verify no errors)

    def test_end_epoch_no_improvement(self, plugin_config, context):
        """Test end epoch with no improvement."""
        plugin = EarlyStoppingPlugin(plugin_config)
        plugin.start_training_session(context)

        # Set up context with validation loss
        context.epoch = 1
        context.validation_loss_total = 1.0
        context.validation_batch_count = 1

        # First call - should set best score
        plugin.end_epoch(context)
        assert plugin.best_score == 1.0
        assert plugin.patience_counter == 0
        assert not hasattr(context, "should_stop_training")

        # Second call - no improvement
        context.epoch = 2
        context.validation_loss_total = 1.1  # Worse score
        context.validation_batch_count = 1
        plugin.end_epoch(context)
        assert plugin.best_score == 1.0  # Should not change
        assert plugin.patience_counter == 1
        assert not hasattr(context, "should_stop_training")

        # Third call - still no improvement
        context.epoch = 3
        context.validation_loss_total = 1.2  # Even worse
        context.validation_batch_count = 1
        plugin.end_epoch(context)
        assert plugin.patience_counter == 2
        assert not hasattr(context, "should_stop_training")

        # Fourth call - should trigger early stopping
        context.epoch = 4
        context.validation_loss_total = 1.3
        context.validation_batch_count = 1
        plugin.end_epoch(context)
        assert plugin.patience_counter == 3
        assert hasattr(context, "should_stop_training")
        assert context.should_stop_training is True

    def test_end_epoch_with_improvement(self, plugin_config, context):
        """Test end epoch with improvement."""
        plugin = EarlyStoppingPlugin(plugin_config)
        plugin.start_training_session(context)

        # Set up context
        context.epoch = 1
        context.validation_loss_total = 1.0
        context.validation_batch_count = 1

        # First call
        plugin.end_epoch(context)
        assert plugin.best_score == 1.0
        assert plugin.patience_counter == 0

        # Second call - improvement
        context.epoch = 2
        context.validation_loss_total = 0.5  # Better score
        context.validation_batch_count = 1
        plugin.end_epoch(context)
        assert plugin.best_score == 0.5  # Should update
        assert plugin.patience_counter == 0  # Should reset
        assert not hasattr(context, "should_stop_training")

    def test_end_epoch_minimum_epochs(self, context):
        """Test that early stopping doesn't trigger before minimum epochs."""
        config = EarlyStoppingConfig(patience=1, min_epochs=5)
        plugin = EarlyStoppingPlugin(config)
        plugin.start_training_session(context)

        # Set up context
        context.epoch = 1
        context.validation_loss_total = 1.0
        context.validation_batch_count = 1

        # Should not trigger early stopping before min_epochs
        for epoch in range(1, 6):
            context.epoch = epoch
            context.validation_loss_total = 1.0 + epoch * 0.1  # Getting worse
            context.validation_batch_count = 1
            plugin.end_epoch(context)
            assert not hasattr(context, "should_stop_training")

        # Should trigger after min_epochs
        context.epoch = 6
        context.validation_loss_total = 1.6
        context.validation_batch_count = 1
        plugin.end_epoch(context)
        assert hasattr(context, "should_stop_training")
        assert context.should_stop_training is True

    def test_get_monitored_metric_validation_loss(self, plugin_config, context):
        """Test getting monitored metric for validation loss."""
        plugin = EarlyStoppingPlugin(plugin_config)

        context.validation_loss_total = 2.0
        context.validation_batch_count = 4

        metric = plugin._get_monitored_metric(context)
        assert metric == 0.5  # 2.0 / 4

    def test_get_monitored_metric_training_loss(self, context):
        """Test getting monitored metric for training loss."""
        config = EarlyStoppingConfig(monitor="training_loss")
        plugin = EarlyStoppingPlugin(config)

        context.training_loss_total = 3.0
        context.training_batch_count = 6

        metric = plugin._get_monitored_metric(context)
        assert metric == 0.5  # 3.0 / 6

    def test_get_monitored_metric_custom(self, context):
        """Test getting monitored metric for custom metrics."""
        config = EarlyStoppingConfig(monitor="accuracy")
        plugin = EarlyStoppingPlugin(config)

        context.metrics = {"accuracy": 0.85, "precision": 0.90}

        metric = plugin._get_monitored_metric(context)
        assert metric == 0.85

    def test_get_monitored_metric_not_found(self, plugin_config, context):
        """Test getting monitored metric when not found."""
        plugin = EarlyStoppingPlugin(plugin_config)

        # No validation loss
        context.validation_batch_count = 0
        context.metrics = {}

        metric = plugin._get_monitored_metric(context)
        assert metric is None

    def test_is_improvement_min_mode(self, plugin_config):
        """Test improvement detection in min mode."""
        plugin = EarlyStoppingPlugin(plugin_config)

        # First score should always be an improvement
        assert plugin._is_improvement(1.0) is True

        # Set best score
        plugin.best_score = 1.0

        # Better score (lower)
        assert plugin._is_improvement(0.5) is True

        # Worse score (higher)
        assert plugin._is_improvement(1.5) is False

        # Same score (within min_delta)
        assert plugin._is_improvement(1.0) is False

        # Slightly better (within min_delta)
        assert plugin._is_improvement(0.999) is False

        # Significantly better
        assert plugin._is_improvement(0.998) is True

    def test_is_improvement_max_mode(self):
        """Test improvement detection in max mode."""
        config = EarlyStoppingConfig(mode="max", min_delta=0.001)
        plugin = EarlyStoppingPlugin(config)

        # First score should always be an improvement
        assert plugin._is_improvement(1.0) is True

        # Set best score
        plugin.best_score = 1.0

        # Better score (higher)
        assert plugin._is_improvement(1.5) is True

        # Worse score (lower)
        assert plugin._is_improvement(0.5) is False

        # Same score (within min_delta)
        assert plugin._is_improvement(1.0) is False

        # Slightly better (within min_delta)
        assert plugin._is_improvement(1.001) is False

        # Significantly better
        assert plugin._is_improvement(1.002) is True

    def test_plateau_strategy(self, context):
        """Test plateau detection strategy."""
        config = EarlyStoppingConfig(strategy="plateau", plateau_patience=2, plateau_threshold=0.01)
        plugin = EarlyStoppingPlugin(config)
        plugin.start_training_session(context)

        # Set up context
        context.epoch = 1
        context.validation_loss_total = 1.0
        context.validation_batch_count = 1

        # First call - set best score
        plugin.end_epoch(context)
        assert plugin.best_score == 1.0

        # Second call - small change (plateau)
        context.epoch = 2
        context.validation_loss_total = 1.005  # Small change
        context.validation_batch_count = 1
        plugin.end_epoch(context)
        assert plugin.plateau_counter == 1

        # Third call - still plateau
        context.epoch = 3
        context.validation_loss_total = 1.008
        context.validation_batch_count = 1
        plugin.end_epoch(context)
        assert plugin.plateau_counter == 2

        # Fourth call - should trigger early stopping
        context.epoch = 4
        context.validation_loss_total = 1.009
        context.validation_batch_count = 1
        plugin.end_epoch(context)
        assert hasattr(context, "should_stop_training")
        assert context.should_stop_training is True

    def test_get_early_stopping_summary(self, plugin_config):
        """Test getting early stopping summary."""
        plugin = EarlyStoppingPlugin(plugin_config)

        summary = plugin.get_early_stopping_summary()

        assert summary["enabled"] is True
        assert summary["strategy"] == "patience"
        assert summary["monitor"] == "validation_loss"
        assert summary["mode"] == "min"
        assert summary["patience"] == 3
        assert summary["min_delta"] == 0.001
        assert summary["best_score"] is None
        assert summary["best_epoch"] == 0
        assert summary["patience_counter"] == 0
        assert summary["plateau_counter"] == 0

    def test_get_best_score(self, plugin_config, context):
        """Test getting best score."""
        plugin = EarlyStoppingPlugin(plugin_config)
        plugin.start_training_session(context)

        # Initially None
        assert plugin.get_best_score() is None

        # Set best score
        context.epoch = 1
        context.validation_loss_total = 1.0
        context.validation_batch_count = 1
        plugin.end_epoch(context)

        assert plugin.get_best_score() == 1.0

    def test_get_best_epoch(self, plugin_config, context):
        """Test getting best epoch."""
        plugin = EarlyStoppingPlugin(plugin_config)
        plugin.start_training_session(context)

        # Initially 0
        assert plugin.get_best_epoch() == 0

        # Set best score
        context.epoch = 5
        context.validation_loss_total = 1.0
        context.validation_batch_count = 1
        plugin.end_epoch(context)

        assert plugin.get_best_epoch() == 5

    def test_get_patience_counter(self, plugin_config, context):
        """Test getting patience counter."""
        plugin = EarlyStoppingPlugin(plugin_config)
        plugin.start_training_session(context)

        # Initially 0
        assert plugin.get_patience_counter() == 0

        # Set up context and trigger patience
        context.epoch = 1
        context.validation_loss_total = 1.0
        context.validation_batch_count = 1
        plugin.end_epoch(context)

        context.epoch = 2
        context.validation_loss_total = 1.1  # Worse
        context.validation_batch_count = 1
        plugin.end_epoch(context)

        assert plugin.get_patience_counter() == 1
