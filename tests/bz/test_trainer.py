"""
Tests for bz Trainer functionality.
"""

import pytest
import tempfile
import os
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from bz import Trainer, CheckpointManager, TrainingLoop
from bz.plugins import PluginContext, Plugin


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class TestCheckpointManager:
    """Test CheckpointManager functionality."""

    def test_checkpoint_manager_initialization(self):
        """Test CheckpointManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, "checkpoints")
            manager = CheckpointManager(checkpoint_dir)

            assert manager.checkpoint_dir == checkpoint_dir
            assert os.path.exists(checkpoint_dir)

    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, "checkpoints")
            manager = CheckpointManager(checkpoint_dir)

            model = SimpleModel()
            optimizer = torch.optim.Adam(model.parameters())
            loss_fn = nn.CrossEntropyLoss()
            training_loader = Mock()
            # Don't mock generator to avoid pickle issues
            training_loader.generator = None

            checkpoint_path = manager.save_checkpoint(1, model, optimizer, loss_fn, training_loader, "cpu")

            assert os.path.exists(checkpoint_path)
            assert checkpoint_path.endswith("model_epoch1.pth")

    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, "checkpoints")
            manager = CheckpointManager(checkpoint_dir)

            model = SimpleModel()
            optimizer = torch.optim.Adam(model.parameters())
            loss_fn = nn.CrossEntropyLoss()
            training_loader = Mock()
            training_loader.generator = None

            # Save checkpoint
            manager.save_checkpoint(5, model, optimizer, loss_fn, training_loader, "cpu")

            # Load checkpoint
            new_model = SimpleModel()
            new_optimizer = torch.optim.Adam(new_model.parameters())
            new_loss_fn = nn.CrossEntropyLoss()

            epoch = manager.load_latest_checkpoint(new_model, new_optimizer, new_loss_fn, training_loader, "cpu")

            assert epoch == 5

    def test_load_checkpoint_nonexistent(self):
        """Test loading checkpoint when none exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, "checkpoints")
            manager = CheckpointManager(checkpoint_dir)

            model = SimpleModel()
            optimizer = torch.optim.Adam(model.parameters())
            loss_fn = nn.CrossEntropyLoss()
            training_loader = Mock()

            epoch = manager.load_latest_checkpoint(model, optimizer, loss_fn, training_loader, "cpu")

            assert epoch is None

    def test_find_latest_checkpoint_epoch(self):
        """Test finding latest checkpoint epoch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, "checkpoints")
            manager = CheckpointManager(checkpoint_dir)

            # Create some checkpoint files
            checkpoint_files = ["model_epoch1.pth", "model_epoch5.pth", "model_epoch3.pth"]
            for filename in checkpoint_files:
                with open(os.path.join(checkpoint_dir, filename), "w") as f:
                    f.write("dummy content")

            latest_epoch = manager._find_latest_checkpoint_epoch()
            assert latest_epoch == 5


class TestTrainingLoop:
    """Test TrainingLoop functionality."""

    def test_training_loop_initialization(self):
        """Test TrainingLoop initialization."""
        trainer = Trainer()
        training_loop = TrainingLoop(trainer)

        assert training_loop.trainer == trainer

    def test_run_training_epoch(self):
        """Test running training epoch."""
        trainer = Trainer()
        training_loop = TrainingLoop(trainer)

        # Mock components
        model = Mock()
        # Mock model to return tensors that require gradients
        mock_output = torch.randn(2, 2, requires_grad=True)
        model.return_value = mock_output
        optimizer = Mock()
        loss_fn = Mock()
        # Mock loss to return a tensor that requires gradients
        loss_fn.return_value = torch.tensor(0.5, requires_grad=True)
        training_loader = [(torch.randn(2, 10), torch.randint(0, 2, (2,)))]
        device = "cpu"
        metrics = []

        context = PluginContext()

        # Mock trainer's _run_stage method
        trainer._run_stage = Mock()

        training_loop.run_training_epoch(context, model, optimizer, loss_fn, training_loader, device, metrics)

        # Verify that training stages were called
        trainer._run_stage.assert_any_call("start_training_loop", context)
        trainer._run_stage.assert_any_call("start_training_batch", context)
        trainer._run_stage.assert_any_call("end_training_batch", context)
        trainer._run_stage.assert_any_call("end_training_loop", context)

    def test_run_validation_epoch(self):
        """Test running validation epoch."""
        trainer = Trainer()
        training_loop = TrainingLoop(trainer)

        # Mock components
        model = Mock()
        loss_fn = Mock()
        loss_fn.return_value = torch.tensor(0.5)
        validation_loader = [(torch.randn(2, 10), torch.randint(0, 2, (2,)))]
        device = "cpu"
        metrics = []

        context = PluginContext()

        # Mock trainer's _run_stage method
        trainer._run_stage = Mock()

        training_loop.run_validation_epoch(context, model, loss_fn, validation_loader, device, metrics)

        # Verify that validation stages were called
        trainer._run_stage.assert_any_call("start_validation_loop", context)
        trainer._run_stage.assert_any_call("start_validation_batch", context)
        trainer._run_stage.assert_any_call("end_validation_batch", context)
        trainer._run_stage.assert_any_call("end_validation_loop", context)

    def test_run_validation_epoch_no_loader(self):
        """Test running validation epoch with no validation loader."""
        trainer = Trainer()
        training_loop = TrainingLoop(trainer)

        # Mock components
        model = Mock()
        loss_fn = Mock()
        validation_loader = None
        device = "cpu"
        metrics = []

        context = PluginContext()

        # Mock trainer's _run_stage method
        trainer._run_stage = Mock()

        training_loop.run_validation_epoch(context, model, loss_fn, validation_loader, device, metrics)

        # Should not call any validation stages
        trainer._run_stage.assert_not_called()


class TestTrainer:
    """Test Trainer functionality."""

    def test_trainer_initialization(self):
        """Test Trainer initialization."""
        trainer = Trainer()

        assert trainer.plugins == []
        assert trainer.logger is not None
        assert trainer.training_loop is not None

    def test_add_plugin(self):
        """Test adding plugin to trainer."""
        trainer = Trainer()
        plugin = Mock(spec=Plugin)

        trainer.add_plugin(plugin)

        assert len(trainer.plugins) == 1
        assert trainer.plugins[0] == plugin

    def test_run_stage(self):
        """Test running training stage."""
        trainer = Trainer()

        # Create mock plugins
        plugin1 = Mock(spec=Plugin)
        plugin2 = Mock(spec=Plugin)
        trainer.plugins = [plugin1, plugin2]

        context = PluginContext()

        trainer._run_stage("start_training_session", context)

        # Verify both plugins were called
        plugin1.run_stage.assert_called_with("start_training_session", context)
        plugin2.run_stage.assert_called_with("start_training_session", context)

    def test_run_stage_plugin_error(self):
        """Test running stage with plugin error."""
        trainer = Trainer()

        # Create mock plugins with proper name attribute
        plugin1 = Mock(spec=Plugin)
        plugin1.name = "test_plugin"
        plugin1.run_stage.side_effect = Exception("Plugin error")
        plugin2 = Mock(spec=Plugin)
        plugin2.name = "test_plugin2"
        trainer.plugins = [plugin1, plugin2]

        context = PluginContext()

        # Should not raise exception, should continue with other plugins
        trainer._run_stage("start_training_session", context)

        # Verify both plugins were called
        plugin1.run_stage.assert_called_with("start_training_session", context)
        plugin2.run_stage.assert_called_with("start_training_session", context)

    def test_compute_training_signature(self):
        """Test computing training signature."""
        trainer = Trainer()

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        config = {"lr": 0.001, "batch_size": 32}

        signature = trainer._compute_training_signature(model, optimizer, loss_fn, config)

        assert isinstance(signature, str)
        assert len(signature) == 16  # SHA256 hex digest truncated to 16 chars

    @patch("bz.Trainer._run_stage")
    def test_train_method(self, mock_run_stage):
        """Test trainer train method."""
        trainer = Trainer()

        # Mock components
        model = Mock()
        optimizer = Mock()
        loss_fn = Mock()
        training_loader = Mock()
        validation_loader = Mock()

        # Mock training loop methods
        trainer.training_loop.run_training_epoch = Mock()
        trainer.training_loop.run_validation_epoch = Mock()

        # Mock checkpoint manager
        with patch("bz.CheckpointManager") as mock_checkpoint_manager_class:
            mock_checkpoint_manager = Mock()
            mock_checkpoint_manager_class.return_value = mock_checkpoint_manager
            mock_checkpoint_manager.load_latest_checkpoint.return_value = None

            # Create a TrainingConfiguration object
            from bz import TrainingConfiguration

            config = TrainingConfiguration(
                epochs=2,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                training_loader=training_loader,
                validation_loader=validation_loader,
                compile=False,
                checkpoint_interval=0,
                metrics=[],
                hyperparameters={},
            )

            trainer.train(config)

            # Verify training stages were called
            mock_run_stage.assert_any_call("start_training_session", mock_run_stage.call_args[0][1])
            mock_run_stage.assert_any_call("end_training_session", mock_run_stage.call_args[0][1])

            # Verify training loop was called for each epoch
            assert trainer.training_loop.run_training_epoch.call_count == 2
            assert trainer.training_loop.run_validation_epoch.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])
