"""
Tests for the Accuracy metric.
"""

import torch
from bz.metrics import Accuracy


class TestAccuracy:
    """Test cases for Accuracy metric."""

    def test_accuracy_basic(self):
        """Test basic accuracy calculation."""
        metric = Accuracy()
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])

        metric.update(preds, targets)
        accuracy = metric.compute()

        # preds.argmax() gives [1, 0, 1], targets is [1, 0, 1]
        # All correct, so accuracy should be 1.0
        assert accuracy == 1.0

    def test_accuracy_partial(self):
        """Test accuracy with some incorrect predictions."""
        metric = Accuracy()
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 0])  # Last one is wrong

        metric.update(preds, targets)
        accuracy = metric.compute()

        # 2 out of 3 correct = 0.666...
        assert abs(accuracy - 2 / 3) < 1e-6

    def test_accuracy_reset(self):
        """Test that reset works correctly."""
        metric = Accuracy()
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1, 0])

        metric.update(preds, targets)
        assert metric.compute() == 1.0

        metric.reset()
        assert metric.compute() == 0.0

    def test_accuracy_binary(self):
        """Test accuracy with binary predictions."""
        metric = Accuracy()
        preds = torch.tensor([0.8, 0.3, 0.9, 0.1])
        targets = torch.tensor([1, 0, 1, 0])

        metric.update(preds, targets)
        accuracy = metric.compute()

        # All correct
        assert accuracy == 1.0
