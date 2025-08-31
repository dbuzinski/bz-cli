"""
Tests for the TopKAccuracy metric.
"""

import pytest
import torch
from bz.metrics import TopKAccuracy


class TestTopKAccuracy:
    """Test cases for TopKAccuracy metric."""

    def test_top5_accuracy_basic(self):
        """Test basic Top-5 accuracy calculation."""
        metric = TopKAccuracy(k=5)
        preds = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            ]
        )
        targets = torch.tensor([9, 0])  # Highest values in preds

        metric.update(preds, targets)
        accuracy = metric.compute()

        # Both targets are in top-5, so accuracy = 1.0
        assert accuracy == 1.0

    def test_top5_accuracy_partial(self):
        """Test Top-5 accuracy with some misses."""
        metric = TopKAccuracy(k=5)
        preds = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ]
        )
        targets = torch.tensor([9, 0])  # First in top-5, second not

        metric.update(preds, targets)
        accuracy = metric.compute()

        # 1 out of 2 correct = 0.5
        assert accuracy == 0.5

    def test_top5_accuracy_invalid_input(self):
        """Test TopKAccuracy with invalid input."""
        metric = TopKAccuracy(k=5)
        preds = torch.tensor([0.5, 0.3, 0.2])  # 1D tensor
        targets = torch.tensor([1])

        with pytest.raises(ValueError):
            metric.update(preds, targets)
