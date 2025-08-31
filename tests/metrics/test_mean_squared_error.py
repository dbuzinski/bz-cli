"""
Tests for the MeanSquaredError metric.
"""

import torch
from bz.metrics import MeanSquaredError


class TestMeanSquaredError:
    """Test cases for MeanSquaredError metric."""

    def test_mse_basic(self):
        """Test basic MSE calculation."""
        metric = MeanSquaredError()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        metric.update(preds, targets)
        mse = metric.compute()

        # Perfect predictions, so MSE = 0
        assert mse == 0.0

    def test_mse_with_errors(self):
        """Test MSE with prediction errors."""
        metric = MeanSquaredError()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([0.0, 2.0, 6.0])

        metric.update(preds, targets)
        mse = metric.compute()

        # Errors: (1-0)², (2-2)², (3-6)² = 1, 0, 9
        # Mean: (1 + 0 + 9) / 3 = 10/3 ≈ 3.333...
        expected_mse = (1 + 0 + 9) / 3
        assert abs(mse - expected_mse) < 1e-6
