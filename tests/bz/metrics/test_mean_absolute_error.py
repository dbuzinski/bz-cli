"""
Tests for the MeanAbsoluteError metric.
"""

import torch
from bz.metrics import MeanAbsoluteError


class TestMeanAbsoluteError:
    """Test cases for MeanAbsoluteError metric."""

    def test_mae_basic(self):
        """Test basic MAE calculation."""
        metric = MeanAbsoluteError()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        metric.update(preds, targets)
        mae = metric.compute()

        # Perfect predictions, so MAE = 0
        assert mae == 0.0

    def test_mae_with_errors(self):
        """Test MAE with prediction errors."""
        metric = MeanAbsoluteError()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([0.0, 2.0, 6.0])

        metric.update(preds, targets)
        mae = metric.compute()

        # Errors: |1-0|, |2-2|, |3-6| = 1, 0, 3
        # Mean: (1 + 0 + 3) / 3 = 4/3 ≈ 1.333...
        expected_mae = (1 + 0 + 3) / 3
        assert abs(mae - expected_mae) < 1e-6
