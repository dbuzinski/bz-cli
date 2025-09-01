"""
Tests for the Precision metric.
"""

import torch
from bz.metrics import Precision


class TestPrecision:
    """Test cases for Precision metric."""

    def test_precision_basic(self):
        """Test basic precision calculation."""
        metric = Precision()
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])

        metric.update(preds, targets)
        precision = metric.compute()

        # preds.argmax() gives [1, 0, 1], targets is [1, 0, 1]
        # TP=2, FP=0, so precision = 2/(2+0) = 1.0
        assert precision == 1.0

    def test_precision_with_false_positives(self):
        """Test precision with false positives."""
        metric = Precision()
        preds = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])

        metric.update(preds, targets)
        precision = metric.compute()

        # preds.argmax() gives [1, 1, 1], targets is [1, 0, 1]
        # TP=2, FP=1, so precision = 2/(2+1) = 0.666...
        assert abs(precision - 2 / 3) < 1e-6

    def test_precision_no_positives(self):
        """Test precision when no positive predictions."""
        metric = Precision()
        preds = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
        targets = torch.tensor([1, 1])

        metric.update(preds, targets)
        precision = metric.compute()

        # preds.argmax() gives [0, 0], targets is [1, 1]
        # TP=0, FP=0, so precision = 0 (by convention)
        assert precision == 0.0
