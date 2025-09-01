"""
Tests for the F1Score metric.
"""

import torch
from bz.metrics import F1Score


class TestF1Score:
    """Test cases for F1Score metric."""

    def test_f1_score_perfect(self):
        """Test F1 score with perfect predictions."""
        metric = F1Score()
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])

        metric.update(preds, targets)
        f1 = metric.compute()

        # Perfect precision and recall = 1.0, so F1 = 1.0
        assert f1 == 1.0

    def test_f1_score_balanced(self):
        """Test F1 score with balanced precision and recall."""
        metric = F1Score()
        preds = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])

        metric.update(preds, targets)
        f1 = metric.compute()

        # precision = 2/3, recall = 2/2 = 1.0
        # F1 = 2 * (2/3 * 1.0) / (2/3 + 1.0) = 2 * (2/3) / (5/3) = 4/5 = 0.8
        expected_f1 = 2 * (2 / 3 * 1.0) / (2 / 3 + 1.0)
        assert abs(f1 - expected_f1) < 1e-6
