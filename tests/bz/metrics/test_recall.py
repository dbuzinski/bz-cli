"""
Tests for the Recall metric.
"""

import torch
from bz.metrics import Recall


class TestRecall:
    """Test cases for Recall metric."""

    def test_recall_basic(self):
        """Test basic recall calculation."""
        metric = Recall()
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])

        metric.update(preds, targets)
        recall = metric.compute()

        # preds.argmax() gives [1, 0, 1], targets is [1, 0, 1]
        # TP=2, FN=0, so recall = 2/(2+0) = 1.0
        assert recall == 1.0

    def test_recall_with_false_negatives(self):
        """Test recall with false negatives."""
        metric = Recall()
        preds = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])

        metric.update(preds, targets)
        recall = metric.compute()

        # preds.argmax() gives [0, 0, 1], targets is [1, 0, 1]
        # TP=1, FN=1, so recall = 1/(1+1) = 0.5
        assert recall == 0.5
