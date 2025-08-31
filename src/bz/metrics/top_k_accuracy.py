"""
Top-K Accuracy metric for the bz metrics system.
"""

from torch import Tensor
from typing import Optional

from .metric import Metric


class TopKAccuracy(Metric):
    """Top-K Accuracy metric for classification tasks."""

    def __init__(self, k: int = 5, name: Optional[str] = None):
        """
        Initialize Top-K accuracy metric.

        Args:
            k: Number of top predictions to consider
            name: Optional custom name
        """
        super().__init__(name)
        self.k = k
        self.correct: int = 0
        self.total: int = 0

    def reset(self) -> None:
        """Reset Top-K accuracy counters."""
        self.correct = 0
        self.total = 0

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Update Top-K accuracy with new predictions and targets.

        Args:
            preds: Model predictions (logits)
            targets: Ground truth labels
        """
        if preds.dim() == 1:
            raise ValueError("TopKAccuracy requires multi-class predictions")

        _, top_k_preds = preds.topk(self.k, dim=1)
        self.correct += top_k_preds.eq(targets.unsqueeze(1)).any(dim=1).sum().item()  # type: ignore
        self.total += targets.size(0)

    def compute(self) -> float:
        """
        Compute Top-K accuracy.

        Returns:
            Top-K accuracy value between 0 and 1
        """
        return self.correct / self.total if self.total > 0 else 0.0
