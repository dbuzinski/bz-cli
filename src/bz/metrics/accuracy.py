"""
Accuracy metric for the bz metrics system.
"""

from torch import Tensor
from typing import Optional

from .metric import Metric


class Accuracy(Metric):
    """Accuracy metric for classification tasks."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.correct: int = 0
        self.total: int = 0

    def reset(self) -> None:
        """Reset accuracy counters."""
        self.correct = 0
        self.total = 0

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Update accuracy with new predictions and targets.

        Args:
            preds: Model predictions (logits)
            targets: Ground truth labels
        """
        if preds.dim() > 1:
            predicted_labels = preds.argmax(dim=1)
        else:
            predicted_labels = (preds > 0.5).long()

        self.correct += (predicted_labels == targets).sum().item()  # type: ignore
        self.total += targets.size(0)

    def compute(self) -> float:
        """
        Compute accuracy.

        Returns:
            Accuracy value between 0 and 1
        """
        return self.correct / self.total if self.total > 0 else 0.0
