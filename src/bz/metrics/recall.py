"""
Recall metric for the bz metrics system.
"""

from torch import Tensor
from typing import Optional

from .metric import Metric


class Recall(Metric):
    """Recall metric for classification tasks."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.true_positives: int = 0
        self.actual_positives: int = 0

    def reset(self) -> None:
        """Reset recall counters."""
        self.true_positives = 0
        self.actual_positives = 0

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Update recall with new predictions and targets.

        Args:
            preds: Model predictions (logits)
            targets: Ground truth labels
        """
        if preds.dim() > 1:
            pred_labels = preds.argmax(dim=1)
        else:
            pred_labels = (preds > 0.5).long()

        # For binary classification, consider class 1 as positive
        self.true_positives += ((pred_labels == 1) & (targets == 1)).sum().item()  # type: ignore
        self.actual_positives += (targets == 1).sum().item()  # type: ignore

    def compute(self) -> float:
        """
        Compute recall.

        Returns:
            Recall value between 0 and 1
        """
        if self.actual_positives == 0:
            return 0.0
        return self.true_positives / self.actual_positives
