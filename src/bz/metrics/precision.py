"""
Precision metric for the bz metrics system.
"""

from torch import Tensor
from typing import Optional

from .metric import Metric


class Precision(Metric):
    """Precision metric for classification tasks."""

    def __init__(self, average: str = "micro", name: Optional[str] = None):
        """
        Initialize precision metric.

        Args:
            average: Averaging method ('micro', 'macro', 'weighted')
            name: Optional custom name
        """
        super().__init__(name)
        self.true_positives: int = 0
        self.predicted_positives: int = 0
        self.average = average

    def reset(self) -> None:
        """Reset precision counters."""
        self.true_positives = 0
        self.predicted_positives = 0

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Update precision with new predictions and targets.

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
        self.predicted_positives += (pred_labels == 1).sum().item()  # type: ignore

    def compute(self) -> float:
        """
        Compute precision.

        Returns:
            Precision value between 0 and 1
        """
        if self.predicted_positives == 0:
            return 0.0
        return self.true_positives / self.predicted_positives
