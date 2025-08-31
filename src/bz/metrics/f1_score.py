"""
F1 Score metric for the bz metrics system.
"""

from torch import Tensor
from typing import Optional

from .metric import Metric
from .precision import Precision
from .recall import Recall


class F1Score(Metric):
    """F1 Score metric for classification tasks."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.precision_metric = Precision()
        self.recall_metric = Recall()

    def reset(self) -> None:
        """Reset F1 score components."""
        self.precision_metric.reset()
        self.recall_metric.reset()

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Update F1 score with new predictions and targets.

        Args:
            preds: Model predictions (logits)
            targets: Ground truth labels
        """
        self.precision_metric.update(preds, targets)
        self.recall_metric.update(preds, targets)

    def compute(self) -> float:
        """
        Compute F1 score.

        Returns:
            F1 score value between 0 and 1
        """
        p = self.precision_metric.compute()
        r = self.recall_metric.compute()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
