"""
Mean Squared Error metric for the bz metrics system.
"""

from torch import Tensor
from typing import Optional

from .metric import Metric


class MeanSquaredError(Metric):
    """Mean Squared Error metric for regression tasks."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.sum_squared_error: float = 0.0
        self.total: int = 0

    def reset(self) -> None:
        """Reset MSE counters."""
        self.sum_squared_error = 0.0
        self.total = 0

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Update MSE with new predictions and targets.

        Args:
            preds: Model predictions
            targets: Ground truth targets
        """
        self.sum_squared_error += ((preds - targets) ** 2).sum().item()  # type: ignore
        self.total += targets.numel()

    def compute(self) -> float:
        """
        Compute mean squared error.

        Returns:
            MSE value (non-negative)
        """
        return self.sum_squared_error / self.total if self.total > 0 else 0.0
