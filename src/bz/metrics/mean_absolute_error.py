"""
Mean Absolute Error metric for the bz metrics system.
"""

import torch
from torch import Tensor
from typing import Optional

from .metric import Metric


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric for regression tasks."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.sum_absolute_error: float = 0.0
        self.total: int = 0

    def reset(self) -> None:
        """Reset MAE counters."""
        self.sum_absolute_error = 0.0
        self.total = 0

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Update MAE with new predictions and targets.

        Args:
            preds: Model predictions
            targets: Ground truth targets
        """
        self.sum_absolute_error += torch.abs(preds - targets).sum().item()  # type: ignore
        self.total += targets.numel()

    def compute(self) -> float:
        """
        Compute mean absolute error.

        Returns:
            MAE value (non-negative)
        """
        return self.sum_absolute_error / self.total if self.total > 0 else 0.0
