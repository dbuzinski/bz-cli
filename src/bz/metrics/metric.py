"""
Base metric class for the bz metrics system.
"""

from abc import ABC, abstractmethod
from typing import Optional
from torch import Tensor


class Metric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, name: Optional[str] = None):
        """
        Initialize metric.

        Args:
            name: Optional custom name for the metric. If None, uses class name.
        """
        self._name = name

    @abstractmethod
    def reset(self) -> None:
        """Reset the metric state."""
        pass

    @abstractmethod
    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Update metric with new predictions and targets.

        Args:
            preds: Model predictions
            targets: Ground truth targets
        """
        pass

    @abstractmethod
    def compute(self) -> float:
        """
        Compute the final metric value.

        Returns:
            Computed metric value
        """
        pass

    @property
    def name(self) -> str:
        """Get the metric name."""
        return self._name or self.__class__.__name__
