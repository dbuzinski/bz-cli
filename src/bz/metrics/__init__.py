"""
Metrics system for bz CLI.
Provides extensible metrics for machine learning model evaluation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch
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


def _create_top5_accuracy(**kwargs):
    return TopKAccuracy(k=5, **kwargs)


# Metric registry for easy access
METRIC_REGISTRY = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "f1_score": F1Score,
    "mse": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "top5_accuracy": _create_top5_accuracy,
}


def get_metric(metric_name: str, **kwargs) -> Metric:
    """
    Get a metric by name.

    Args:
        metric_name: Name of the metric
        **kwargs: Additional arguments for the metric

    Returns:
        Metric instance

    Raises:
        ValueError: If metric name is not found
    """
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {list(METRIC_REGISTRY.keys())}")

    metric_class = METRIC_REGISTRY[metric_name]
    return metric_class(**kwargs)  # type: ignore


def list_available_metrics() -> List[str]:
    """
    List all available metrics.

    Returns:
        List of available metric names
    """
    return list(METRIC_REGISTRY.keys())
