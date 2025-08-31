"""
Metrics system for bz CLI.
Provides extensible metrics for machine learning model evaluation.
"""

from typing import List

# Import base class
from .metric import Metric

# Import individual metrics
from .accuracy import Accuracy
from .precision import Precision
from .recall import Recall
from .f1_score import F1Score
from .top_k_accuracy import TopKAccuracy
from .mean_squared_error import MeanSquaredError
from .mean_absolute_error import MeanAbsoluteError


def _create_top5_accuracy(**kwargs):
    """Factory function for Top-5 accuracy metric."""
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


# Export all metrics for backward compatibility
__all__ = [
    "Metric",
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "TopKAccuracy",
    "MeanSquaredError",
    "MeanAbsoluteError",
    "METRIC_REGISTRY",
    "get_metric",
    "list_available_metrics",
]
