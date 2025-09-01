"""
Tests for the metric registry functionality.
"""

import pytest
from bz.metrics import (
    Accuracy,
    Precision,
    MeanSquaredError,
    TopKAccuracy,
    get_metric,
    list_available_metrics,
)


class TestMetricRegistry:
    """Test cases for metric registry functionality."""

    def test_get_metric(self):
        """Test getting metrics by name."""
        accuracy = get_metric("accuracy")
        assert isinstance(accuracy, Accuracy)

        precision = get_metric("precision")
        assert isinstance(precision, Precision)

        mse = get_metric("mse")
        assert isinstance(mse, MeanSquaredError)

    def test_get_metric_invalid(self):
        """Test getting invalid metric name."""
        with pytest.raises(ValueError):
            get_metric("invalid_metric")

    def test_list_available_metrics(self):
        """Test listing available metrics."""
        metrics = list_available_metrics()
        expected = ["accuracy", "precision", "recall", "f1_score", "mse", "mae", "top5_accuracy"]

        for metric in expected:
            assert metric in metrics

    def test_get_metric_with_kwargs(self):
        """Test getting metric with additional arguments."""
        top5 = get_metric("top5_accuracy")
        assert isinstance(top5, TopKAccuracy)
        assert top5.k == 5
