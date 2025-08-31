"""
Tests for the base Metric class.
"""

from bz.metrics import Metric


class TestMetricNames:
    """Test cases for metric naming."""

    def test_default_names(self):
        """Test default metric names."""

        # Create a simple test metric that inherits from Metric
        class TestMetric(Metric):
            def reset(self):
                pass

            def update(self, preds, targets):
                pass

            def compute(self):
                return 0.0

        accuracy = TestMetric()
        assert accuracy.name == "TestMetric"

    def test_custom_names(self):
        """Test custom metric names."""

        class TestMetric(Metric):
            def reset(self):
                pass

            def update(self, preds, targets):
                pass

            def compute(self):
                return 0.0

        accuracy = TestMetric(name="CustomAccuracy")
        assert accuracy.name == "CustomAccuracy"

        precision = TestMetric(name="CustomPrecision")
        assert precision.name == "CustomPrecision"
