"""
Tests for the metrics system.
"""

import pytest
import torch
from bz.metrics import (
    Accuracy, Precision, Recall, F1Score, MeanSquaredError, 
    MeanAbsoluteError, TopKAccuracy, get_metric, list_available_metrics
)


class TestAccuracy:
    """Test cases for Accuracy metric."""
    
    def test_accuracy_basic(self):
        """Test basic accuracy calculation."""
        metric = Accuracy()
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])
        
        metric.update(preds, targets)
        accuracy = metric.compute()
        
        # preds.argmax() gives [1, 0, 1], targets is [1, 0, 1]
        # All correct, so accuracy should be 1.0
        assert accuracy == 1.0
    
    def test_accuracy_partial(self):
        """Test accuracy with some incorrect predictions."""
        metric = Accuracy()
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 0])  # Last one is wrong
        
        metric.update(preds, targets)
        accuracy = metric.compute()
        
        # 2 out of 3 correct = 0.666...
        assert abs(accuracy - 2/3) < 1e-6
    
    def test_accuracy_reset(self):
        """Test that reset works correctly."""
        metric = Accuracy()
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1, 0])
        
        metric.update(preds, targets)
        assert metric.compute() == 1.0
        
        metric.reset()
        assert metric.compute() == 0.0
    
    def test_accuracy_binary(self):
        """Test accuracy with binary predictions."""
        metric = Accuracy()
        preds = torch.tensor([0.8, 0.3, 0.9, 0.1])
        targets = torch.tensor([1, 0, 1, 0])
        
        metric.update(preds, targets)
        accuracy = metric.compute()
        
        # All correct
        assert accuracy == 1.0


class TestPrecision:
    """Test cases for Precision metric."""
    
    def test_precision_basic(self):
        """Test basic precision calculation."""
        metric = Precision()
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])
        
        metric.update(preds, targets)
        precision = metric.compute()
        
        # preds.argmax() gives [1, 0, 1], targets is [1, 0, 1]
        # TP=2, FP=0, so precision = 2/(2+0) = 1.0
        assert precision == 1.0
    
    def test_precision_with_false_positives(self):
        """Test precision with false positives."""
        metric = Precision()
        preds = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])
        
        metric.update(preds, targets)
        precision = metric.compute()
        
        # preds.argmax() gives [1, 1, 1], targets is [1, 0, 1]
        # TP=2, FP=1, so precision = 2/(2+1) = 0.666...
        assert abs(precision - 2/3) < 1e-6
    
    def test_precision_no_positives(self):
        """Test precision when no positive predictions."""
        metric = Precision()
        preds = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
        targets = torch.tensor([0, 0])
        
        metric.update(preds, targets)
        precision = metric.compute()
        
        # No positive predictions, so precision should be 0
        assert precision == 0.0


class TestRecall:
    """Test cases for Recall metric."""
    
    def test_recall_basic(self):
        """Test basic recall calculation."""
        metric = Recall()
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])
        
        metric.update(preds, targets)
        recall = metric.compute()
        
        # preds.argmax() gives [1, 0, 1], targets is [1, 0, 1]
        # TP=2, FN=0, so recall = 2/(2+0) = 1.0
        assert recall == 1.0
    
    def test_recall_with_false_negatives(self):
        """Test recall with false negatives."""
        metric = Recall()
        preds = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])
        
        metric.update(preds, targets)
        recall = metric.compute()
        
        # preds.argmax() gives [0, 0, 1], targets is [1, 0, 1]
        # TP=1, FN=1, so recall = 1/(1+1) = 0.5
        assert recall == 0.5


class TestF1Score:
    """Test cases for F1Score metric."""
    
    def test_f1_score_perfect(self):
        """Test F1 score with perfect predictions."""
        metric = F1Score()
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])
        
        metric.update(preds, targets)
        f1 = metric.compute()
        
        # Perfect precision and recall, so F1 = 1.0
        assert f1 == 1.0
    
    def test_f1_score_balanced(self):
        """Test F1 score with balanced precision and recall."""
        metric = F1Score()
        preds = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1]])
        targets = torch.tensor([1, 0, 1])
        
        metric.update(preds, targets)
        f1 = metric.compute()
        
        # preds.argmax() gives [1, 1, 0], targets is [1, 0, 1]
        # TP=1, FP=1, FN=1
        # Precision = 1/(1+1) = 0.5, Recall = 1/(1+1) = 0.5
        # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        assert f1 == 0.5


class TestMeanSquaredError:
    """Test cases for MeanSquaredError metric."""
    
    def test_mse_basic(self):
        """Test basic MSE calculation."""
        metric = MeanSquaredError()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        
        metric.update(preds, targets)
        mse = metric.compute()
        
        # Perfect predictions, so MSE = 0
        assert mse == 0.0
    
    def test_mse_with_errors(self):
        """Test MSE with prediction errors."""
        metric = MeanSquaredError()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([2.0, 3.0, 4.0])
        
        metric.update(preds, targets)
        mse = metric.compute()
        
        # Errors: [1, 1, 1], squared: [1, 1, 1], mean = 1.0
        assert mse == 1.0


class TestMeanAbsoluteError:
    """Test cases for MeanAbsoluteError metric."""
    
    def test_mae_basic(self):
        """Test basic MAE calculation."""
        metric = MeanAbsoluteError()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        
        metric.update(preds, targets)
        mae = metric.compute()
        
        # Perfect predictions, so MAE = 0
        assert mae == 0.0
    
    def test_mae_with_errors(self):
        """Test MAE with prediction errors."""
        metric = MeanAbsoluteError()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([2.0, 3.0, 4.0])
        
        metric.update(preds, targets)
        mae = metric.compute()
        
        # Errors: [1, 1, 1], mean = 1.0
        assert mae == 1.0


class TestTopKAccuracy:
    """Test cases for TopKAccuracy metric."""
    
    def test_top5_accuracy_basic(self):
        """Test basic Top-5 accuracy calculation."""
        metric = TopKAccuracy(k=5)
        preds = torch.tensor([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        ])
        targets = torch.tensor([9, 0])  # Highest values in preds
        
        metric.update(preds, targets)
        accuracy = metric.compute()
        
        # Both targets are in top-5, so accuracy = 1.0
        assert accuracy == 1.0
    
    def test_top5_accuracy_partial(self):
        """Test Top-5 accuracy with some misses."""
        metric = TopKAccuracy(k=5)
        preds = torch.tensor([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ])
        targets = torch.tensor([9, 0])  # First in top-5, second not
        
        metric.update(preds, targets)
        accuracy = metric.compute()
        
        # 1 out of 2 correct = 0.5
        assert accuracy == 0.5
    
    def test_top5_accuracy_invalid_input(self):
        """Test TopKAccuracy with invalid input."""
        metric = TopKAccuracy(k=5)
        preds = torch.tensor([0.5, 0.3, 0.2])  # 1D tensor
        targets = torch.tensor([1])
        
        with pytest.raises(ValueError):
            metric.update(preds, targets)


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


class TestMetricNames:
    """Test cases for metric naming."""
    
    def test_default_names(self):
        """Test default metric names."""
        accuracy = Accuracy()
        assert accuracy.name == "Accuracy"
        
        precision = Precision()
        assert precision.name == "Precision"
    
    def test_custom_names(self):
        """Test custom metric names."""
        accuracy = Accuracy(name="CustomAccuracy")
        assert accuracy.name == "CustomAccuracy"
        
        precision = Precision(name="CustomPrecision")
        assert precision.name == "CustomPrecision"


if __name__ == "__main__":
    pytest.main([__file__])
