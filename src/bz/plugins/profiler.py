"""
Performance profiling plugin for bz.
Provides detailed performance monitoring and profiling during training.
"""

import time
import psutil
import torch
from typing import Optional
from dataclasses import dataclass
from bz.plugins import Plugin, PluginContext


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    epoch: int
    batch: int
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float]
    training_time: float
    validation_time: float
    total_time: float
    throughput_samples_per_sec: float


class ProfilerPlugin(Plugin):
    """Plugin for performance profiling and monitoring."""

    def __init__(self, log_interval: int = 10, enable_gpu_monitoring: bool = True, **kwargs):
        """
        Initialize profiler plugin.

        Args:
            log_interval: How often to log performance metrics (in batches)
            enable_gpu_monitoring: Whether to monitor GPU usage
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(**kwargs)
        self.log_interval = log_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.performance_history: list[PerformanceMetrics] = []
        self.epoch_start_time: Optional[float] = None
        self.training_start_time: Optional[float] = None
        self.batch_start_time: Optional[float] = None
        self.training_time = 0.0
        self.validation_time = 0.0

    def _get_gpu_memory(self) -> Optional[float]:
        """Get GPU memory usage in MB."""
        if not self.enable_gpu_monitoring or not torch.cuda.is_available():
            return None

        try:
            return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        except Exception:
            return None

    def _record_metrics(self, context: PluginContext, stage: str = "batch") -> None:
        """Record performance metrics."""
        if context.training_batch_count % self.log_interval != 0:
            return

        current_time = time.time()
        process = psutil.Process()

        metrics = PerformanceMetrics(
            epoch=context.epoch,
            batch=context.training_batch_count,
            timestamp=current_time,
            cpu_percent=process.cpu_percent(),
            memory_mb=process.memory_info().rss / 1024 / 1024,  # Convert to MB
            gpu_memory_mb=self._get_gpu_memory(),
            training_time=self.training_time,
            validation_time=self.validation_time,
            total_time=current_time - (self.training_start_time or current_time),
            throughput_samples_per_sec=0.0,  # Will be calculated later
        )

        self.performance_history.append(metrics)

        # Log to console
        self.logger.info(
            f"Performance [{stage}] - "
            f"CPU: {metrics.cpu_percent:.1f}%, "
            f"Memory: {metrics.memory_mb:.1f}MB"
            + (f", GPU: {metrics.gpu_memory_mb:.1f}MB" if metrics.gpu_memory_mb else "")
        )

    def start_training_session(self, context: PluginContext) -> None:
        """Initialize profiling session."""
        self.training_start_time = time.time()
        self.performance_history.clear()
        self.logger.info("Performance profiling started")

    def start_epoch(self, context: PluginContext) -> None:
        """Start epoch timing."""
        self.epoch_start_time = time.time()
        self.training_time = 0.0
        self.validation_time = 0.0

    def start_training_batch(self, context: PluginContext) -> None:
        """Start batch timing."""
        self.batch_start_time = time.time()

    def end_training_batch(self, context: PluginContext) -> None:
        """End batch timing and record metrics."""
        if self.batch_start_time:
            self.training_time += time.time() - self.batch_start_time
            self._record_metrics(context, "training_batch")

    def start_validation_loop(self, context: PluginContext) -> None:
        """Start validation timing."""
        self.validation_start_time = time.time()

    def end_validation_loop(self, context: PluginContext) -> None:
        """End validation timing."""
        if hasattr(self, "validation_start_time"):
            self.validation_time = time.time() - self.validation_start_time

    def end_epoch(self, context: PluginContext) -> None:
        """End epoch and record epoch-level metrics."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.logger.info(
                f"Epoch {context.epoch} Performance - "
                f"Training: {self.training_time:.2f}s, "
                f"Validation: {self.validation_time:.2f}s, "
                f"Total: {epoch_time:.2f}s"
            )

    def end_training_session(self, context: PluginContext) -> None:
        """Generate performance summary."""
        if not self.performance_history:
            return

        total_time = time.time() - (self.training_start_time or time.time())

        # Calculate summary statistics
        cpu_avg = sum(m.cpu_percent for m in self.performance_history) / len(self.performance_history)
        memory_avg = sum(m.memory_mb for m in self.performance_history) / len(self.performance_history)
        memory_max = max(m.memory_mb for m in self.performance_history)

        gpu_metrics = [m.gpu_memory_mb for m in self.performance_history if m.gpu_memory_mb is not None]
        gpu_avg = sum(gpu_metrics) / len(gpu_metrics) if gpu_metrics else None
        gpu_max = max(gpu_metrics) if gpu_metrics else None

        self.logger.info("=" * 60)
        self.logger.info("PERFORMANCE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Training Time: {total_time:.2f}s")
        self.logger.info(f"Average CPU Usage: {cpu_avg:.1f}%")
        self.logger.info(f"Average Memory Usage: {memory_avg:.1f}MB")
        self.logger.info(f"Peak Memory Usage: {memory_max:.1f}MB")

        if gpu_avg:
            self.logger.info(f"Average GPU Memory: {gpu_avg:.1f}MB")
            self.logger.info(f"Peak GPU Memory: {gpu_max:.1f}MB")

        # Save detailed metrics to file
        self._save_performance_report()

    def _save_performance_report(self) -> None:
        """Save detailed performance report to file."""
        try:
            import json
            from datetime import datetime

            report = {
                "timestamp": datetime.now().isoformat(),
                "metrics": [
                    {
                        "epoch": m.epoch,
                        "batch": m.batch,
                        "timestamp": m.timestamp,
                        "cpu_percent": m.cpu_percent,
                        "memory_mb": m.memory_mb,
                        "gpu_memory_mb": m.gpu_memory_mb,
                        "training_time": m.training_time,
                        "validation_time": m.validation_time,
                        "total_time": m.total_time,
                    }
                    for m in self.performance_history
                ],
            }

            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Detailed performance report saved to: {filename}")

        except Exception as e:
            self.logger.error(f"Failed to save performance report: {e}")

    @staticmethod
    def init(spec, log_interval: int = 10, enable_gpu_monitoring: bool = True) -> "ProfilerPlugin":
        """
        Create a ProfilerPlugin instance.

        Args:
            spec: Training specification
            log_interval: How often to log performance metrics
            enable_gpu_monitoring: Whether to monitor GPU usage

        Returns:
            Configured ProfilerPlugin instance
        """
        return ProfilerPlugin(log_interval=log_interval, enable_gpu_monitoring=enable_gpu_monitoring)
