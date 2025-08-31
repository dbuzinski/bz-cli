"""
Health monitoring and system checks for bz.
Provides system health status and resource monitoring.
"""

import os
import sys
import psutil
import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class HealthStatus(Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class SystemResource:
    """System resource information."""

    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float


@dataclass
class HealthCheck:
    """Health check result."""

    name: str
    status: HealthStatus
    message: str
    details: Optional[Dict[str, Any]] = None


class HealthMonitor:
    """System health monitoring and checks."""

    def __init__(self):
        self.checks: List[HealthCheck] = []

    def check_system_resources(self) -> List[SystemResource]:
        """Check system resource usage."""
        resources = []

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_status = self._get_status(cpu_percent, 80, 95)
        resources.append(
            SystemResource(
                name="CPU Usage",
                value=cpu_percent,
                unit="%",
                status=cpu_status,
                threshold_warning=80,
                threshold_critical=95,
            )
        )

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_status = self._get_status(memory_percent, 85, 95)
        resources.append(
            SystemResource(
                name="Memory Usage",
                value=memory_percent,
                unit="%",
                status=memory_status,
                threshold_warning=85,
                threshold_critical=95,
            )
        )

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100
        disk_status = self._get_status(disk_percent, 85, 95)
        resources.append(
            SystemResource(
                name="Disk Usage",
                value=disk_percent,
                unit="%",
                status=disk_status,
                threshold_warning=85,
                threshold_critical=95,
            )
        )

        # GPU memory (if available)
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                gpu_percent = gpu_memory * 100
                gpu_status = self._get_status(gpu_percent, 85, 95)
                resources.append(
                    SystemResource(
                        name="GPU Memory Usage",
                        value=gpu_percent,
                        unit="%",
                        status=gpu_status,
                        threshold_warning=85,
                        threshold_critical=95,
                    )
                )
            except Exception:
                pass

        return resources

    def check_dependencies(self) -> List[HealthCheck]:
        """Check if required dependencies are available."""
        checks = []

        # Check PyTorch
        try:
            torch_version = torch.__version__
            checks.append(
                HealthCheck(
                    name="PyTorch",
                    status=HealthStatus.HEALTHY,
                    message=f"PyTorch {torch_version} is available",
                    details={"version": torch_version},
                )
            )
        except Exception as e:
            checks.append(
                HealthCheck(name="PyTorch", status=HealthStatus.CRITICAL, message=f"PyTorch is not available: {e}")
            )

        # Check CUDA
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            checks.append(
                HealthCheck(
                    name="CUDA",
                    status=HealthStatus.HEALTHY,
                    message=f"CUDA {cuda_version} is available with {device_count} device(s)",
                    details={"version": cuda_version, "device_count": device_count},
                )
            )
        else:
            checks.append(
                HealthCheck(name="CUDA", status=HealthStatus.WARNING, message="CUDA is not available (CPU-only mode)")
            )

        # Check psutil
        try:
            psutil_version = psutil.__version__
            checks.append(
                HealthCheck(
                    name="psutil",
                    status=HealthStatus.HEALTHY,
                    message=f"psutil {psutil_version} is available",
                    details={"version": psutil_version},
                )
            )
        except Exception as e:
            checks.append(
                HealthCheck(name="psutil", status=HealthStatus.CRITICAL, message=f"psutil is not available: {e}")
            )

        # Check optional dependencies
        optional_deps = {
            "tensorboard": "TensorBoard logging",
            "wandb": "Weights & Biases integration",
            "tqdm": "Progress bars",
        }

        for dep, description in optional_deps.items():
            try:
                module = __import__(dep)
                version = getattr(module, "__version__", "unknown")
                checks.append(
                    HealthCheck(
                        name=dep,
                        status=HealthStatus.HEALTHY,
                        message=f"{description} is available",
                        details={"version": version},
                    )
                )
            except ImportError:
                checks.append(
                    HealthCheck(
                        name=dep, status=HealthStatus.WARNING, message=f"{description} is not available (optional)"
                    )
                )

        return checks

    def check_environment(self) -> List[HealthCheck]:
        """Check environment configuration."""
        checks = []

        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            checks.append(
                HealthCheck(
                    name="Python Version",
                    status=HealthStatus.HEALTHY,
                    message=f"Python {python_version.major}.{python_version.minor}.{python_version.micro}",
                    details={"version": f"{python_version.major}.{python_version.minor}.{python_version.micro}"},
                )
            )
        else:
            checks.append(
                HealthCheck(
                    name="Python Version",
                    status=HealthStatus.CRITICAL,
                    message=f"Python {python_version.major}.{python_version.minor}.{python_version.micro} is too old (requires 3.8+)",
                )
            )

        # Check working directory
        try:
            cwd = os.getcwd()
            checks.append(
                HealthCheck(
                    name="Working Directory",
                    status=HealthStatus.HEALTHY,
                    message=f"Working directory: {cwd}",
                    details={"path": cwd},
                )
            )
        except Exception as e:
            checks.append(
                HealthCheck(
                    name="Working Directory",
                    status=HealthStatus.CRITICAL,
                    message=f"Cannot access working directory: {e}",
                )
            )

        # Check environment variables
        env_vars = ["BZ_ENV", "BZ_CONFIG"]
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                checks.append(
                    HealthCheck(
                        name=f"Environment Variable {var}",
                        status=HealthStatus.HEALTHY,
                        message=f"{var}={value}",
                        details={"value": value},
                    )
                )
            else:
                checks.append(
                    HealthCheck(
                        name=f"Environment Variable {var}",
                        status=HealthStatus.WARNING,
                        message=f"{var} is not set (using defaults)",
                    )
                )

        return checks

    def run_full_health_check(self) -> Dict[str, Any]:
        """Run a complete health check."""
        resources = self.check_system_resources()
        dependencies = self.check_dependencies()
        environment = self.check_environment()

        all_checks = dependencies + environment

        # Determine overall status
        critical_count = sum(1 for check in all_checks if check.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for check in all_checks if check.status == HealthStatus.WARNING)

        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_count > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY

        return {
            "status": overall_status,
            "timestamp": self._get_timestamp(),
            "resources": resources,
            "dependencies": dependencies,
            "environment": environment,
            "summary": {
                "total_checks": len(all_checks),
                "healthy": sum(1 for check in all_checks if check.status == HealthStatus.HEALTHY),
                "warnings": warning_count,
                "critical": critical_count,
            },
        }

    def _get_status(self, value: float, warning_threshold: float, critical_threshold: float) -> HealthStatus:
        """Get health status based on thresholds."""
        if value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()

    def print_health_report(self, health_data: Dict[str, Any]) -> None:
        """Print a formatted health report."""
        status = health_data["status"]
        summary = health_data["summary"]

        print("=" * 60)
        print("BZ HEALTH CHECK REPORT")
        print("=" * 60)
        print(f"Overall Status: {status.value.upper()}")
        print(f"Timestamp: {health_data['timestamp']}")
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Healthy: {summary['healthy']}, Warnings: {summary['warnings']}, Critical: {summary['critical']}")
        print()

        # System Resources
        print("SYSTEM RESOURCES:")
        print("-" * 30)
        for resource in health_data["resources"]:
            status_icon = (
                "🟢"
                if resource.status == HealthStatus.HEALTHY
                else "🟡" if resource.status == HealthStatus.WARNING else "🔴"
            )
            print(f"{status_icon} {resource.name}: {resource.value:.1f}{resource.unit}")
        print()

        # Dependencies
        print("DEPENDENCIES:")
        print("-" * 30)
        for check in health_data["dependencies"]:
            status_icon = (
                "🟢" if check.status == HealthStatus.HEALTHY else "🟡" if check.status == HealthStatus.WARNING else "🔴"
            )
            print(f"{status_icon} {check.name}: {check.message}")
        print()

        # Environment
        print("ENVIRONMENT:")
        print("-" * 30)
        for check in health_data["environment"]:
            status_icon = (
                "🟢" if check.status == HealthStatus.HEALTHY else "🟡" if check.status == HealthStatus.WARNING else "🔴"
            )
            print(f"{status_icon} {check.name}: {check.message}")
        print("=" * 60)


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def run_health_check() -> Dict[str, Any]:
    """Run a health check and return results."""
    monitor = get_health_monitor()
    return monitor.run_full_health_check()


def print_health_report() -> None:
    """Run a health check and print the report."""
    monitor = get_health_monitor()
    health_data = monitor.run_full_health_check()
    monitor.print_health_report(health_data)
