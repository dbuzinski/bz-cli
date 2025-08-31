"""
Tests for bz health monitoring functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from bz.health import (
    HealthMonitor,
    HealthStatus,
    SystemResource,
    HealthCheck,
    get_health_monitor,
    run_health_check,
    print_health_report,
)


class TestHealthStatus:
    """Test health status enumeration."""

    def test_health_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestSystemResource:
    """Test system resource data structure."""

    def test_system_resource_creation(self):
        """Test creating a system resource."""
        resource = SystemResource(
            name="Test Resource",
            value=50.0,
            unit="%",
            status=HealthStatus.WARNING,
            threshold_warning=80,
            threshold_critical=95,
        )

        assert resource.name == "Test Resource"
        assert resource.value == 50.0
        assert resource.unit == "%"
        assert resource.status == HealthStatus.WARNING
        assert resource.threshold_warning == 80
        assert resource.threshold_critical == 95


class TestHealthCheck:
    """Test health check data structure."""

    def test_health_check_creation(self):
        """Test creating a health check."""
        check = HealthCheck(
            name="Test Check", status=HealthStatus.HEALTHY, message="All good", details={"version": "1.0.0"}
        )

        assert check.name == "Test Check"
        assert check.status == HealthStatus.HEALTHY
        assert check.message == "All good"
        assert check.details == {"version": "1.0.0"}


class TestHealthMonitor:
    """Test health monitor functionality."""

    def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        monitor = HealthMonitor()
        assert monitor.checks == []

    def test_get_status_healthy(self):
        """Test status calculation for healthy values."""
        monitor = HealthMonitor()
        status = monitor._get_status(50.0, 80, 95)
        assert status == HealthStatus.HEALTHY

    def test_get_status_warning(self):
        """Test status calculation for warning values."""
        monitor = HealthMonitor()
        status = monitor._get_status(85.0, 80, 95)
        assert status == HealthStatus.WARNING

    def test_get_status_critical(self):
        """Test status calculation for critical values."""
        monitor = HealthMonitor()
        status = monitor._get_status(96.0, 80, 95)
        assert status == HealthStatus.CRITICAL

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_check_system_resources(self, mock_disk, mock_memory, mock_cpu):
        """Test system resource checking."""
        # Mock system resources
        mock_cpu.return_value = 75.0

        mock_memory_obj = MagicMock()
        mock_memory_obj.percent = 70.0
        mock_memory.return_value = mock_memory_obj

        mock_disk_obj = MagicMock()
        mock_disk_obj.used = 500
        mock_disk_obj.total = 1000
        mock_disk.return_value = mock_disk_obj

        monitor = HealthMonitor()
        resources = monitor.check_system_resources()

        assert len(resources) >= 3  # CPU, Memory, Disk
        assert all(isinstance(r, SystemResource) for r in resources)

        # Check CPU resource
        cpu_resource = next(r for r in resources if r.name == "CPU Usage")
        assert cpu_resource.value == 75.0
        assert cpu_resource.status == HealthStatus.HEALTHY

    @patch("torch.__version__", "2.0.0")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("psutil.__version__", "5.9.0")
    def test_check_dependencies(self, mock_cuda):
        """Test dependency checking."""
        monitor = HealthMonitor()
        checks = monitor.check_dependencies()

        assert len(checks) >= 3  # PyTorch, CUDA, psutil, plus optional deps
        assert all(isinstance(c, HealthCheck) for c in checks)

        # Check PyTorch
        pytorch_check = next(c for c in checks if c.name == "PyTorch")
        assert pytorch_check.status == HealthStatus.HEALTHY
        assert "2.0.0" in pytorch_check.message

        # Check CUDA (not available)
        cuda_check = next(c for c in checks if c.name == "CUDA")
        assert cuda_check.status == HealthStatus.WARNING

    def test_check_environment(self):
        """Test environment checking."""
        monitor = HealthMonitor()
        checks = monitor.check_environment()

        assert len(checks) >= 3  # Python version, working directory, env vars
        assert all(isinstance(c, HealthCheck) for c in checks)

        # Check Python version
        python_check = next(c for c in checks if c.name == "Python Version")
        assert python_check.status == HealthStatus.HEALTHY

        # Check working directory
        wd_check = next(c for c in checks if c.name == "Working Directory")
        assert wd_check.status == HealthStatus.HEALTHY

    def test_run_full_health_check(self):
        """Test full health check."""
        monitor = HealthMonitor()
        health_data = monitor.run_full_health_check()

        assert "status" in health_data
        assert "timestamp" in health_data
        assert "resources" in health_data
        assert "dependencies" in health_data
        assert "environment" in health_data
        assert "summary" in health_data

        summary = health_data["summary"]
        assert "total_checks" in summary
        assert "healthy" in summary
        assert "warnings" in summary
        assert "critical" in summary

        # Should have some status
        assert health_data["status"] in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL]

    def test_print_health_report(self, capsys):
        """Test health report printing."""
        monitor = HealthMonitor()

        # Create mock health data
        health_data = {
            "status": HealthStatus.HEALTHY,
            "timestamp": "2023-01-01T00:00:00",
            "resources": [
                SystemResource("CPU", 50.0, "%", HealthStatus.HEALTHY, 80, 95),
                SystemResource("Memory", 60.0, "%", HealthStatus.HEALTHY, 85, 95),
            ],
            "dependencies": [HealthCheck("PyTorch", HealthStatus.HEALTHY, "PyTorch 2.0.0 is available")],
            "environment": [HealthCheck("Python", HealthStatus.HEALTHY, "Python 3.10.0")],
            "summary": {"total_checks": 2, "healthy": 2, "warnings": 0, "critical": 0},
        }

        monitor.print_health_report(health_data)
        captured = capsys.readouterr()

        assert "BZ HEALTH CHECK REPORT" in captured.out
        assert "Overall Status: HEALTHY" in captured.out
        assert "CPU" in captured.out
        assert "Memory" in captured.out


class TestHealthFunctions:
    """Test health monitoring functions."""

    def test_get_health_monitor_singleton(self):
        """Test that get_health_monitor returns a singleton."""
        monitor1 = get_health_monitor()
        monitor2 = get_health_monitor()
        assert monitor1 is monitor2

    @patch("bz.health.get_health_monitor")
    def test_run_health_check(self, mock_get_monitor):
        """Test run_health_check function."""
        mock_monitor = MagicMock()
        mock_monitor.run_full_health_check.return_value = {"status": "healthy"}
        mock_get_monitor.return_value = mock_monitor

        result = run_health_check()

        assert result == {"status": "healthy"}
        mock_monitor.run_full_health_check.assert_called_once()

    @patch("bz.health.get_health_monitor")
    def test_print_health_report(self, mock_get_monitor):
        """Test print_health_report function."""
        mock_monitor = MagicMock()
        mock_get_monitor.return_value = mock_monitor

        print_health_report()

        mock_monitor.run_full_health_check.assert_called_once()
        mock_monitor.print_health_report.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
