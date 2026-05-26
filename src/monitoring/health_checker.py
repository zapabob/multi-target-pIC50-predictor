"""
Health checking system for DAT Activity Predictor + TxGemma AI.
Comprehensive health checks for all system components.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil
import redis
import requests
import sqlalchemy
import torch

from ..config.settings import get_settings
from ..logging.production_logger import get_production_logger


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result."""

    name: str
    status: HealthStatus
    message: str
    details: dict[str, Any]
    response_time: float
    timestamp: float


class HealthChecker:
    """Comprehensive health checking system."""

    def __init__(self):
        """Initialize health checker."""
        self.settings = get_settings()
        self.logger = get_production_logger()
        self.checks = []
        self._setup_checks()

    def _setup_checks(self) -> None:
        """Setup health checks."""

        self.checks = [
            ("system_resources", self._check_system_resources),
            ("gpu_health", self._check_gpu_health),
            ("database_health", self._check_database_health),
            ("redis_health", self._check_redis_health),
            ("ollama_health", self._check_ollama_health),
            ("model_cache", self._check_model_cache),
            ("disk_space", self._check_disk_space),
            ("memory_usage", self._check_memory_usage),
            ("cpu_usage", self._check_cpu_usage),
            ("network_connectivity", self._check_network_connectivity),
        ]

    def check_all(self) -> dict[str, Any]:
        """Run all health checks."""

        results = []
        overall_status = HealthStatus.HEALTHY

        for check_name, check_func in self.checks:
            try:
                result = check_func()
                results.append(result)

                # Update overall status
                if result.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif (
                    result.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY
                ):
                    overall_status = HealthStatus.WARNING

            except Exception as e:
                error_result = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    details={"error": str(e)},
                    response_time=0.0,
                    timestamp=time.time(),
                )
                results.append(error_result)
                overall_status = HealthStatus.CRITICAL

        return {
            "overall": overall_status.value,
            "timestamp": time.time(),
            "checks": [result.__dict__ for result in results],
            "summary": self._generate_summary(results),
        }

    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resources."""

        start_time = time.time()

        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_usage = (disk.used / disk.total) * 100

            # Determine status
            status = HealthStatus.HEALTHY
            message = "System resources are healthy"

            if cpu_usage > 90 or memory_usage > 90 or disk_usage > 90:
                status = HealthStatus.CRITICAL
                message = "System resources are critically high"
            elif cpu_usage > 80 or memory_usage > 80 or disk_usage > 80:
                status = HealthStatus.WARNING
                message = "System resources are high"

            response_time = time.time() - start_time

            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                details={
                    "cpu_usage_percent": cpu_usage,
                    "memory_usage_percent": memory_usage,
                    "disk_usage_percent": disk_usage,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3),
                },
                response_time=response_time,
                timestamp=time.time(),
            )

        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}",
                details={"error": str(e)},
                response_time=time.time() - start_time,
                timestamp=time.time(),
            )

    def _check_gpu_health(self) -> HealthCheckResult:
        """Check GPU health."""

        start_time = time.time()

        try:
            if not torch.cuda.is_available():
                return HealthCheckResult(
                    name="gpu_health",
                    status=HealthStatus.WARNING,
                    message="CUDA not available",
                    details={"cuda_available": False},
                    response_time=time.time() - start_time,
                    timestamp=time.time(),
                )

            # GPU memory usage
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            memory_total = torch.cuda.get_device_properties(0).total_memory

            memory_usage_percent = (memory_allocated / memory_total) * 100

            # GPU temperature (if available)
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            except Exception:
                temperature = None
                utilization = None

            # Determine status
            status = HealthStatus.HEALTHY
            message = "GPU is healthy"

            if memory_usage_percent > 95:
                status = HealthStatus.CRITICAL
                message = "GPU memory usage is critically high"
            elif memory_usage_percent > 85:
                status = HealthStatus.WARNING
                message = "GPU memory usage is high"

            if temperature and temperature > 85:
                status = HealthStatus.CRITICAL
                message = "GPU temperature is critically high"
            elif temperature and temperature > 75:
                status = HealthStatus.WARNING
                message = "GPU temperature is high"

            response_time = time.time() - start_time

            return HealthCheckResult(
                name="gpu_health",
                status=status,
                message=message,
                details={
                    "cuda_available": True,
                    "device_name": torch.cuda.get_device_name(),
                    "memory_allocated_gb": memory_allocated / (1024**3),
                    "memory_reserved_gb": memory_reserved / (1024**3),
                    "memory_total_gb": memory_total / (1024**3),
                    "memory_usage_percent": memory_usage_percent,
                    "temperature_c": temperature,
                    "utilization_percent": utilization.gpu if utilization else None,
                },
                response_time=response_time,
                timestamp=time.time(),
            )

        except Exception as e:
            return HealthCheckResult(
                name="gpu_health",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check GPU health: {str(e)}",
                details={"error": str(e)},
                response_time=time.time() - start_time,
                timestamp=time.time(),
            )

    def _check_database_health(self) -> HealthCheckResult:
        """Check database health."""

        start_time = time.time()

        try:
            # Test database connection
            engine = sqlalchemy.create_engine(self.settings.database.url)
            with engine.connect() as conn:
                result = conn.execute(sqlalchemy.text("SELECT 1"))
                result.fetchone()

            response_time = time.time() - start_time

            return HealthCheckResult(
                name="database_health",
                status=HealthStatus.HEALTHY,
                message="Database connection is healthy",
                details={"connection_successful": True, "response_time_ms": response_time * 1000},
                response_time=response_time,
                timestamp=time.time(),
            )

        except Exception as e:
            return HealthCheckResult(
                name="database_health",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}",
                details={"error": str(e)},
                response_time=time.time() - start_time,
                timestamp=time.time(),
            )

    def _check_redis_health(self) -> HealthCheckResult:
        """Check Redis health."""

        start_time = time.time()

        try:
            # Test Redis connection
            r = redis.from_url(self.settings.redis.url)
            r.ping()

            # Get Redis info
            info = r.info()

            response_time = time.time() - start_time

            return HealthCheckResult(
                name="redis_health",
                status=HealthStatus.HEALTHY,
                message="Redis connection is healthy",
                details={
                    "connection_successful": True,
                    "response_time_ms": response_time * 1000,
                    "used_memory_mb": info.get("used_memory", 0) / (1024**2),
                    "connected_clients": info.get("connected_clients", 0),
                },
                response_time=response_time,
                timestamp=time.time(),
            )

        except Exception as e:
            return HealthCheckResult(
                name="redis_health",
                status=HealthStatus.CRITICAL,
                message=f"Redis connection failed: {str(e)}",
                details={"error": str(e)},
                response_time=time.time() - start_time,
                timestamp=time.time(),
            )

    def _check_ollama_health(self) -> HealthCheckResult:
        """Check Ollama health."""

        start_time = time.time()

        try:
            # Test Ollama API
            response = requests.get(f"{self.settings.txgemma.host}/api/tags", timeout=5)

            if response.status_code == 200:
                models = response.json().get("models", [])
                txgemma_available = any(model["name"].startswith("txgemma") for model in models)

                response_time = time.time() - start_time

                return HealthCheckResult(
                    name="ollama_health",
                    status=HealthStatus.HEALTHY,
                    message="Ollama is healthy",
                    details={
                        "connection_successful": True,
                        "response_time_ms": response_time * 1000,
                        "txgemma_available": txgemma_available,
                        "total_models": len(models),
                    },
                    response_time=response_time,
                    timestamp=time.time(),
                )
            else:
                return HealthCheckResult(
                    name="ollama_health",
                    status=HealthStatus.CRITICAL,
                    message=f"Ollama API returned status {response.status_code}",
                    details={"status_code": response.status_code},
                    response_time=time.time() - start_time,
                    timestamp=time.time(),
                )

        except Exception as e:
            return HealthCheckResult(
                name="ollama_health",
                status=HealthStatus.WARNING,
                message=f"Ollama connection failed: {str(e)}",
                details={"error": str(e)},
                response_time=time.time() - start_time,
                timestamp=time.time(),
            )

    def _check_model_cache(self) -> HealthCheckResult:
        """Check model cache health."""

        start_time = time.time()

        try:
            from pathlib import Path

            cache_dir = Path(self.settings.model.cache_dir)

            if not cache_dir.exists():
                return HealthCheckResult(
                    name="model_cache",
                    status=HealthStatus.WARNING,
                    message="Model cache directory does not exist",
                    details={"cache_dir": str(cache_dir)},
                    response_time=time.time() - start_time,
                    timestamp=time.time(),
                )

            # Check cache directory size
            total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            total_size_gb = total_size / (1024**3)

            # Check for model files
            model_files = list(cache_dir.glob("*.pt")) + list(cache_dir.glob("*.pth"))

            response_time = time.time() - start_time

            return HealthCheckResult(
                name="model_cache",
                status=HealthStatus.HEALTHY,
                message="Model cache is healthy",
                details={
                    "cache_dir": str(cache_dir),
                    "total_size_gb": total_size_gb,
                    "model_files_count": len(model_files),
                    "model_files": [str(f.name) for f in model_files[:5]],  # First 5 files
                },
                response_time=response_time,
                timestamp=time.time(),
            )

        except Exception as e:
            return HealthCheckResult(
                name="model_cache",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check model cache: {str(e)}",
                details={"error": str(e)},
                response_time=time.time() - start_time,
                timestamp=time.time(),
            )

    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space."""

        start_time = time.time()

        try:
            disk = psutil.disk_usage("/")
            free_percent = (disk.free / disk.total) * 100

            status = HealthStatus.HEALTHY
            message = "Disk space is healthy"

            if free_percent < 5:
                status = HealthStatus.CRITICAL
                message = "Disk space is critically low"
            elif free_percent < 10:
                status = HealthStatus.WARNING
                message = "Disk space is low"

            response_time = time.time() - start_time

            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                details={
                    "free_percent": free_percent,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                },
                response_time=response_time,
                timestamp=time.time(),
            )

        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {str(e)}",
                details={"error": str(e)},
                response_time=time.time() - start_time,
                timestamp=time.time(),
            )

    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage."""

        start_time = time.time()

        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent

            status = HealthStatus.HEALTHY
            message = "Memory usage is healthy"

            if usage_percent > 95:
                status = HealthStatus.CRITICAL
                message = "Memory usage is critically high"
            elif usage_percent > 85:
                status = HealthStatus.WARNING
                message = "Memory usage is high"

            response_time = time.time() - start_time

            return HealthCheckResult(
                name="memory_usage",
                status=status,
                message=message,
                details={
                    "usage_percent": usage_percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3),
                    "used_gb": memory.used / (1024**3),
                },
                response_time=response_time,
                timestamp=time.time(),
            )

        except Exception as e:
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check memory usage: {str(e)}",
                details={"error": str(e)},
                response_time=time.time() - start_time,
                timestamp=time.time(),
            )

    def _check_cpu_usage(self) -> HealthCheckResult:
        """Check CPU usage."""

        start_time = time.time()

        try:
            usage_percent = psutil.cpu_percent(interval=1)

            status = HealthStatus.HEALTHY
            message = "CPU usage is healthy"

            if usage_percent > 95:
                status = HealthStatus.CRITICAL
                message = "CPU usage is critically high"
            elif usage_percent > 85:
                status = HealthStatus.WARNING
                message = "CPU usage is high"

            response_time = time.time() - start_time

            return HealthCheckResult(
                name="cpu_usage",
                status=status,
                message=message,
                details={"usage_percent": usage_percent, "cpu_count": psutil.cpu_count()},
                response_time=response_time,
                timestamp=time.time(),
            )

        except Exception as e:
            return HealthCheckResult(
                name="cpu_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check CPU usage: {str(e)}",
                details={"error": str(e)},
                response_time=time.time() - start_time,
                timestamp=time.time(),
            )

    def _check_network_connectivity(self) -> HealthCheckResult:
        """Check network connectivity."""

        start_time = time.time()

        try:
            # Test external connectivity
            requests.get("https://www.google.com", timeout=5)

            response_time = time.time() - start_time

            return HealthCheckResult(
                name="network_connectivity",
                status=HealthStatus.HEALTHY,
                message="Network connectivity is healthy",
                details={"external_connectivity": True, "response_time_ms": response_time * 1000},
                response_time=response_time,
                timestamp=time.time(),
            )

        except Exception as e:
            return HealthCheckResult(
                name="network_connectivity",
                status=HealthStatus.WARNING,
                message=f"Network connectivity failed: {str(e)}",
                details={"error": str(e)},
                response_time=time.time() - start_time,
                timestamp=time.time(),
            )

    def _generate_summary(self, results: list[HealthCheckResult]) -> dict[str, Any]:
        """Generate health check summary."""

        total_checks = len(results)
        healthy_checks = sum(1 for r in results if r.status == HealthStatus.HEALTHY)
        warning_checks = sum(1 for r in results if r.status == HealthStatus.WARNING)
        critical_checks = sum(1 for r in results if r.status == HealthStatus.CRITICAL)

        return {
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "warning_checks": warning_checks,
            "critical_checks": critical_checks,
            "health_percentage": (healthy_checks / total_checks) * 100 if total_checks > 0 else 0,
        }
