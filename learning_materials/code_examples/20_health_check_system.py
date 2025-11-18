"""
Example 20: Health Check System

Implements comprehensive health checks for production deployments.

Usage:
    python 20_health_check_system.py
"""

import asyncio
import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    latency_ms: float
    metadata: Optional[Dict] = None


class HealthChecker:
    """Comprehensive health check system."""

    def __init__(self):
        self.checks: List[callable] = []

    def add_check(self, check_func: callable) -> None:
        """Add a health check function."""
        self.checks.append(check_func)

    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}

        for check in self.checks:
            start = time.time()
            try:
                result = await check()
                latency = (time.time() - start) * 1000
                results[result.name] = HealthCheckResult(
                    name=result.name,
                    status=result.status,
                    message=result.message,
                    latency_ms=latency,
                    metadata=result.metadata
                )
            except Exception as e:
                latency = (time.time() - start) * 1000
                results[check.__name__] = HealthCheckResult(
                    name=check.__name__,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    latency_ms=latency
                )

        return results

    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Determine overall health status."""
        if any(r.status == HealthStatus.UNHEALTHY for r in results.values()):
            return HealthStatus.UNHEALTHY
        elif any(r.status == HealthStatus.DEGRADED for r in results.values()):
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY


# Define health checks
async def check_gpu_availability() -> HealthCheckResult:
    """Check if GPU is available."""
    available = torch.cuda.is_available()

    if available:
        device_count = torch.cuda.device_count()
        return HealthCheckResult(
            name="gpu_availability",
            status=HealthStatus.HEALTHY,
            message=f"GPU available: {device_count} device(s)",
            latency_ms=0,
            metadata={"device_count": device_count}
        )
    else:
        return HealthCheckResult(
            name="gpu_availability",
            status=HealthStatus.DEGRADED,
            message="No GPU available, using CPU",
            latency_ms=0
        )


async def check_model_loaded() -> HealthCheckResult:
    """Check if model is loaded (simulated)."""
    # In real implementation, check actual model state
    model_loaded = True

    if model_loaded:
        return HealthCheckResult(
            name="model_status",
            status=HealthStatus.HEALTHY,
            message="Model loaded successfully",
            latency_ms=0
        )
    else:
        return HealthCheckResult(
            name="model_status",
            status=HealthStatus.UNHEALTHY,
            message="Model not loaded",
            latency_ms=0
        )


async def check_memory_usage() -> HealthCheckResult:
    """Check GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        usage_pct = (allocated / total) * 100

        if usage_pct < 80:
            status = HealthStatus.HEALTHY
            message = f"Memory usage: {usage_pct:.1f}%"
        elif usage_pct < 95:
            status = HealthStatus.DEGRADED
            message = f"High memory usage: {usage_pct:.1f}%"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"Critical memory usage: {usage_pct:.1f}%"

        return HealthCheckResult(
            name="memory_usage",
            status=status,
            message=message,
            latency_ms=0,
            metadata={"allocated_gb": allocated, "total_gb": total}
        )

    return HealthCheckResult(
        name="memory_usage",
        status=HealthStatus.HEALTHY,
        message="GPU not available",
        latency_ms=0
    )


async def main():
    """Demo health check system."""
    print("=== Health Check System Demo ===\n")

    # Setup health checker
    checker = HealthChecker()
    checker.add_check(check_gpu_availability)
    checker.add_check(check_model_loaded)
    checker.add_check(check_memory_usage)

    # Run checks
    results = await checker.run_all_checks()
    overall = checker.get_overall_status(results)

    # Display results
    print(f"Overall Status: {overall.value.upper()}\n")
    print("Individual Checks:")
    print("-" * 80)

    for name, result in results.items():
        status_symbol = {
            HealthStatus.HEALTHY: "✓",
            HealthStatus.DEGRADED: "⚠",
            HealthStatus.UNHEALTHY: "✗"
        }[result.status]

        print(f"{status_symbol} {result.name}")
        print(f"  Status: {result.status.value}")
        print(f"  Message: {result.message}")
        print(f"  Latency: {result.latency_ms:.2f}ms")
        if result.metadata:
            print(f"  Metadata: {result.metadata}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
