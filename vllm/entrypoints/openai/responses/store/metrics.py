from __future__ import annotations

import threading
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class StoragePressureMetrics:
    used_bytes: int
    capacity_bytes: int

    @property
    def usage_ratio(self) -> float:
        if self.capacity_bytes <= 0:
            return 0.0
        return self.used_bytes / self.capacity_bytes


@dataclass(frozen=True, slots=True)
class CleanupMetricsSnapshot:
    duration_seconds: float
    memory_pressure: StoragePressureMetrics
    disk_pressure: StoragePressureMetrics
    memory_freed_bytes: int
    disk_freed_bytes: int


@dataclass(frozen=True, slots=True)
class EncryptionSpeedMetrics:
    operation_count: int
    plaintext_bytes: int
    duration_ns: int

    @property
    def throughput_mib_per_second(self) -> float:
        if self.duration_ns == 0:
            return 0.0
        return self.plaintext_bytes * 1_000_000_000 / self.duration_ns / (1024**2)


class ResponseStoreMetrics:
    """Aggregate hot-path measurements and log one cleanup snapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._encryption_operation_count = 0
        self._encryption_plaintext_bytes = 0
        self._encryption_duration_ns = 0

    def record_encryption(
        self,
        plaintext_bytes: int,
        duration_ns: int,
    ) -> None:
        with self._lock:
            self._encryption_operation_count += 1
            self._encryption_plaintext_bytes += plaintext_bytes
            self._encryption_duration_ns += duration_ns

    def log_cleanup(self, cleanup: CleanupMetricsSnapshot) -> None:
        encryption = self._take_encryption_snapshot()
        memory = cleanup.memory_pressure
        disk = cleanup.disk_pressure

        logger.info(
            "event=response_store_metrics "
            "cleanup_duration_ms=%.3f "
            "memory_used_bytes=%d memory_capacity_bytes=%d "
            "memory_usage_ratio=%.6f memory_freed_bytes=%d "
            "disk_used_bytes=%d disk_capacity_bytes=%d "
            "disk_usage_ratio=%.6f disk_freed_bytes=%d "
            "encryption_operations=%d encryption_plaintext_bytes=%d "
            "encryption_duration_ms=%.3f "
            "encryption_throughput_mib_per_second=%.3f",
            cleanup.duration_seconds * 1_000,
            memory.used_bytes,
            memory.capacity_bytes,
            memory.usage_ratio,
            cleanup.memory_freed_bytes,
            disk.used_bytes,
            disk.capacity_bytes,
            disk.usage_ratio,
            cleanup.disk_freed_bytes,
            encryption.operation_count,
            encryption.plaintext_bytes,
            encryption.duration_ns / 1_000_000,
            encryption.throughput_mib_per_second,
        )

    def _take_encryption_snapshot(self) -> EncryptionSpeedMetrics:
        with self._lock:
            snapshot = EncryptionSpeedMetrics(
                operation_count=self._encryption_operation_count,
                plaintext_bytes=self._encryption_plaintext_bytes,
                duration_ns=self._encryption_duration_ns,
            )
            self._encryption_operation_count = 0
            self._encryption_plaintext_bytes = 0
            self._encryption_duration_ns = 0
            return snapshot
