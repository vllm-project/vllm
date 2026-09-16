from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass

from vllm.logger import init_logger

from .base import SessionMetadata
from .eviction import (
    CapacityWaterMarks,
    DiskEvictionBatchResult,
    DiskEvictionExecutor,
    EvictionCandidate,
    EvictionExecutionContext,
    EvictionPolicy,
    EvictionSelectionBudget,
    EvictionTriggerDecision,
    MemoryEvictionBatchResult,
    MemoryEvictionExecutor,
    StorageTier,
)
from .metrics import (
    CleanupMetricsSnapshot,
    StoragePressureMetrics,
)
from .tiered import TieredSessionStore

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class TierCleanupConfig:
    watermarks: CapacityWaterMarks
    budget: EvictionSelectionBudget

    def __post_init__(self) -> None:
        if self.budget.max_candidates <= 0:
            raise ValueError("max_candidates must be greater than 0")
        if self.budget.max_bytes <= 0:
            raise ValueError("max_bytes must be greater than 0")


@dataclass(frozen=True, slots=True)
class PeriodicCleanupConfig:
    interval_seconds: float
    memory: TierCleanupConfig
    disk: TierCleanupConfig

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than 0")


@dataclass(frozen=True, slots=True)
class CleanupRunResult:
    """Cleanup outcome with started_at in Unix milliseconds."""

    started_at: int
    duration_seconds: float
    scanned_session_count: int
    memory_decision: EvictionTriggerDecision
    disk_decision: EvictionTriggerDecision
    disk_pressure: DiskPressureResult
    memory_result: MemoryEvictionBatchResult | None
    disk_result: DiskEvictionBatchResult | None


@dataclass(frozen=True, slots=True)
class DiskPressureResult:
    total_used_bytes: int
    protected_used_bytes: int
    reclaimable_used_bytes: int
    unavailable_used_bytes: int
    required_free_bytes: int
    blocked_bytes: int
    pressure_blocked: bool

    @property
    def blocked_by_protected_bytes(self) -> bool:
        return self.pressure_blocked and self.protected_used_bytes > 0


class PeriodicSessionStoreCleanup:
    """Run bounded Memory and Disk cleanup batches at a fixed interval."""

    def __init__(
        self,
        store: TieredSessionStore,
        config: PeriodicCleanupConfig,
    ) -> None:
        self._store = store
        self._config = config
        self._memory_policy = EvictionPolicy(StorageTier.MEMORY)
        self._memory_executor = MemoryEvictionExecutor(store)
        self._disk_policy = (
            EvictionPolicy(StorageTier.DISK) if store.disk_enabled else None
        )
        self._disk_executor = (
            DiskEvictionExecutor(store) if store.disk_enabled else None
        )
        self._run_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._last_result: CleanupRunResult | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def last_result(self) -> CleanupRunResult | None:
        return self._last_result

    def start(self) -> None:
        if self.is_running:
            return
        self._task = asyncio.create_task(
            self._run_periodically(),
            name="responses-session-store-cleanup",
        )

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return

        self._task = None
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def run_once(self) -> CleanupRunResult:
        """Build one snapshot and run at most one bounded batch per tier."""
        async with self._run_lock:
            started_at = time.time_ns() // 1_000_000
            started_monotonic = time.monotonic()

            records = await self._store.list_metadata()
            scanned_session_count = len(records)
            memory_candidates = self._memory_policy.build_sorted_candidates(
                records,
                started_at,
            )

            if self._store.disk_enabled:
                memory_used_bytes, disk_used_bytes = await asyncio.gather(
                    self._store.memory_used_bytes(),
                    self._store.disk_used_bytes(),
                )
                assert self._disk_policy is not None
                disk_candidates = self._disk_policy.build_sorted_candidates(
                    records,
                    started_at,
                )
                disk_decision = self._disk_policy.evaluate_trigger(
                    disk_candidates,
                    disk_used_bytes,
                    self._config.disk.watermarks,
                )
                disk_pressure = self._build_disk_pressure(
                    records=records,
                    candidates=disk_candidates,
                    decision=disk_decision,
                )
            else:
                memory_used_bytes = await self._store.memory_used_bytes()
                disk_candidates = ()
                disk_decision = self._disabled_disk_decision()
                disk_pressure = self._disabled_disk_pressure()

            memory_decision = self._memory_policy.evaluate_trigger(
                memory_candidates,
                memory_used_bytes,
                self._config.memory.watermarks,
            )

            memory_result = await self._cleanup_memory(
                memory_candidates,
                memory_decision,
                scanned_session_count,
                started_at,
            )
            disk_result = await self._cleanup_disk(
                disk_candidates,
                disk_decision,
                scanned_session_count,
            )

            result = CleanupRunResult(
                started_at=started_at,
                duration_seconds=time.monotonic() - started_monotonic,
                scanned_session_count=scanned_session_count,
                memory_decision=memory_decision,
                disk_decision=disk_decision,
                disk_pressure=disk_pressure,
                memory_result=memory_result,
                disk_result=disk_result,
            )
            self._last_result = result
            self._store.metrics.log_cleanup(self._build_metrics_snapshot(result))
            return result

    async def _cleanup_memory(
        self,
        candidates: Sequence[EvictionCandidate],
        decision: EvictionTriggerDecision,
        scanned_session_count: int,
        now: int,
    ) -> MemoryEvictionBatchResult | None:
        if not decision.should_evict:
            return None

        assert decision.trigger_reason is not None
        selection = self._memory_policy.select_candidates(
            candidates=candidates,
            allow_unexpired=decision.high_watermark_reached,
            target_free_bytes=self._target_free_bytes(decision),
            budget=self._config.memory.budget,
        )
        context = EvictionExecutionContext(
            trigger_reason=decision.trigger_reason,
            scanned_count=scanned_session_count,
            initial_used_bytes=decision.used_bytes,
            target_used_bytes=decision.target_used_bytes,
        )
        return await self._memory_executor.execute(
            selection=selection,
            context=context,
            now=now,
            force=(decision.high_watermark_reached and not self._store.disk_enabled),
        )

    async def _cleanup_disk(
        self,
        candidates: Sequence[EvictionCandidate],
        decision: EvictionTriggerDecision,
        scanned_session_count: int,
    ) -> DiskEvictionBatchResult | None:
        if not decision.should_evict:
            return None

        assert self._disk_policy is not None
        assert self._disk_executor is not None
        assert decision.trigger_reason is not None
        selection = self._disk_policy.select_candidates(
            candidates=candidates,
            allow_unexpired=decision.high_watermark_reached,
            target_free_bytes=self._target_free_bytes(decision),
            budget=self._config.disk.budget,
        )
        if not selection.candidates:
            return None

        context = EvictionExecutionContext(
            trigger_reason=decision.trigger_reason,
            scanned_count=scanned_session_count,
            initial_used_bytes=decision.used_bytes,
            target_used_bytes=decision.target_used_bytes,
        )
        return await self._disk_executor.execute(
            selection=selection,
            context=context,
        )

    async def _run_periodically(self) -> None:
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Periodic session store cleanup failed")

            await asyncio.sleep(self._config.interval_seconds)

    @staticmethod
    def _target_free_bytes(decision: EvictionTriggerDecision) -> int | None:
        if decision.target_used_bytes is None:
            return None
        return max(0, decision.used_bytes - decision.target_used_bytes)

    def _build_metrics_snapshot(
        self,
        result: CleanupRunResult,
    ) -> CleanupMetricsSnapshot:
        memory_used_after = (
            result.memory_result.final_used_bytes
            if result.memory_result is not None
            else result.memory_decision.used_bytes
        )
        disk_used_after = (
            result.disk_result.final_used_bytes
            if result.disk_result is not None
            else result.disk_decision.used_bytes
        )
        return CleanupMetricsSnapshot(
            duration_seconds=result.duration_seconds,
            memory_pressure=StoragePressureMetrics(
                used_bytes=memory_used_after,
                capacity_bytes=self._config.memory.watermarks.max_bytes,
            ),
            disk_pressure=StoragePressureMetrics(
                used_bytes=disk_used_after,
                capacity_bytes=self._config.disk.watermarks.max_bytes,
            ),
            memory_freed_bytes=(
                0
                if result.memory_result is None
                else result.memory_result.actual_free_bytes
            ),
            disk_freed_bytes=(
                0
                if result.disk_result is None
                else result.disk_result.actual_free_bytes
            ),
        )

    @staticmethod
    def _build_disk_pressure(
        records: Sequence[SessionMetadata],
        candidates: Sequence[EvictionCandidate],
        decision: EvictionTriggerDecision,
    ) -> DiskPressureResult:
        protected_used_bytes = sum(
            record.disk_size_bytes for record in records if record.memory_resident
        )
        reclaimable_used_bytes = sum(candidate.size_bytes for candidate in candidates)
        unavailable_used_bytes = max(
            0,
            decision.used_bytes - protected_used_bytes - reclaimable_used_bytes,
        )
        required_free_bytes = (
            PeriodicSessionStoreCleanup._target_free_bytes(decision) or 0
        )
        blocked_bytes = max(
            0,
            required_free_bytes - reclaimable_used_bytes,
        )

        return DiskPressureResult(
            total_used_bytes=decision.used_bytes,
            protected_used_bytes=protected_used_bytes,
            reclaimable_used_bytes=reclaimable_used_bytes,
            unavailable_used_bytes=unavailable_used_bytes,
            required_free_bytes=required_free_bytes,
            blocked_bytes=blocked_bytes,
            pressure_blocked=(decision.high_watermark_reached and blocked_bytes > 0),
        )

    @staticmethod
    def _disabled_disk_decision() -> EvictionTriggerDecision:
        return EvictionTriggerDecision(
            should_evict=False,
            has_expired_candidates=False,
            high_watermark_reached=False,
            used_bytes=0,
            target_used_bytes=None,
            trigger_reason=None,
        )

    @staticmethod
    def _disabled_disk_pressure() -> DiskPressureResult:
        return DiskPressureResult(
            total_used_bytes=0,
            protected_used_bytes=0,
            reclaimable_used_bytes=0,
            unavailable_used_bytes=0,
            required_free_bytes=0,
            blocked_bytes=0,
            pressure_blocked=False,
        )
