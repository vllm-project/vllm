import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from vllm.logger import init_logger

from .base import (
    DiskEvictionResult,
    DiskEvictionStatus,
    MemoryEvictionResult,
    MemoryEvictionStatus,
    SessionMetadata,
)

logger = init_logger(__name__)


class StorageTier(str, Enum):
    MEMORY = "memory"
    DISK = "disk"


class EvictionTriggerReason(str, Enum):
    TTL = "ttl"
    HIGH_WATERMARK = "high_watermark"
    TTL_AND_HIGH_WATERMARK = "ttl_and_high_watermark"
    CAPACITY = "capacity"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class EvictionExecutionContext:
    trigger_reason: EvictionTriggerReason
    scanned_count: int
    initial_used_bytes: int
    target_used_bytes: int | None


@dataclass(frozen=True, slots=True)
class EvictionCandidate:
    """Eviction candidate snapshot with timestamps in Unix milliseconds."""
    session_id: str
    response_id: str
    created_at: int
    updated_at: int
    expires_at: int | None
    size_bytes: int
    is_expired: bool

@dataclass(frozen=True, slots=True)
class EvictionTriggerDecision:
    """the decision of whether to start to evict"""

    should_evict: bool
    has_expired_candidates: bool
    high_watermark_reached: bool
    used_bytes: int
    target_used_bytes: int | None
    trigger_reason: EvictionTriggerReason | None

@dataclass(frozen=True, slots=True)
class CapacityTriggerDecision:
    """Capacity admission decision before a write."""

    should_cleanup: bool
    should_reject: bool
    used_bytes: int
    old_size_bytes: int
    new_size_bytes: int
    projected_used_bytes: int
    minimum_bytes_to_free: int
    preferred_bytes_to_free: int

@dataclass(frozen=True, slots=True)
class EvictionSelectionBudget:
    """The number of candidates for single-batch
        elimination tasks and the byte limit
        for planned releases."""

    max_candidates: int
    max_bytes: int

@dataclass(frozen=True, slots=True)
class EvictionSelection:
    candidates: tuple[EvictionCandidate,...]
    planned_free_bytes: int
    target_free_bytes: int | None
    planned_target_satisfied: bool
    candidates_limit_reached: bool
    byte_limit_reached: bool
    blocked_candidate: EvictionCandidate | None


@dataclass(frozen=True, slots=True)
class CapacityWaterMarks:
    """the storage and defaults of watermarks"""

    max_bytes: int
    high_watermark_bytes: int
    low_watermark_bytes: int

    def __post_init__(self) -> None:
        if not (
            0 <= self.low_watermark_bytes < self.high_watermark_bytes <= self.max_bytes
        ):
            raise ValueError(
                "Capacity watermarks must satisfy "
                "0 <= low_watermark < high_watermark <= max_bytes."
            )

@dataclass(frozen=True, slots=True)
class MemoryEvictionBatchResult:
    """the result of a single batch eviction"""

    trigger_reason: EvictionTriggerReason
    scanned_count: int
    results: tuple[MemoryEvictionResult, ...]
    selected_count: int
    processed_count: int
    remaining_candidate_count: int
    actual_free_bytes: int
    initial_used_bytes: int
    estimated_final_used_bytes: int
    final_used_bytes: int
    target_free_bytes: int | None
    target_used_bytes: int | None
    actual_target_satisfied: bool | None
    stopped_after_reaching_target: bool
    candidates_limit_reached: bool
    byte_limit_reached: bool

    @property
    def evicted_count(self) -> int:
        """the number of memory copies that have been evicted"""
        return sum(result.evicted for result in self.results)

    @property
    def skipped_count(self) -> int:
        """the number of memory copies that have been skipped"""
        return self.processed_count - self.evicted_count

    @property
    def version_conflict_count(self) -> int:
        return sum(
            result.status
            in {
                MemoryEvictionStatus.RESPONSE_CHANGED,
                MemoryEvictionStatus.VERSION_CHANGED,
                MemoryEvictionStatus.ACCESSED_AFTER_SELECTION,
            }
            for result in self.results
        )

    @property
    def pending_skipped_count(self) -> int:
        return sum(
            result.status is MemoryEvictionStatus.PERSISTENCE_PENDING
            for result in self.results
        )

    @property
    def persistence_failed_count(self) -> int:
        return sum(
            result.status is MemoryEvictionStatus.PERSISTENCE_FAILED
            for result in self.results
        )

    @property
    def reached_low_watermark(self) -> bool | None:
        return self.actual_target_satisfied

    @property
    def budget_exhausted(self) -> bool:
        return self.candidates_limit_reached or self.byte_limit_reached


@dataclass(frozen=True, slots=True)
class DiskEvictionBatchResult:
    trigger_reason: EvictionTriggerReason
    scanned_count: int
    results: tuple[DiskEvictionResult, ...]
    selected_count: int
    processed_count: int
    remaining_candidate_count: int
    actual_free_bytes: int
    initial_used_bytes: int
    estimated_final_used_bytes: int
    final_used_bytes: int
    target_free_bytes: int | None
    target_used_bytes: int | None
    actual_target_satisfied: bool | None
    stopped_after_reaching_target: bool
    candidates_limit_reached: bool
    byte_limit_reached: bool

    @property
    def evicted_count(self) -> int:
        return sum(result.evicted for result in self.results)

    @property
    def skipped_count(self) -> int:
        return self.processed_count - self.evicted_count

    @property
    def version_conflict_count(self) -> int:
        return sum(
            result.status
            in {
                DiskEvictionStatus.RESPONSE_CHANGED,
                DiskEvictionStatus.VERSION_CHANGED,
            }
            for result in self.results
        )

    @property
    def pending_skipped_count(self) -> int:
        return 0

    @property
    def persistence_failed_count(self) -> int:
        return 0

    @property
    def reached_low_watermark(self) -> bool | None:
        return self.actual_target_satisfied

    @property
    def budget_exhausted(self) -> bool:
        return self.candidates_limit_reached or self.byte_limit_reached



@dataclass(frozen=True, slots=True)
class EvictionPolicy:
    """Build and sort eviction candidates
        for the specified storage tier."""
    tier: StorageTier

    def build_sorted_candidates(
        self,
        records: Iterable[SessionMetadata],
        now: int,
    ) -> list[EvictionCandidate]:
        """
        All timestamps, including now, are Unix milliseconds.

        The sorting rules are:
        1. Records that have expired in the current storage tier;
        2. Records with an earlier updated_at timestamp;
        3. Records with an earlier created_at timestamp;
        4. Lexicographic order of session_id."
        """
        candidates = [
            self._build_candidate(metadata, now)
            for metadata in records
            if (
                self.tier is StorageTier.MEMORY
                and metadata.memory_resident
            )
            or (
                self.tier is StorageTier.DISK
                and not metadata.memory_resident
            )
        ]
        return sorted(candidates, key=self._sort_key)

    def _build_candidate(
        self,
        metadata: SessionMetadata,
        now: int,
    ) -> EvictionCandidate:
        """ build the snapshot of candidates """
        expires_at, size_bytes = self._get_tier_values(metadata)

        return EvictionCandidate(
            session_id=metadata.session_id,
            response_id=metadata.response_id,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
            expires_at=expires_at,
            size_bytes=size_bytes,
            is_expired=expires_at is not None and expires_at <= now,
        )

    def evaluate_trigger(
        self,
        candidates: Sequence[EvictionCandidate],
        used_bytes: int,
        watermarks: CapacityWaterMarks
    ) -> EvictionTriggerDecision:
        """Determine whether the current storage tier needs eviction.

        TTL-based expiry cleanup does not depend on watermarks. As long as
        expired candidates exist, cleanup should be triggered.

        LRU eviction of non-expired sessions is only triggered when the
        current usage reaches the high watermark, and continues until
        usage falls back to the low watermark.

        Args:
            candidates: Eviction candidates that have already been sorted.
            used_bytes: The actual billed bytes currently used by this storage tier.
            watermarks: The capacity and watermark configuration of this storage tier.

        Returns:
            The eviction trigger decision for this round.
        """
        has_expired_candidates = any(
            candidate.is_expired for candidate in candidates
        )
        high_watermark_reached = used_bytes >= watermarks.high_watermark_bytes

        should_evict = has_expired_candidates or high_watermark_reached
        target_used_bytes = (
                            watermarks.low_watermark_bytes
                            if high_watermark_reached else None
                            )
        if has_expired_candidates and high_watermark_reached:
            trigger_reason = EvictionTriggerReason.TTL_AND_HIGH_WATERMARK
        elif has_expired_candidates:
            trigger_reason = EvictionTriggerReason.TTL
        elif high_watermark_reached:
            trigger_reason = EvictionTriggerReason.HIGH_WATERMARK
        else:
            trigger_reason = None

        return EvictionTriggerDecision(
            should_evict=should_evict,
            has_expired_candidates=has_expired_candidates,
            high_watermark_reached=high_watermark_reached,
            used_bytes=used_bytes,
            target_used_bytes=target_used_bytes,
            trigger_reason=trigger_reason,
        )

    def evaluate_capacity_trigger(
        self,
        used_bytes: int,
        old_size_bytes: int,
        new_size_bytes: int,
        watermarks: CapacityWaterMarks,
    ) -> CapacityTriggerDecision:
        """Determine whether capacity eviction is needed before a write.

        The projected usage after adding or overwriting a record is:

            used_bytes - old_size_bytes + new_size_bytes

        If a single new record exceeds the maximum capacity, no amount of
        evicting other records can make room for it in the current storage
        tier; the write to this tier should be rejected outright.

        If the projected usage exceeds the maximum capacity, TTL and LRU
        eviction must be performed before the write. After eviction
        completes, the caller must re-evaluate using the latest capacity
        data.

        Args:
            used_bytes: The actual billed bytes currently used by this storage tier.
            old_size_bytes: The byte size of the existing record for the same
                session; 0 for a new record.
            new_size_bytes: The byte size of the new record to be written.
            watermarks: The capacity and watermark configuration of this storage tier.

        Returns:
            The capacity admission decision before the write.
        """

        projected_used_bytes = used_bytes - old_size_bytes + new_size_bytes

        """
        When a single record exceeds the maximum capacity,
        clearing other sessions cannot accommodate it either.
        """
        should_reject = new_size_bytes > watermarks.max_bytes
        should_cleanup = (
            not should_reject
            and projected_used_bytes > watermarks.max_bytes
        )

        minimum_bytes_to_free = (
            max(0, projected_used_bytes - watermarks.max_bytes)
            if should_cleanup else 0
        )

        preferred_bytes_to_free = (
            max(0, projected_used_bytes - watermarks.low_watermark_bytes)
            if should_cleanup else 0
        )

        return CapacityTriggerDecision(
            should_cleanup=should_cleanup,
            should_reject=should_reject,
            used_bytes=used_bytes,
            old_size_bytes=old_size_bytes,
            new_size_bytes=new_size_bytes,
            projected_used_bytes=projected_used_bytes,
            minimum_bytes_to_free=minimum_bytes_to_free,
            preferred_bytes_to_free=preferred_bytes_to_free,
        )

    def select_candidates(
        self,
        candidates: Sequence[EvictionCandidate],
        allow_unexpired: bool,
        target_free_bytes: int | None,
        budget: EvictionSelectionBudget,
    ) -> EvictionSelection:
        """
        Select eviction candidates based on the release target and batch budget.

        Expired candidates are always eligible for eviction.

        Unexpired candidates are only allowed into this batch eviction plan
        under high watermark or maximum capacity pressure, i.e., when
        allow_unexpired=True.

        This method assumes that candidates have already been sorted according
        to the following rules:

        1. Expired records take priority;
        2. update_at in ascending order;
        3. create_at in ascending order;
        4. session_id in lexicographic order.

        Args:
            candidates: Candidates that have already been sorted by TTL/LRU.
            allow_unexpired: Whether to allow selecting unexpired candidates.
            target_free_bytes: The number of bytes intended to be freed in this
                round. May be set to None when performing TTL-only cleanup.
            budget: The limit on the number of candidates in a single batch
                and the number of bytes to be released as planned.

        Returns:
            The selected candidates for this batch eviction and the budget usage.
        """

        selected: list[EvictionCandidate] = []
        planned_free_bytes = 0
        candidate_limit_reached = False
        byte_limit_reached = False
        blocked_candidate: EvictionCandidate | None = None

        for candidate in candidates:
            if candidate.size_bytes < 0:
                raise ValueError(
                    "Candidate size_bytes must be non-negative:"
                    f"{candidate.size_bytes!r}"
                )

            if not candidate.is_expired and not allow_unexpired:
                break

            if (
                not candidate.is_expired
                and target_free_bytes is not None
                and planned_free_bytes >= target_free_bytes
            ):
                break

            if len(selected) >= budget.max_candidates:
                candidate_limit_reached = True
                blocked_candidate = candidate
                break

            projected_free_bytes = planned_free_bytes + candidate.size_bytes

            if projected_free_bytes > budget.max_bytes:
                byte_limit_reached = True
                # The first candidate must still make progress even when it is
                # larger than the batch budget; otherwise the same LRU entry
                # can block every cleanup run indefinitely.
                if not selected:
                    selected.append(candidate)
                    planned_free_bytes = projected_free_bytes
                else:
                    blocked_candidate = candidate
                break

            selected.append(candidate)
            planned_free_bytes = projected_free_bytes

        planned_target_satisfied = (
            target_free_bytes is None
            or planned_free_bytes >= target_free_bytes
        )

        return EvictionSelection(
            candidates=tuple(selected),
            planned_free_bytes=planned_free_bytes,
            target_free_bytes=target_free_bytes,
            planned_target_satisfied=planned_target_satisfied,
            candidates_limit_reached=candidate_limit_reached,
            byte_limit_reached=byte_limit_reached,
            blocked_candidate=blocked_candidate
        )


    def _get_tier_values(
        self,
        metadata: SessionMetadata,
    ) -> tuple[int | None, int]:
        """get expire time and used bytes"""
        if self.tier is StorageTier.MEMORY:
            return (
                metadata.mem_idle_expires_at,
                metadata.mem_size_bytes
            )
        if self.tier is StorageTier.DISK:
            return (
                metadata.disk_idle_expires_at,
                metadata.disk_size_bytes
            )
        raise ValueError(f"Unsupported storage tier: {self.tier!r}")

    @staticmethod
    def _sort_key(
        candidate: EvictionCandidate,
    ) -> tuple[int, int, int, str]:
        """get sorted key"""
        return(
            0 if candidate.is_expired else 1,
            candidate.updated_at,
            candidate.created_at,
            candidate.session_id,
        )



class MemoryEvictionCoordinator(Protocol):
    """
    执行单Session安全内存淘汰的接口
    """
    async def evict_memory_candidate(
            self,
            session_id: str,
            expected_response_id: str,
            expected_updated_at: int,
            now: int,
            force: bool = False,
    ) -> MemoryEvictionResult:
        ...

    async def memory_used_bytes(self) -> int:
        ...

class MemoryEvictionExecutor:
    """
    Execute the selected
    memory eviction candidates
    """
    def __init__(
        self,
        coordinator: MemoryEvictionCoordinator,
    ) -> None:
        self._coordinator = coordinator

    async def execute(
        self,
        selection: EvictionSelection,
        context: EvictionExecutionContext,
        now: int,
        force: bool = False,
    ) -> MemoryEvictionBatchResult:
        results: list[MemoryEvictionResult] = []
        actual_free_bytes = 0
        estimated_used_bytes = context.initial_used_bytes
        stopped_after_reaching_target = False

        for candidate in selection.candidates:
            if (
                not candidate.is_expired
                and selection.target_free_bytes is not None
                and actual_free_bytes >= selection.target_free_bytes
            ):
                stopped_after_reaching_target = True
                break

            result = await self._evict_candidate(candidate, now, force)
            results.append(result)

            if result.evicted:
                actual_free_bytes += result.freed_bytes
                estimated_used_bytes = max(
                    0, estimated_used_bytes - result.freed_bytes
                )

        selected_count = len(selection.candidates)
        processed_count = len(results)
        final_used_bytes = await self._coordinator.memory_used_bytes()
        actual_target_satisfied = (
            None
            if context.target_used_bytes is None
            else final_used_bytes <= context.target_used_bytes
        )

        return MemoryEvictionBatchResult(
            trigger_reason=context.trigger_reason,
            scanned_count=context.scanned_count,
            results=tuple(results),
            selected_count=selected_count,
            processed_count=processed_count,
            remaining_candidate_count=selected_count - processed_count,
            actual_free_bytes=actual_free_bytes,
            initial_used_bytes=context.initial_used_bytes,
            estimated_final_used_bytes=estimated_used_bytes,
            final_used_bytes=final_used_bytes,
            target_free_bytes=selection.target_free_bytes,
            target_used_bytes=context.target_used_bytes,
            actual_target_satisfied=actual_target_satisfied,
            stopped_after_reaching_target=stopped_after_reaching_target,
            candidates_limit_reached=selection.candidates_limit_reached,
            byte_limit_reached=selection.byte_limit_reached,
        )


    async def _evict_candidate(
        self,
        candidate: EvictionCandidate,
        now: int,
        force: bool
    ) -> MemoryEvictionResult:
        """
        evict a single candidate from memory
        thorugh the TieredSessionStore
        """
        try:
            return await self._coordinator.evict_memory_candidate(
                session_id=candidate.session_id,
                expected_response_id=candidate.response_id,
                expected_updated_at=candidate.updated_at,
                now=now,
                force=force,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Failed to evict memory candidate for session %r",
                candidate.session_id
            )
            return MemoryEvictionResult(
                session_id=candidate.session_id,
                response_id=candidate.response_id,
                status=MemoryEvictionStatus.ERROR,
                freed_bytes=0,
            )


class DiskEvictionCoordinator(Protocol):
    async def evict_disk_candidate(
        self,
        session_id: str,
        expected_response_id: str,
    ) -> DiskEvictionResult:
        ...

    async def disk_used_bytes(self) -> int:
        ...


class DiskEvictionExecutor:
    def __init__(self, coordinator: DiskEvictionCoordinator) -> None:
        self._coordinator = coordinator

    async def execute(
        self,
        selection: EvictionSelection,
        context: EvictionExecutionContext,
    ) -> DiskEvictionBatchResult:
        results: list[DiskEvictionResult] = []
        actual_free_bytes = 0
        estimated_used_bytes = context.initial_used_bytes
        stopped_after_reaching_target = False

        for candidate in selection.candidates:
            if (
                not candidate.is_expired
                and selection.target_free_bytes is not None
                and actual_free_bytes >= selection.target_free_bytes
            ):
                stopped_after_reaching_target = True
                break

            try:
                result = await self._coordinator.evict_disk_candidate(
                    session_id=candidate.session_id,
                    expected_response_id=candidate.response_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Failed to evict disk candidate for session %r",
                    candidate.session_id,
                )
                result = DiskEvictionResult(
                    session_id=candidate.session_id,
                    response_id=candidate.response_id,
                    status=DiskEvictionStatus.ERROR,
                )

            results.append(result)
            if result.evicted:
                actual_free_bytes += result.freed_bytes
                estimated_used_bytes = max(
                    0, estimated_used_bytes - result.freed_bytes
                )

        selected_count = len(selection.candidates)
        processed_count = len(results)
        target_free_bytes = selection.target_free_bytes
        final_used_bytes = await self._coordinator.disk_used_bytes()
        actual_target_satisfied = (
            None
            if context.target_used_bytes is None
            else final_used_bytes <= context.target_used_bytes
        )

        return DiskEvictionBatchResult(
            trigger_reason=context.trigger_reason,
            scanned_count=context.scanned_count,
            results=tuple(results),
            selected_count=selected_count,
            processed_count=processed_count,
            remaining_candidate_count=selected_count - processed_count,
            actual_free_bytes=actual_free_bytes,
            initial_used_bytes=context.initial_used_bytes,
            estimated_final_used_bytes=estimated_used_bytes,
            final_used_bytes=final_used_bytes,
            target_free_bytes=target_free_bytes,
            target_used_bytes=context.target_used_bytes,
            actual_target_satisfied=actual_target_satisfied,
            stopped_after_reaching_target=stopped_after_reaching_target,
            candidates_limit_reached=selection.candidates_limit_reached,
            byte_limit_reached=selection.byte_limit_reached,
        )
