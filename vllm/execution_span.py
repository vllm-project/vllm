# SPDX-License-Identifier: Apache-2.0
"""Source-owned execution-span identity and lifecycle helpers.

This module deliberately owns no collector or decoder state.  It creates
execution-span identities only in EngineCore and projects lifecycle events
through the typed primary USDT transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import os
import threading
from typing import Any

from vllm.primary_usdt import (
    PRIMARY_SEMANTIC_DISPATCH_ENABLED,
    PrimaryUSDTError,
    UINT64_MAX,
    emit_primary_usdt,
    engine_instance_id_from_config,
    primary_trace_session_id,
)

MAX_EXECUTION_SPAN_ID = UINT64_MAX >> 3
EXECUTION_SPAN_SCHEMA_VERSION = 0x0001_0001


def execution_span_enabled() -> bool:
    return os.environ.get("VLLM_PRIMARY_PROFILE", "primary_current") in {
        "execution_span_only",
        "primary_current_plus_execution_span",
    }


class ExecutionKind(IntEnum):
    STANDARD_GENERATE = 1


class SpanEventStatus(IntEnum):
    CREATED = 1
    DISPATCHED = 2
    COMPLETED = 3
    FAILED = 4


class RankCompletionSemantics(IntEnum):
    NONE = 0
    SYNC_OUTPUT_READY = 1
    ASYNC_OUTPUT_READY = 2


class TerminalStatus(IntEnum):
    COMPLETED = 1
    ABORTED_BEFORE_DISPATCH = 2
    FAILED_BEFORE_WORKER = 3
    FAILED_IN_WORKER = 4
    CONTEXT_MISMATCH = 5
    PARTIAL_WORKER_COMPLETION = 6
    NOT_EVALUABLE = 7


class CreationRejectionReason(IntEnum):
    MISSING_TRACE_SESSION = 1
    MISSING_ENGINE_INSTANCE = 2
    MISSING_SCHEDULER_STEP = 3
    MISSING_SCHEDULER_OUTPUT = 4
    MISSING_CANONICAL_MEMBERSHIP = 5
    UNSUPPORTED_EXECUTION_PATH = 6


class ContextMismatchReason(IntEnum):
    MISSING_CONTEXT = 1
    NONEXECUTABLE_WITH_CONTEXT = 2
    INVALID_CANONICAL_KEY = 3
    INVALID_PARENT_KEY = 4
    INVALID_ATTEMPT = 5


@dataclass(frozen=True)
class ExecutionSpanContext:
    schema_version: int
    trace_session_id_hi: int
    trace_session_id_lo: int
    engine_instance_id_hi: int
    engine_instance_id_lo: int
    execution_span_id: int
    scheduler_step_id: int
    scheduler_output_id: int
    execution_attempt_no: int
    dispatch_sequence: int
    total_num_scheduled_tokens: int
    execution_kind: int

    @property
    def engine_instance_id(self) -> tuple[int, int]:
        return self.engine_instance_id_hi, self.engine_instance_id_lo

    @property
    def canonical_key(self) -> tuple[int, int, int, int, int]:
        return (
            self.trace_session_id_hi,
            self.trace_session_id_lo,
            self.engine_instance_id_hi,
            self.engine_instance_id_lo,
            self.execution_span_id,
        )

    @property
    def parent_key(self) -> tuple[int, int, int, int]:
        return (
            self.trace_session_id_hi,
            self.trace_session_id_lo,
            self.scheduler_step_id,
            self.scheduler_output_id,
        )


def _event_instance_id(context: ExecutionSpanContext, ordinal: int) -> int:
    if ordinal <= 0 or ordinal > 7:
        raise ValueError("execution-span event ordinal outside frozen range")
    value = (context.execution_span_id << 3) | ordinal
    if value > UINT64_MAX:
        raise OverflowError("execution-span event instance identity exhausted")
    return value


def _common_tail(
    context: ExecutionSpanContext,
    *,
    ordinal: int,
    event_status: int,
) -> dict[str, int]:
    return {
        "execution_span_schema_version": EXECUTION_SPAN_SCHEMA_VERSION,
        "execution_span_id": context.execution_span_id,
        "event_instance_id": _event_instance_id(context, ordinal),
        "event_status": event_status,
        "scheduler_step_id": context.scheduler_step_id,
        "scheduler_output_id": context.scheduler_output_id,
        "execution_attempt_no": context.execution_attempt_no,
        "dispatch_sequence": context.dispatch_sequence,
        "total_num_scheduled_tokens": context.total_num_scheduled_tokens,
        "execution_kind": context.execution_kind,
    }


def emit_execution_span_created(context: ExecutionSpanContext) -> None:
    if not PRIMARY_SEMANTIC_DISPATCH_ENABLED:
        return
    emit_primary_usdt(
        "execution_span_created_v1",
        engine_instance_id=context.engine_instance_id,
        **_common_tail(
            context, ordinal=1, event_status=SpanEventStatus.CREATED
        ),
    )


def emit_execution_span_dispatched(context: ExecutionSpanContext) -> None:
    if not PRIMARY_SEMANTIC_DISPATCH_ENABLED:
        return
    emit_primary_usdt(
        "execution_span_dispatched_v1",
        engine_instance_id=context.engine_instance_id,
        **_common_tail(
            context, ordinal=2, event_status=SpanEventStatus.DISPATCHED
        ),
    )


def _stable_u64(text: str) -> int:
    value = int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big"
    )
    return value or 1


def worker_instance_id_hash(worker_rank: int) -> int:
    try:
        with open("/proc/self/stat", encoding="utf-8") as proc_stat:
            stat = proc_stat.read()
        closing_parenthesis = stat.rfind(")")
        start_ticks = stat[closing_parenthesis + 2 :].split()[19]
    except Exception:
        start_ticks = "unknown"
    return _stable_u64(f"{os.getpid()}:{start_ticks}:{worker_rank}")


def resolve_full_gpu_uuid(device_index: int) -> str:
    configured = os.environ.get("VLLM_PRIMARY_GPU_UUID", "").strip()
    if configured:
        return configured
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            value = pynvml.nvmlDeviceGetUUID(handle)
            return value.decode() if isinstance(value, bytes) else str(value)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        raise PrimaryUSDTError("full GPU UUID unavailable") from exc


_gpu_side_table_lock = threading.Lock()
_gpu_side_table_written: set[tuple[int, int, str]] = set()


def publish_gpu_uuid_side_table(
    *,
    context: ExecutionSpanContext,
    worker_rank: int,
    device_index: int,
    gpu_uuid: str,
) -> int:
    gpu_uuid_hash = _stable_u64(gpu_uuid)
    control_dir = os.environ.get("VLLM_PRIMARY_USDT_CONTROL_DIR")
    if not control_dir:
        return gpu_uuid_hash
    key = (os.getpid(), worker_rank, gpu_uuid)
    with _gpu_side_table_lock:
        if key in _gpu_side_table_written:
            return gpu_uuid_hash
        directory = os.path.join(control_dir, "gpu_uuid")
        os.makedirs(directory, mode=0o700, exist_ok=True)
        path = os.path.join(directory, f"{os.getpid()}-rank-{worker_rank}.json")
        temporary = f"{path}.tmp-{threading.get_ident()}"
        payload = {
            "schema_version": "execution_span.gpu_uuid_side_table.v1",
            "evidence_tier": "OUTCOME_OR_ENVIRONMENT",
            "semantic_backfill_allowed": False,
            "pid": os.getpid(),
            "worker_rank": worker_rank,
            "device_index": device_index,
            "gpu_uuid": gpu_uuid,
            "gpu_uuid_hash": gpu_uuid_hash,
            "trace_session_id_hi": context.trace_session_id_hi,
            "trace_session_id_lo": context.trace_session_id_lo,
            "engine_instance_id_hi": context.engine_instance_id_hi,
            "engine_instance_id_lo": context.engine_instance_id_lo,
        }
        with open(temporary, "w", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _gpu_side_table_written.add(key)
    return gpu_uuid_hash


def emit_execution_span_rank_begin(
    context: ExecutionSpanContext,
    *,
    worker_instance_id_hash: int,
    worker_rank: int,
    gpu_uuid_hash: int,
) -> None:
    if not PRIMARY_SEMANTIC_DISPATCH_ENABLED:
        return
    emit_primary_usdt(
        "execution_span_rank_begin_v1",
        engine_instance_id=context.engine_instance_id,
        **_common_tail(
            context, ordinal=3, event_status=SpanEventStatus.DISPATCHED
        ),
        worker_instance_id_hash=worker_instance_id_hash,
        worker_rank=worker_rank,
        gpu_uuid_hash=gpu_uuid_hash,
        completion_semantics=RankCompletionSemantics.NONE,
    )


def emit_execution_span_rank_end(
    context: ExecutionSpanContext,
    *,
    worker_instance_id_hash: int,
    worker_rank: int,
    gpu_uuid_hash: int,
    completion_semantics: RankCompletionSemantics,
    failed: bool = False,
) -> None:
    if not PRIMARY_SEMANTIC_DISPATCH_ENABLED:
        return
    emit_primary_usdt(
        "execution_span_rank_end_v1",
        engine_instance_id=context.engine_instance_id,
        **_common_tail(
            context,
            ordinal=4,
            event_status=SpanEventStatus.FAILED
            if failed
            else SpanEventStatus.COMPLETED,
        ),
        worker_instance_id_hash=worker_instance_id_hash,
        worker_rank=worker_rank,
        gpu_uuid_hash=gpu_uuid_hash,
        completion_semantics=completion_semantics,
    )


def bind_async_execution_span(
    output: Any,
    *,
    context: ExecutionSpanContext,
    worker_instance_hash: int,
    worker_rank: int,
    gpu_uuid_hash: int,
) -> None:
    output._execution_span_context = context
    output._execution_span_worker_instance_hash = worker_instance_hash
    output._execution_span_worker_rank = worker_rank
    output._execution_span_gpu_uuid_hash = gpu_uuid_hash
    output._execution_span_rank_end_emitted = False
    output._execution_span_rank_end_lock = threading.Lock()


def complete_async_execution_span_once(output: Any, *, failed: bool = False) -> bool:
    context = getattr(output, "_execution_span_context", None)
    if context is None:
        return False
    lock = output._execution_span_rank_end_lock
    with lock:
        if output._execution_span_rank_end_emitted:
            return False
        output._execution_span_rank_end_emitted = True
    emit_execution_span_rank_end(
        context,
        worker_instance_id_hash=output._execution_span_worker_instance_hash,
        worker_rank=output._execution_span_worker_rank,
        gpu_uuid_hash=output._execution_span_gpu_uuid_hash,
        completion_semantics=RankCompletionSemantics.ASYNC_OUTPUT_READY,
        failed=failed,
    )
    return True


def emit_execution_span_terminal(
    context: ExecutionSpanContext, status: TerminalStatus
) -> None:
    if not PRIMARY_SEMANTIC_DISPATCH_ENABLED:
        return
    emit_primary_usdt(
        "execution_span_terminal_v1",
        engine_instance_id=context.engine_instance_id,
        **_common_tail(
            context,
            ordinal=5,
            event_status=SpanEventStatus.COMPLETED
            if status == TerminalStatus.COMPLETED
            else SpanEventStatus.FAILED,
        ),
        terminal_status=status,
    )


def emit_creation_rejected(
    *,
    engine_instance_id: tuple[int, int],
    scheduler_step_id: int,
    scheduler_output_id: int,
    total_num_scheduled_tokens: int,
    reason: CreationRejectionReason,
) -> None:
    if not PRIMARY_SEMANTIC_DISPATCH_ENABLED:
        return
    if os.environ.get("VLLM_EXECUTION_SPAN_DIAGNOSTICS_ENABLE", "0") != "1":
        return
    emit_primary_usdt(
        "execution_span_creation_rejected_v1",
        engine_instance_id=engine_instance_id,
        critical=False,
        execution_span_schema_version=EXECUTION_SPAN_SCHEMA_VERSION,
        execution_span_id=0,
        event_instance_id=0,
        event_status=SpanEventStatus.FAILED,
        scheduler_step_id=scheduler_step_id,
        scheduler_output_id=scheduler_output_id,
        execution_attempt_no=0,
        dispatch_sequence=0,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        execution_kind=ExecutionKind.STANDARD_GENERATE,
        rejection_reason=reason,
    )


def emit_context_mismatch(
    context: ExecutionSpanContext, reason: ContextMismatchReason
) -> None:
    if not PRIMARY_SEMANTIC_DISPATCH_ENABLED:
        return
    if os.environ.get("VLLM_EXECUTION_SPAN_DIAGNOSTICS_ENABLE", "0") != "1":
        return
    emit_primary_usdt(
        "execution_span_context_mismatch_v1",
        engine_instance_id=context.engine_instance_id,
        **_common_tail(
            context, ordinal=6, event_status=SpanEventStatus.FAILED
        ),
        mismatch_reason=reason,
    )


class ExecutionSpanManager:
    """EngineCore-owned allocator and exactly-once terminal state."""

    def __init__(self, vllm_config: Any) -> None:
        self.engine_instance_id = (
            engine_instance_id_from_config(vllm_config)
            if PRIMARY_SEMANTIC_DISPATCH_ENABLED
            else (0, 0)
        )
        self.trace_session_id = (
            primary_trace_session_id()
            if PRIMARY_SEMANTIC_DISPATCH_ENABLED
            else (0, 0)
        )
        self._next_span_id = 1
        self._dispatch_sequence = 1
        self._attempts: dict[tuple[int, int], int] = {}
        self._terminal: dict[int, TerminalStatus] = {}
        self._lock = threading.Lock()

    def create_if_executable(
        self, scheduler_output: Any
    ) -> ExecutionSpanContext | None:
        if not PRIMARY_SEMANTIC_DISPATCH_ENABLED:
            return None
        if not execution_span_enabled():
            return None
        tokens = int(scheduler_output.total_num_scheduled_tokens)
        if tokens <= 0:
            return None
        step_id = int(getattr(scheduler_output, "primary_scheduler_step_id", UINT64_MAX))
        output_id = int(
            getattr(scheduler_output, "primary_scheduler_output_id", UINT64_MAX)
        )
        trace_hi, trace_lo = self.trace_session_id
        engine_hi, engine_lo = self.engine_instance_id
        reason: CreationRejectionReason | None = None
        if trace_hi == 0 and trace_lo == 0:
            reason = CreationRejectionReason.MISSING_TRACE_SESSION
        elif engine_hi == 0 and engine_lo == 0:
            reason = CreationRejectionReason.MISSING_ENGINE_INSTANCE
        elif step_id == UINT64_MAX:
            reason = CreationRejectionReason.MISSING_SCHEDULER_STEP
        elif output_id == UINT64_MAX:
            reason = CreationRejectionReason.MISSING_SCHEDULER_OUTPUT
        elif not getattr(scheduler_output, "num_scheduled_tokens", None):
            reason = CreationRejectionReason.MISSING_CANONICAL_MEMBERSHIP
        if reason is not None:
            emit_creation_rejected(
                engine_instance_id=self.engine_instance_id,
                scheduler_step_id=step_id,
                scheduler_output_id=output_id,
                total_num_scheduled_tokens=tokens,
                reason=reason,
            )
            return None
        parent = (step_id, output_id)
        with self._lock:
            if self._next_span_id > MAX_EXECUTION_SPAN_ID:
                raise OverflowError("execution-span sequence exhausted")
            attempt = self._attempts.get(parent, 0)
            self._attempts[parent] = attempt + 1
            context = ExecutionSpanContext(
                schema_version=EXECUTION_SPAN_SCHEMA_VERSION,
                trace_session_id_hi=trace_hi,
                trace_session_id_lo=trace_lo,
                engine_instance_id_hi=engine_hi,
                engine_instance_id_lo=engine_lo,
                execution_span_id=self._next_span_id,
                scheduler_step_id=step_id,
                scheduler_output_id=output_id,
                execution_attempt_no=attempt,
                dispatch_sequence=self._dispatch_sequence,
                total_num_scheduled_tokens=tokens,
                execution_kind=ExecutionKind.STANDARD_GENERATE,
            )
            self._next_span_id += 1
            self._dispatch_sequence += 1
        emit_execution_span_created(context)
        return context

    def terminal_once(
        self, context: ExecutionSpanContext, status: TerminalStatus
    ) -> bool:
        with self._lock:
            if context.execution_span_id in self._terminal:
                return False
            self._terminal[context.execution_span_id] = status
        emit_execution_span_terminal(context, status)
        return True


def validate_worker_context(scheduler_output: Any) -> ExecutionSpanContext | None:
    context = getattr(scheduler_output, "execution_span_context", None)
    executable = int(scheduler_output.total_num_scheduled_tokens) > 0
    if context is None:
        if executable and execution_span_enabled():
            raise PrimaryUSDTError("nonempty SchedulerOutput lacks execution span")
        return None
    if not isinstance(context, ExecutionSpanContext):
        raise PrimaryUSDTError("invalid execution-span context type")
    if not executable:
        emit_context_mismatch(
            context, ContextMismatchReason.NONEXECUTABLE_WITH_CONTEXT
        )
        raise PrimaryUSDTError("zero-token SchedulerOutput carries execution span")
    if (
        context.execution_span_id == 0
        or context.engine_instance_id == (0, 0)
        or context.trace_session_id_hi == context.trace_session_id_lo == 0
    ):
        emit_context_mismatch(context, ContextMismatchReason.INVALID_CANONICAL_KEY)
        raise PrimaryUSDTError("invalid execution-span canonical key")
    if (
        context.scheduler_step_id
        != int(scheduler_output.primary_scheduler_step_id)
        or context.scheduler_output_id
        != int(scheduler_output.primary_scheduler_output_id)
    ):
        emit_context_mismatch(context, ContextMismatchReason.INVALID_PARENT_KEY)
        raise PrimaryUSDTError("execution-span parent mismatch")
    return context
