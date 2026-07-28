# SPDX-License-Identifier: Apache-2.0
"""Typed, source-owned USDT projection for primary observability.

The installed libstapsdt ABI accepts at most six scalar arguments per probe.
Consequently a semantic v2 record is projected into fixed, event-specific
fragments.  A fragment carries the ABI version, source event sequence, three
payload scalars, and the number of valid payload scalars.  Probe name and
fragment ordinal define the field positions; PID is also present in the common
payload and is implicit in the USDT firing process.  Missing fragments are a
fail-closed incomplete record and must never be repaired from an auxiliary
transport.
"""

from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
import ctypes
from dataclasses import dataclass
import enum
from functools import wraps
import hashlib
import itertools
import json
import os
import secrets
import struct
import threading
import time
from typing import Any, Protocol

UINT64_MAX = (1 << 64) - 1
ABI_VERSION = 0x0002_0004
PROVIDER_NAME = "vllm_primary_observability"
MAX_PROVIDER_ARGUMENTS = 6
VALUES_PER_FRAGMENT = 3
AUTHORITATIVE_REGISTRY_SHA256 = (
    "e634be4b030b1e67e46145dbae7566c8bd4c4124c4101f0a8340d94dadf6f310"
)
DENOMINATOR_MAGIC = b"VLLMDEN1"
DENOMINATOR_VERSION = 1
DENOMINATOR_LOGICAL_ATTEMPT = 1
DENOMINATOR_FRAGMENT_ATTEMPT = 2
DENOMINATOR_FRAGMENT_FIRE_FAILURE = 3
DENOMINATOR_NO_FRAGMENT = 0xFFFF
DENOMINATOR_RECORD = struct.Struct("<8sIHHQQQQQQQQIIII")
assert DENOMINATOR_RECORD.size == 96

# O0 is a process-start contract. Cache it once so callsites can bypass the
# complete semantic-dispatch path before evaluating any event arguments.
# O1 keeps this enabled while VLLM_PRIMARY_USDT_ENABLE remains disabled.
PRIMARY_SEMANTIC_DISPATCH_ENABLED = (
    os.environ.get("VLLM_PRIMARY_SEMANTIC_DISPATCH_ENABLE", "1") == "1"
)

COMMON_FIELDS = (
    "abi_version",
    "trace_session_id_hi",
    "trace_session_id_lo",
    "engine_instance_id_hi",
    "engine_instance_id_lo",
    "source_process_id",
    "source_event_sequence",
    "monotonic_timestamp_ns",
    "request_id_hash",
    "integrity_flags",
)

EVENT_TAIL_FIELDS: dict[str, tuple[str, ...]] = {
    "request_arrival_v2": ("submission_attempt_id", "lifecycle_state"),
    "request_identity_assigned_v2": (
        "submission_attempt_id",
        "identity_assignment_status",
    ),
    "request_admitted_v2": ("submission_attempt_id", "lifecycle_state"),
    "request_output_progress_v2": (
        "output_ordinal",
        "output_token_count",
        "first_output",
        "final_output",
    ),
    "request_terminal_v2": (
        "terminal_reason",
        "terminal_class",
        "cleanup_required",
        "submission_attempt_id",
    ),
    "request_cleanup_v3": ("closure_reason", "closure_status"),
    "scheduler_step_begin_v2": (
        "scheduler_step_id",
        "running_queue_size",
        "waiting_queue_size",
    ),
    "scheduler_step_end_v2": (
        "scheduler_step_id",
        "scheduler_output_id",
        "scheduler_batch_id",
        "scheduled_request_count",
        "step_status",
    ),
    "scheduler_queue_snapshot_v2": (
        "scheduler_step_id",
        "scheduler_output_id",
        "scheduler_batch_id",
        "queue_snapshot_id",
        "queue_name",
        "expected_member_count",
        "emitted_member_count",
        "membership_complete",
    ),
    "scheduler_queue_member_v2": (
        "scheduler_step_id",
        "scheduler_output_id",
        "scheduler_batch_id",
        "queue_snapshot_id",
        "queue_name",
        "queue_position",
        "member_count",
        "request_state",
    ),
    "scheduler_output_v2": (
        "scheduler_step_id",
        "scheduler_output_id",
        "scheduler_batch_id",
        "request_count",
        "token_count",
        "membership_complete",
    ),
    "scheduler_output_member_v2": (
        "scheduler_step_id",
        "scheduler_output_id",
        "scheduler_batch_id",
        "member_index",
        "member_count",
        "scheduled_token_count",
        "request_state",
    ),
    "scheduler_state_transition_v2": (
        "scheduler_step_id",
        "scheduler_output_id",
        "scheduler_batch_id",
        "state_before",
        "state_after",
        "transition_reason",
    ),
    "kv_block_table_entry_v2": (
        "scheduler_step_id",
        "scheduler_output_id",
        "scheduler_batch_id",
        "physical_block_id",
        "block_generation",
        "logical_block_index",
        "kv_cache_group_id",
    ),
    "kv_block_alloc_v2": (
        "physical_block_id",
        "block_generation",
        "refcount_before",
        "refcount_after",
        "transition_reason",
        "cached_before",
    ),
    "kv_block_free_v2": (
        "physical_block_id",
        "block_generation",
        "refcount_before",
        "refcount_after",
        "release_reason",
        "cached_after",
    ),
    "kv_block_refcount_change_v2": (
        "physical_block_id",
        "block_generation",
        "refcount_before",
        "refcount_after",
        "mutation_kind",
    ),
    "kv_block_owner_change_v2": (
        "physical_block_id",
        "block_generation",
        "owner_present_before",
        "owner_present_after",
        "owner_count_before",
        "owner_count_after",
        "owner_change_reason",
        "refcount_after",
    ),
    "worker_request_mapping_v2": (
        "scheduler_step_id",
        "scheduler_output_id",
        "scheduler_batch_id",
        "worker_process_id",
        "worker_rank",
        "worker_request_index",
        "request_order_in_batch",
        "member_count",
    ),
    "worker_slot_mapping_begin_v2": (
        "scheduler_step_id",
        "scheduler_output_id",
        "scheduler_batch_id",
        "worker_process_id",
        "worker_rank",
        "mapping_id",
        "expected_total_entries",
        "fragment_count",
    ),
    "worker_slot_mapping_entry_v2": (
        "scheduler_output_id",
        "scheduler_batch_id",
        "worker_process_id",
        "worker_rank",
        "mapping_id",
        "fragment_index",
        "entry_index",
        "worker_request_index",
        "token_or_slot_index",
        "slot_index",
        "physical_block_id",
        "block_generation",
    ),
    "worker_slot_mapping_end_v2": (
        "scheduler_output_id",
        "mapping_id",
        "expected_total_entries",
        "emitted_total_entries",
        "mapping_complete",
        "failure_reason",
    ),
    "execution_span_created_v1": (
        "execution_span_schema_version",
        "execution_span_id",
        "event_instance_id",
        "event_status",
        "scheduler_step_id",
        "scheduler_output_id",
        "execution_attempt_no",
        "dispatch_sequence",
        "total_num_scheduled_tokens",
        "execution_kind",
    ),
    "execution_span_dispatched_v1": (
        "execution_span_schema_version",
        "execution_span_id",
        "event_instance_id",
        "event_status",
        "scheduler_step_id",
        "scheduler_output_id",
        "execution_attempt_no",
        "dispatch_sequence",
        "total_num_scheduled_tokens",
        "execution_kind",
    ),
    "execution_span_rank_begin_v1": (
        "execution_span_schema_version",
        "execution_span_id",
        "event_instance_id",
        "event_status",
        "scheduler_step_id",
        "scheduler_output_id",
        "execution_attempt_no",
        "dispatch_sequence",
        "total_num_scheduled_tokens",
        "execution_kind",
        "worker_instance_id_hash",
        "worker_rank",
        "gpu_uuid_hash",
        "completion_semantics",
    ),
    "execution_span_rank_end_v1": (
        "execution_span_schema_version",
        "execution_span_id",
        "event_instance_id",
        "event_status",
        "scheduler_step_id",
        "scheduler_output_id",
        "execution_attempt_no",
        "dispatch_sequence",
        "total_num_scheduled_tokens",
        "execution_kind",
        "worker_instance_id_hash",
        "worker_rank",
        "gpu_uuid_hash",
        "completion_semantics",
    ),
    "execution_span_terminal_v1": (
        "execution_span_schema_version",
        "execution_span_id",
        "event_instance_id",
        "event_status",
        "scheduler_step_id",
        "scheduler_output_id",
        "execution_attempt_no",
        "dispatch_sequence",
        "total_num_scheduled_tokens",
        "execution_kind",
        "terminal_status",
    ),
    "execution_span_creation_rejected_v1": (
        "execution_span_schema_version",
        "execution_span_id",
        "event_instance_id",
        "event_status",
        "scheduler_step_id",
        "scheduler_output_id",
        "execution_attempt_no",
        "dispatch_sequence",
        "total_num_scheduled_tokens",
        "execution_kind",
        "rejection_reason",
    ),
    "execution_span_context_mismatch_v1": (
        "execution_span_schema_version",
        "execution_span_id",
        "event_instance_id",
        "event_status",
        "scheduler_step_id",
        "scheduler_output_id",
        "execution_attempt_no",
        "dispatch_sequence",
        "total_num_scheduled_tokens",
        "execution_kind",
        "mismatch_reason",
    ),
    "scheduler_state_transition_v3": (
        "scheduler_step_id",
        "scheduler_output_id",
        "scheduler_batch_id",
        "state_before",
        "state_after",
        "transition_reason",
        "transition_initiator",
        "transition_action_id_hi",
        "transition_action_id_lo",
    ),
    "request_terminal_v3": (
        "terminal_state",
        "terminal_reason",
        "transition_reason",
        "transition_initiator",
        "transition_action_id_hi",
        "transition_action_id_lo",
        "cleanup_required",
        "submission_attempt_id",
    ),
}

EXECUTION_SPAN_EVENTS = frozenset(
    event_name
    for event_name in EVENT_TAIL_FIELDS
    if event_name.startswith("execution_span_")
)

PHYSICAL_PROBE_COUNT = sum(
    (len(COMMON_FIELDS) + len(tail_fields) + VALUES_PER_FRAGMENT - 1)
    // VALUES_PER_FRAGMENT
    for tail_fields in EVENT_TAIL_FIELDS.values()
)


class PrimaryUSDTError(RuntimeError):
    pass


class TransitionReason(enum.IntEnum):
    NORMAL_COMPLETION = 1
    RESOURCE_PREEMPTION = 2
    SCHEDULE_RESUME = 3
    EXPLICIT_ABORT = 4
    CLIENT_CANCEL = 5
    CONTROL_PLANE_ABORT = 6
    INTERNAL_ABORT = 7
    GRAMMAR_VALIDATION_ERROR = 8
    KV_LOAD_ERROR = 9


class TransitionInitiator(enum.IntEnum):
    MODEL_EXECUTION = 1
    SCHEDULER_POLICY = 2
    API_CALLER = 3
    CLIENT_RUNTIME = 4
    ENGINE_CONTROL_PLANE = 5
    OUTPUT_PROCESSOR = 6
    GRAMMAR_VALIDATOR = 7
    KV_CONNECTOR = 8


_TRANSITION_MATRIX = (
    ({5, 6}, {7, 8, 10, 12}, 1, 1),
    ({5}, {6}, 2, 2),
    ({6}, {5}, 3, 2),
    ({3}, {1, 6}, 3, 2),
    ({1, 2, 3, 4, 5, 6}, {9}, 4, 3),
    ({1, 2, 3, 4, 5, 6}, {9}, 5, 4),
    ({1, 2, 3, 4, 5, 6}, {9}, 6, 5),
    ({1, 2, 3, 4, 5, 6}, {9}, 7, 6),
    ({5, 6}, {11}, 8, 7),
    ({1, 2, 3, 4, 5, 6}, {11}, 9, 8),
)


def validate_transition_semantics(
    state_before: int,
    state_after: int,
    reason: int,
    initiator: int,
    action_id: tuple[int, int],
) -> None:
    if state_before == state_after:
        raise PrimaryUSDTError("Q2_TRANSITION_NO_CHANGE")
    if state_before not in range(1, 13) or state_after not in range(1, 13):
        raise PrimaryUSDTError("Q2_TRANSITION_UNKNOWN_STATE")
    try:
        TransitionReason(reason)
    except ValueError as exc:
        raise PrimaryUSDTError("Q2_TRANSITION_UNKNOWN_REASON") from exc
    try:
        TransitionInitiator(initiator)
    except ValueError as exc:
        raise PrimaryUSDTError("Q2_TRANSITION_UNKNOWN_INITIATOR") from exc
    if action_id == (0, 0):
        raise PrimaryUSDTError("Q2_TRANSITION_MISSING_ACTION_ID")
    if not any(
        state_before in before
        and state_after in after
        and reason == expected_reason
        and initiator == expected_initiator
        for before, after, expected_reason, expected_initiator in _TRANSITION_MATRIX
    ):
        raise PrimaryUSDTError("Q2_TRANSITION_IMPOSSIBLE_TUPLE")


_transition_action_lock = threading.Lock()
_transition_action_owner_pid = os.getpid()
_transition_action_ids: set[tuple[int, int]] = set()


def new_transition_action_id() -> tuple[int, int]:
    global _transition_action_owner_pid
    with _transition_action_lock:
        pid = os.getpid()
        if pid != _transition_action_owner_pid:
            _transition_action_owner_pid = pid
            _transition_action_ids.clear()
        while True:
            value = secrets.randbits(128)
            action = (value >> 64, value & UINT64_MAX)
            if action != (0, 0) and action not in _transition_action_ids:
                _transition_action_ids.add(action)
                return action


class SourceOwnedCounter:
    """Checked uint64 counter whose unavailable sentinel is never allocated."""

    def __init__(self, start: int = 0) -> None:
        if start < 0 or start >= UINT64_MAX:
            raise ValueError("counter start outside valid identity range")
        self._next = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            if self._next >= UINT64_MAX:
                raise OverflowError("source-owned counter exhausted")
            value = self._next
            self._next += 1
            return value


def next_block_generation(current: int) -> int:
    if current < 0 or current >= UINT64_MAX - 1:
        raise OverflowError("KV block generation exhausted")
    return current + 1


def validate_refcount_transition(before: int, after: int, mutation_kind: int) -> None:
    expected_delta = {1: 1, 2: 1, 3: -1}.get(mutation_kind)
    if expected_delta is None or after - before != expected_delta or after < 0:
        raise PrimaryUSDTError(
            f"invalid refcount transition {before}->{after} kind={mutation_kind}"
        )


def mapping_is_complete(
    expected_entries: int,
    emitted_entries: int,
    failure_reason: int,
) -> bool:
    return (
        expected_entries >= 0
        and emitted_entries == expected_entries
        and failure_reason == 0
    )


@dataclass(frozen=True)
class OwnerTransition:
    present_before: bool
    present_after: bool
    count_before: int
    count_after: int


class PrimaryOwnerLedger:
    """Source-owned many-request ownership edges by block generation."""

    def __init__(self) -> None:
        self._owners: dict[tuple[int, int], set[str]] = {}

    def add(self, key: tuple[int, int], request_id: str) -> OwnerTransition:
        owners = self._owners.setdefault(key, set())
        before = len(owners)
        if request_id in owners:
            raise PrimaryUSDTError("duplicate owner add")
        owners.add(request_id)
        return OwnerTransition(False, True, before, len(owners))

    def remove(self, key: tuple[int, int], request_id: str) -> OwnerTransition:
        owners = self._owners.get(key)
        before = len(owners) if owners is not None else 0
        if owners is None or request_id not in owners:
            raise PrimaryUSDTError("missing owner remove")
        owners.remove(request_id)
        after = len(owners)
        if not owners:
            self._owners.pop(key, None)
        return OwnerTransition(True, False, before, after)

    def owners(self, key: tuple[int, int]) -> frozenset[str]:
        return frozenset(self._owners.get(key, ()))


class _Backend(Protocol):
    def define(self, probe_name: str) -> None: ...

    def load(self) -> None: ...

    def fire(self, probe_name: str, arguments: tuple[int, ...]) -> None: ...

    def readiness(self) -> dict[str, Any]: ...


class CaptureBackend:
    """CPU-only fake backend used by contract tests."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.probes: set[str] = set()
        self.records: list[tuple[str, tuple[int, ...]]] = []
        self.fail_after = fail_after

    def define(self, probe_name: str) -> None:
        self.probes.add(probe_name)

    def load(self) -> None:
        return

    def fire(self, probe_name: str, arguments: tuple[int, ...]) -> None:
        if self.fail_after is not None and len(self.records) >= self.fail_after:
            raise PrimaryUSDTError("injected fragment emission failure")
        if probe_name not in self.probes:
            raise PrimaryUSDTError(f"undefined probe: {probe_name}")
        self.records.append((probe_name, arguments))

    def readiness(self) -> dict[str, Any]:
        return {
            "provider_path": "capture://in-memory",
            "provider_sha256": "CAPTURE_BACKEND",
        }


class _SDTProvider(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("probes", ctypes.c_void_p),
        ("errno", ctypes.c_int),
        ("error", ctypes.c_char_p),
        ("_handle", ctypes.c_void_p),
        ("_filename", ctypes.c_char_p),
        ("_memfd", ctypes.c_int),
        ("_use_memfd", ctypes.c_int),
    ]


class _LibStapSDTBackend:
    """Minimal ctypes binding to the installed six-argument libstapsdt ABI."""

    _UINT64_ARG_TYPE = 8

    def __init__(self) -> None:
        self._lib = ctypes.CDLL("libstapsdt.so")
        self._lib.providerInit.restype = ctypes.c_void_p
        self._lib.providerLoad.argtypes = [ctypes.c_void_p]
        self._lib.providerLoad.restype = ctypes.c_int
        self._lib.providerAddProbe.restype = ctypes.c_void_p
        self._lib.providerUseMemfd.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.providerUseMemfd.restype = ctypes.c_int
        self._provider = self._lib.providerInit(PROVIDER_NAME.encode())
        if not self._provider:
            raise PrimaryUSDTError("providerInit returned null")
        self._probes: dict[str, int] = {}
        self._provider_path: str | None = None

    def define(self, probe_name: str) -> None:
        probe = self._lib.providerAddProbe(
            ctypes.c_void_p(self._provider),
            probe_name.encode(),
            MAX_PROVIDER_ARGUMENTS,
            *([self._UINT64_ARG_TYPE] * MAX_PROVIDER_ARGUMENTS),
        )
        if not probe:
            raise PrimaryUSDTError(f"providerAddProbe failed: {probe_name}")
        self._probes[probe_name] = int(probe)

    def load(self) -> None:
        if self._lib.providerLoad(ctypes.c_void_p(self._provider)) != 0:
            raise PrimaryUSDTError("providerLoad failed")
        provider = ctypes.cast(
            ctypes.c_void_p(self._provider), ctypes.POINTER(_SDTProvider)
        ).contents
        if not provider._filename:
            raise PrimaryUSDTError("providerLoad did not expose provider path")
        self._provider_path = provider._filename.decode("utf-8", errors="strict")

    def fire(self, probe_name: str, arguments: tuple[int, ...]) -> None:
        if len(arguments) != MAX_PROVIDER_ARGUMENTS:
            raise PrimaryUSDTError("libstapsdt requires exactly six arguments")
        probe = self._probes[probe_name]
        self._lib.probeFire(
            ctypes.c_void_p(probe),
            *(ctypes.c_uint64(value) for value in arguments),
        )

    def readiness(self) -> dict[str, Any]:
        if not self._provider_path:
            raise PrimaryUSDTError("provider readiness requested before load")
        with open(self._provider_path, "rb") as provider_file:
            digest = hashlib.sha256(provider_file.read()).hexdigest()
        return {
            "provider_path": self._provider_path,
            "provider_sha256": digest,
        }


def _parse_128(value: str | None) -> tuple[int, int]:
    if not value:
        return 0, 0
    text = value.replace("-", "").strip()
    try:
        number = int(text, 16)
    except ValueError:
        return 0, 0
    if number < 0 or number >= 1 << 128:
        return 0, 0
    return number >> 64, number & UINT64_MAX


def primary_trace_session_id() -> tuple[int, int]:
    return _parse_128(os.environ.get("VLLM_PRIMARY_TRACE_SESSION_ID"))


def new_engine_instance_id() -> tuple[int, int]:
    number = 0
    while number == 0:
        number = secrets.randbits(128)
    return number >> 64, number & UINT64_MAX


def engine_instance_id_from_config(vllm_config: Any) -> tuple[int, int]:
    observability = getattr(vllm_config, "observability_config", None)
    return (
        int(getattr(observability, "primary_engine_instance_id_hi", 0)),
        int(getattr(observability, "primary_engine_instance_id_lo", 0)),
    )


_request_hash_lock = threading.Lock()
_request_hash_registry: dict[int, str] = {}
_lifecycle_lock = threading.Lock()
_submission_counters: dict[tuple[int, int], itertools.count] = {}
_submission_by_request: dict[tuple[tuple[int, int], int], int] = {}
_output_ordinal_by_request: dict[tuple[tuple[int, int], int], int] = {}
_terminal_requests: set[tuple[tuple[int, int], int]] = set()
_frontend_closed_requests: set[tuple[tuple[int, int], int]] = set()


@dataclass
class _SubmissionContext:
    engine_instance_id: tuple[int, int]
    submission_attempt_id: int
    request_id_hash: int = 0


_current_submission: ContextVar[_SubmissionContext | None] = ContextVar(
    "primary_submission", default=None
)


def primary_request_id_hash(request_id: Any | None) -> int:
    if request_id is None:
        return 0
    canonical = str(request_id)
    digest = hashlib.blake2b(
        canonical.encode("utf-8", errors="strict"), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "big")
    if value == 0:
        raise PrimaryUSDTError("request identity hashed to unavailable sentinel")
    with _request_hash_lock:
        prior = _request_hash_registry.setdefault(value, canonical)
        if prior != canonical:
            raise PrimaryUSDTError("request_id_hash collision")
    return value


def new_submission_attempt(engine_instance_id: tuple[int, int]) -> int:
    if engine_instance_id == (0, 0):
        raise PrimaryUSDTError("submission attempt lacks engine namespace")
    with _lifecycle_lock:
        counter = _submission_counters.setdefault(engine_instance_id, itertools.count())
        value = next(counter)
        if value >= UINT64_MAX:
            raise PrimaryUSDTError("submission attempt counter exhausted")
        return value


def bind_submission_attempt(
    engine_instance_id: tuple[int, int], request_id_hash: int, attempt_id: int
) -> None:
    key = (engine_instance_id, request_id_hash)
    with _lifecycle_lock:
        prior = _submission_by_request.setdefault(key, attempt_id)
        if prior != attempt_id:
            raise PrimaryUSDTError("request identity mapped to multiple submissions")


def submission_attempt_for_request(
    engine_instance_id: tuple[int, int], request_id_hash: int
) -> int:
    with _lifecycle_lock:
        return _submission_by_request.get(
            (engine_instance_id, request_id_hash), UINT64_MAX
        )


def next_output_ordinal(
    engine_instance_id: tuple[int, int], request_id_hash: int
) -> int:
    key = (engine_instance_id, request_id_hash)
    with _lifecycle_lock:
        value = _output_ordinal_by_request.get(key, 0)
        if value >= UINT64_MAX:
            raise PrimaryUSDTError("output ordinal exhausted")
        _output_ordinal_by_request[key] = value + 1
        return value


def mark_terminal_once(
    engine_instance_id: tuple[int, int], request_id_hash: int
) -> bool:
    key = (engine_instance_id, request_id_hash)
    with _lifecycle_lock:
        if key in _terminal_requests:
            return False
        _terminal_requests.add(key)
        return True


def terminal_was_marked(
    engine_instance_id: tuple[int, int], request_id_hash: int
) -> bool:
    with _lifecycle_lock:
        return (engine_instance_id, request_id_hash) in _terminal_requests


def mark_frontend_closed_once(
    engine_instance_id: tuple[int, int], request_id_hash: int
) -> bool:
    key = (engine_instance_id, request_id_hash)
    with _lifecycle_lock:
        if key in _frontend_closed_requests:
            return False
        _frontend_closed_requests.add(key)
        return True


def current_submission_attempt_id() -> int:
    context = _current_submission.get()
    if context is None:
        raise PrimaryUSDTError("no active submission context")
    return context.submission_attempt_id


def set_current_submission_request_hash(request_id_hash: int) -> None:
    context = _current_submission.get()
    if context is None:
        raise PrimaryUSDTError("no active submission context")
    context.request_id_hash = request_id_hash


def primary_async_submission(method):
    """Close every async pre-admission failure with an explicit terminal."""

    @wraps(method)
    async def wrapped(self, *args, **kwargs):
        if not PRIMARY_SEMANTIC_DISPATCH_ENABLED:
            return await method(self, *args, **kwargs)
        engine_instance_id = self._primary_engine_instance_id
        attempt_id = new_submission_attempt(engine_instance_id)
        context = _SubmissionContext(engine_instance_id, attempt_id)
        token = _current_submission.set(context)
        emit_primary_usdt(
            "request_arrival_v2",
            engine_instance_id=engine_instance_id,
            request_id_hash=0,
            submission_attempt_id=attempt_id,
            lifecycle_state=1,
        )
        try:
            return await method(self, *args, **kwargs)
        except Exception:
            should_emit = context.request_id_hash == 0 or mark_terminal_once(
                engine_instance_id, context.request_id_hash
            )
            if should_emit:
                emit_primary_usdt(
                    "request_terminal_v2",
                    engine_instance_id=engine_instance_id,
                    request_id_hash=context.request_id_hash,
                    terminal_reason=7,
                    terminal_class=3,
                    cleanup_required=0,
                    submission_attempt_id=attempt_id,
                )
            raise
        finally:
            _current_submission.reset(token)

    return wrapped


def _as_u64(name: str, value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, int):
        raise PrimaryUSDTError(f"{name} must be an integer scalar")
    if value < 0 or value > UINT64_MAX:
        raise PrimaryUSDTError(f"{name} outside uint64 range")
    return value


def fragment_layout(event_name: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    fields = COMMON_FIELDS + EVENT_TAIL_FIELDS[event_name]
    rows = []
    for index, start in enumerate(range(0, len(fields), VALUES_PER_FRAGMENT)):
        rows.append((f"{event_name}__f{index:02d}", fields[start : start + 3]))
    return tuple(rows)


LOGICAL_EVENT_IDS = {
    event_name: index
    for index, event_name in enumerate(EVENT_TAIL_FIELDS, start=1)
}
PHYSICAL_PROBE_IDS: dict[tuple[str, int], int] = {}
_next_physical_probe_id = 1
for _event_name in EVENT_TAIL_FIELDS:
    for _fragment_index, _ in enumerate(fragment_layout(_event_name)):
        PHYSICAL_PROBE_IDS[(_event_name, _fragment_index)] = _next_physical_probe_id
        _next_physical_probe_id += 1
assert _next_physical_probe_id - 1 == PHYSICAL_PROBE_COUNT


def _stable_u64(text: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8", errors="strict"), digest_size=8).digest(),
        "big",
    )


def _process_start_time_ticks() -> int:
    with open("/proc/self/stat", encoding="utf-8") as proc_stat:
        stat = proc_stat.read()
    closing_parenthesis = stat.rfind(")")
    if closing_parenthesis < 0:
        raise PrimaryUSDTError("cannot parse /proc/self/stat")
    fields_after_comm = stat[closing_parenthesis + 2 :].split()
    return int(fields_after_comm[19])


class SourceDenominatorWriter:
    """Fixed binary loss denominator, independent from semantic transport."""

    def __init__(self, directory: str, process_role: str) -> None:
        self.directory = directory
        self.process_role = process_role
        self.pid = os.getpid()
        os.makedirs(directory, mode=0o700, exist_ok=True)
        self.path = os.path.join(directory, f"{self.pid}.source-denominator.bin")
        self.fd = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC,
            0o600,
        )
        run_hi, run_lo = _parse_128(os.environ.get("VLLM_PRIMARY_RUN_ID"))
        if run_hi == 0 and run_lo == 0:
            raise PrimaryUSDTError("source denominator requires nonzero run ID")
        attempt_id = os.environ.get("VLLM_PRIMARY_ATTEMPT_ID", "")
        if not attempt_id:
            raise PrimaryUSDTError("source denominator requires attempt ID")
        self.run_id = (run_hi, run_lo)
        self.attempt_id_hash = _stable_u64(attempt_id)
        self.process_role_hash = _stable_u64(process_role)

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def record(
        self,
        kind: int,
        *,
        source_event_sequence: int,
        logical_event_id: int,
        physical_probe_id: int = 0,
        fragment_index: int = DENOMINATOR_NO_FRAGMENT,
        failure_code: int = 0,
    ) -> None:
        if os.getpid() != self.pid:
            raise PrimaryUSDTError("source denominator crossed a process boundary")
        record = DENOMINATOR_RECORD.pack(
            DENOMINATOR_MAGIC,
            DENOMINATOR_VERSION,
            kind,
            fragment_index,
            ABI_VERSION,
            self.run_id[0],
            self.run_id[1],
            self.attempt_id_hash,
            self.process_role_hash,
            self.pid,
            time.monotonic_ns(),
            source_event_sequence,
            logical_event_id,
            physical_probe_id,
            failure_code,
            0,
        )
        written = os.write(self.fd, record)
        if written != len(record):
            raise PrimaryUSDTError("short source denominator write")


@dataclass(frozen=True)
class ProviderReadiness:
    ready: bool
    pid: int
    process_start_time_ticks: int
    roles: tuple[str, ...]
    provider_path: str
    provider_sha256: str
    denominator_path: str | None
    eager_loaded_before_first_semantic_event: bool


@dataclass(frozen=True)
class EmissionResult:
    event_name: str
    source_event_sequence: int
    expected_fragments: int
    emitted_fragments: int
    complete: bool
    failure: str | None = None


class PrimaryUSDTEmitter:
    def __init__(
        self,
        backend: _Backend | None = None,
        *,
        enabled: bool | None = None,
        strict_failure: bool | None = None,
    ) -> None:
        self.enabled = (
            os.environ.get("VLLM_PRIMARY_USDT_ENABLE", "0") == "1"
            if enabled is None
            else enabled
        )
        self.strict_failure = (
            os.environ.get("VLLM_PRIMARY_USDT_STRICT_FAILURE", "0") == "1"
            if strict_failure is None
            else strict_failure
        )
        self._backend = backend
        self._loaded = False
        self._sequence = itertools.count(1)
        self._lock = threading.Lock()
        self.failures: Counter[str] = Counter()
        self._owner_pid = os.getpid()
        self._roles: set[str] = set()
        self._readiness_published = False
        self._logical_events_attempted = 0
        self._denominator: SourceDenominatorWriter | None = None

    def _reset_after_process_boundary(self) -> None:
        pid = os.getpid()
        if pid == self._owner_pid:
            return
        if self._denominator is not None:
            self._denominator.close()
        self._owner_pid = pid
        self._backend = None
        self._loaded = False
        self._sequence = itertools.count(1)
        self._roles.clear()
        self._readiness_published = False
        self._logical_events_attempted = 0
        self._denominator = None

    def _ensure_loaded(self) -> None:
        if self._loaded or not self.enabled:
            return
        backend = self._backend or _LibStapSDTBackend()
        for event_name in EVENT_TAIL_FIELDS:
            for probe_name, _ in fragment_layout(event_name):
                backend.define(probe_name)
        backend.load()
        self._backend = backend
        self._loaded = True

    def _ensure_denominator(self, process_role: str) -> None:
        directory = os.environ.get("VLLM_PRIMARY_USDT_DENOMINATOR_DIR")
        required = os.environ.get("VLLM_PRIMARY_USDT_REQUIRE_DENOMINATOR", "0") == "1"
        if not directory:
            if required:
                raise PrimaryUSDTError("required source denominator directory missing")
            return
        if self._denominator is None:
            self._denominator = SourceDenominatorWriter(directory, process_role)

    def _write_readiness(self, readiness: ProviderReadiness) -> None:
        control_dir = os.environ.get("VLLM_PRIMARY_USDT_CONTROL_DIR")
        required = os.environ.get("VLLM_PRIMARY_USDT_REQUIRE_READINESS", "0") == "1"
        if not control_dir:
            if required:
                raise PrimaryUSDTError("required provider readiness directory missing")
            return
        readiness_dir = os.path.join(control_dir, "readiness")
        os.makedirs(readiness_dir, mode=0o700, exist_ok=True)
        path = os.path.join(readiness_dir, f"{readiness.pid}.json")
        temporary = f"{path}.tmp-{threading.get_ident()}"
        with open(__file__, "rb") as source_file:
            source_sha256 = hashlib.sha256(source_file.read()).hexdigest()
        payload = {
            "schema_version": "vllm.primary_usdt.provider_readiness.v1",
            "evidence_tier": "AUXILIARY_SIDECAR_CONTROL",
            "semantic_backfill_allowed": False,
            "ready": readiness.ready,
            "pid": readiness.pid,
            "ppid": os.getppid(),
            "process_start_time_ticks": readiness.process_start_time_ticks,
            "roles": list(readiness.roles),
            "provider": PROVIDER_NAME,
            "provider_path": readiness.provider_path,
            "provider_sha256": readiness.provider_sha256,
            "abi_version": ABI_VERSION,
            "logical_event_count": len(EVENT_TAIL_FIELDS),
            "physical_probe_count": PHYSICAL_PROBE_COUNT,
            "maximum_fragment_count": max(
                len(fragment_layout(event_name)) for event_name in EVENT_TAIL_FIELDS
            ),
            "registry_sha256": AUTHORITATIVE_REGISTRY_SHA256,
            "source_sha256": source_sha256,
            "denominator_path": readiness.denominator_path,
            "denominator_record_bytes": DENOMINATOR_RECORD.size,
            "eager_loaded_before_first_semantic_event": (
                readiness.eager_loaded_before_first_semantic_event
            ),
            "semantic_events_attempted_at_handshake": self._logical_events_attempted,
        }
        with open(temporary, "w", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)

    def prepare(self, process_role: str) -> ProviderReadiness:
        if not process_role or any(character.isspace() for character in process_role):
            raise PrimaryUSDTError("process role must be a nonempty token")
        with self._lock:
            self._reset_after_process_boundary()
            self._roles.add(process_role)
            self._readiness_published = False
            if not self.enabled:
                return ProviderReadiness(
                    False,
                    os.getpid(),
                    _process_start_time_ticks(),
                    tuple(sorted(self._roles)),
                    "",
                    "",
                    None,
                    self._logical_events_attempted == 0,
                )
            self._ensure_denominator(process_role)
            self._ensure_loaded()
            assert self._backend is not None
            backend_readiness = self._backend.readiness()
            readiness = ProviderReadiness(
                True,
                os.getpid(),
                _process_start_time_ticks(),
                tuple(sorted(self._roles)),
                str(backend_readiness["provider_path"]),
                str(backend_readiness["provider_sha256"]),
                self._denominator.path if self._denominator else None,
                self._logical_events_attempted == 0,
            )
            self._write_readiness(readiness)
            self._readiness_published = True
            return readiness

    def emit(
        self,
        event_name: str,
        *,
        engine_instance_id: tuple[int, int],
        request_id_hash: int = 0,
        integrity_flags: int = 0,
        critical: bool = True,
        **tail: int,
    ) -> EmissionResult:
        if event_name not in EVENT_TAIL_FIELDS:
            raise PrimaryUSDTError(f"unknown typed event: {event_name}")
        profile = os.environ.get(
            "VLLM_PRIMARY_PROFILE", "primary_current"
        )
        is_execution_span = event_name in EXECUTION_SPAN_EVENTS
        enabled_by_profile = (
            profile == "primary_current_plus_execution_span"
            or (profile == "primary_current" and not is_execution_span)
            or (profile == "execution_span_only" and is_execution_span)
        )
        if not enabled_by_profile:
            return EmissionResult(
                event_name,
                0,
                len(fragment_layout(event_name)),
                0,
                True,
            )
        expected_tail = EVENT_TAIL_FIELDS[event_name]
        missing = set(expected_tail) - set(tail)
        extra = set(tail) - set(expected_tail)
        if missing or extra:
            raise PrimaryUSDTError(
                f"{event_name} tail mismatch missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )
        engine_hi, engine_lo = engine_instance_id
        if critical and engine_hi == 0 and engine_lo == 0:
            raise PrimaryUSDTError("critical event lacks engine_instance_id")
        with self._lock:
            self._reset_after_process_boundary()
            if (
                self.enabled
                and os.environ.get("VLLM_PRIMARY_USDT_REQUIRE_READINESS", "0") == "1"
                and (
                    not self._loaded
                    or not self._roles
                    or not self._readiness_published
                )
            ):
                raise PrimaryUSDTError("semantic event attempted before readiness")
            sequence = next(self._sequence)
            if sequence >= UINT64_MAX:
                raise PrimaryUSDTError("source event sequence exhausted")
            self._ensure_loaded()
            trace_hi, trace_lo = _parse_128(
                os.environ.get("VLLM_PRIMARY_TRACE_SESSION_ID")
            )
            common = {
                "abi_version": ABI_VERSION,
                "trace_session_id_hi": trace_hi,
                "trace_session_id_lo": trace_lo,
                "engine_instance_id_hi": engine_hi,
                "engine_instance_id_lo": engine_lo,
                "source_process_id": os.getpid(),
                "source_event_sequence": sequence,
                "monotonic_timestamp_ns": time.monotonic_ns(),
                "request_id_hash": request_id_hash,
                "integrity_flags": integrity_flags,
            }
            values = tuple(
                _as_u64(name, common[name]) for name in COMMON_FIELDS
            ) + tuple(_as_u64(name, tail[name]) for name in expected_tail)
            layout = fragment_layout(event_name)
            if not self.enabled:
                return EmissionResult(event_name, sequence, len(layout), 0, True)
            process_role = next(iter(sorted(self._roles)), "unprepared")
            self._ensure_denominator(process_role)
            assert self._backend is not None
            logical_event_id = LOGICAL_EVENT_IDS[event_name]
            if self._denominator is not None:
                self._denominator.record(
                    DENOMINATOR_LOGICAL_ATTEMPT,
                    source_event_sequence=sequence,
                    logical_event_id=logical_event_id,
                )
            self._logical_events_attempted += 1
            emitted = 0
            try:
                for index, (probe_name, names) in enumerate(layout):
                    start = index * VALUES_PER_FRAGMENT
                    chunk = values[start : start + VALUES_PER_FRAGMENT]
                    padded = chunk + (0,) * (VALUES_PER_FRAGMENT - len(chunk))
                    arguments = (ABI_VERSION, sequence, *padded, len(names))
                    physical_probe_id = PHYSICAL_PROBE_IDS[(event_name, index)]
                    if self._denominator is not None:
                        self._denominator.record(
                            DENOMINATOR_FRAGMENT_ATTEMPT,
                            source_event_sequence=sequence,
                            logical_event_id=logical_event_id,
                            physical_probe_id=physical_probe_id,
                            fragment_index=index,
                        )
                    self._backend.fire(probe_name, arguments)
                    emitted += 1
            except Exception as exc:
                if self._denominator is not None:
                    self._denominator.record(
                        DENOMINATOR_FRAGMENT_FIRE_FAILURE,
                        source_event_sequence=sequence,
                        logical_event_id=logical_event_id,
                        physical_probe_id=PHYSICAL_PROBE_IDS[(event_name, emitted)],
                        fragment_index=emitted,
                        failure_code=1,
                    )
                self.failures[event_name] += 1
                result = EmissionResult(
                    event_name,
                    sequence,
                    len(layout),
                    emitted,
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
                if critical and self.strict_failure:
                    raise PrimaryUSDTError(result.failure) from exc
                return result
            return EmissionResult(event_name, sequence, len(layout), emitted, True)


_GLOBAL_EMITTER = PrimaryUSDTEmitter()


def emit_primary_usdt(event_name: str, **kwargs: Any) -> EmissionResult:
    return _GLOBAL_EMITTER.emit(event_name, **kwargs)


def prepare_primary_usdt_provider(process_role: str) -> ProviderReadiness:
    """Load every physical probe and publish control readiness before events."""

    return _GLOBAL_EMITTER.prepare(process_role)


def primary_usdt_failure_counts() -> dict[str, int]:
    return dict(_GLOBAL_EMITTER.failures)
