# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transactional, pre-input admission for synchronous state-only engines."""

import math
import time
from collections import Counter, deque
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import msgspec

from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import (
    EngineCoreRequest,
    StreamAdmissionContext,
    StreamAdmissionId,
    StreamAdmissionReason,
    StreamAdmissionResult,
    StreamAdmissionStatus,
    StreamUpdateRejection,
)
from vllm.v1.kv_cache_interface import StateSlotHandle
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler


@dataclass
class StreamAdmissionConfig:
    never_fed_timeout: float = 15.0
    parked_idle_timeout: float = 60.0
    draining_timeout: float = 40.0
    max_records: int = 4096
    max_owners: int = 64
    owner_epoch_timeout: float = 300.0

    def __post_init__(self) -> None:
        for value in (
            self.never_fed_timeout,
            self.parked_idle_timeout,
            self.draining_timeout,
            self.owner_epoch_timeout,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Stream deadlines must be finite and positive")
        for value in (self.max_records, self.max_owners):
            if type(value) is not int or value < 1:
                raise ValueError("Stream replay capacities must be positive integers")


@dataclass
class AdmissionRecord:
    owner: str
    admission_id: StreamAdmissionId
    fingerprint: bytes
    request_id: str
    accepted: bool
    status: StreamAdmissionStatus
    reason: StreamAdmissionReason | None = None
    handle: StateSlotHandle | None = None
    deadline: float | None = None
    deadline_reason: Literal["never_fed", "parked_idle", "draining"] | None = None
    input_closed: bool = False


@dataclass
class OwnerNamespace:
    epoch: str
    expires_at: float
    replay_floor: int = 0
    records: dict[int, AdmissionRecord] = field(default_factory=dict)


class StreamAdmissions:
    """Bounded owner namespaces with independent replay windows.

    Only renewal by the same owner or an engine-clock expiry discards its
    namespace. Unknown epochs never create records. Renewal recovers even from
    int64 sequence exhaustion; old sessions keep their original tags and handles.
    """

    def __init__(self, scheduler: "Scheduler", config: StreamAdmissionConfig):
        self.scheduler = scheduler
        self.config = config
        self.manager = next(
            iter(scheduler.kv_cache_manager.coordinator.state_managers.values())
        )
        self.clock = time.monotonic
        self.max_records = config.max_records
        self.owners: dict[str, OwnerNamespace] = {}
        self.sessions: dict[str, AdmissionRecord] = {}
        self.refusals: Counter[str] = Counter()
        self.expiries: Counter[str] = Counter()

    @staticmethod
    def _validate_owner(owner: str) -> None:
        if not isinstance(owner, str) or not 1 <= len(owner) <= 256:
            raise ValueError("Admission owner must contain 1 to 256 characters")

    def get_epoch(self, owner: str, renew: bool = False) -> str:
        """Acquire or explicitly renew an owner's replay namespace.

        A full registry refuses new owners until an existing namespace expires;
        it never evicts another owner's records in response to traffic pressure.
        Acquiring an existing epoch does not extend its fixed lifetime.
        """
        self._validate_owner(owner)
        now = self.clock()
        self.owners = {
            name: ns for name, ns in self.owners.items() if ns.expires_at > now
        }
        namespace = self.owners.get(owner)
        if namespace is not None and not renew:
            return namespace.epoch
        if namespace is None and len(self.owners) >= self.config.max_owners:
            raise ValueError("Owner namespace capacity; retry after epoch expiry")
        namespace = OwnerNamespace(uuid4().hex, now + self.config.owner_epoch_timeout)
        self.owners[owner] = namespace
        return namespace.epoch

    def admit(
        self,
        owner: str,
        admission_id: StreamAdmissionId,
        payload: EngineCoreRequest,
        *,
        reject_new_requests: bool = False,
    ) -> StreamAdmissionResult:
        """Replay recorded results before applying the current admission policy.

        A pause refusal is remembered; retry after resume with a fresh ID.
        """
        self._validate_owner(owner)
        if type(admission_id.sequence) is not int or not (
            0 <= admission_id.sequence < 2**63
        ):
            raise ValueError("Admission sequence must be a nonnegative int64")
        namespace = self.owners.get(owner)
        if (
            namespace is None
            or namespace.epoch != admission_id.owner_epoch
            or namespace.expires_at <= self.clock()
        ):
            return self._refuse(payload.request_id, "owner_epoch")
        if admission_id.sequence < namespace.replay_floor:
            return self._refuse(
                payload.request_id, "expired_id", namespace.replay_floor
            )

        self._validate(payload)
        fingerprint = sha256(
            msgspec.msgpack.encode(payload, order="deterministic")
        ).digest()
        key = admission_id.sequence
        if record := namespace.records.get(key):
            if record.fingerprint != fingerprint:
                raise ValueError("Conflicting payload for stream admission ID")
            return self._result(record, namespace.replay_floor)

        request_id = payload.request_id
        reason: StreamAdmissionReason | None = None
        if len(namespace.records) >= self.max_records and key <= min(namespace.records):
            reason = "expired_id"
        elif reject_new_requests:
            reason = "engine_paused"
        elif request_id in self.scheduler.requests or any(
            r.request_id == request_id for r in self.scheduler.state_retirements
        ):
            reason = "request_id_in_use"
        elif not self.manager.num_free_slots:
            reason = "capacity"
        elif len(self.scheduler.requests) >= self.scheduler.max_num_active_reqs:
            reason = "active_limit"
        record = AdmissionRecord(
            owner,
            admission_id,
            fingerprint,
            request_id,
            reason is None,
            "refused" if reason else "never_fed",
            reason,
        )
        # Only admission-local resources can mutate here: no preprocessing caches,
        # grammar tasks, connector callbacks, or worker dispatch.
        with ExitStack() as undo:
            undo.callback(
                self._restore_records,
                namespace,
                namespace.records.copy(),
                namespace.replay_floor,
            )
            if reason is None:
                request = Request.from_engine_core_request(deepcopy(payload), None)
                request.stream_admission = StreamAdmissionContext(owner, admission_id)
                record.handle = self.manager.allocate_for_admission(request_id, undo)
                assert record.handle is not None
                old_count = self.scheduler.num_waiting_for_streaming_input
                undo.callback(self._undo_insertion, request, old_count)
                self._insert_parked_request(request, record)
            self._remember(namespace, key, record)
            if reason:
                undo.callback(setattr, self, "refusals", self.refusals.copy())
                self.refusals[reason] += 1
            result = self._result(record, namespace.replay_floor)
            undo.pop_all()
        return result

    def _validate(self, payload: EngineCoreRequest) -> None:
        if (
            not isinstance(payload.request_id, str)
            or not 1 <= len(payload.request_id) <= 256
        ):
            raise ValueError("Admission request_id must contain 1 to 256 characters")
        if (
            not payload.resumable
            or payload.prompt_token_ids != []
            or payload.mm_features
            or payload.prompt_embeds is not None
            or payload.prompt_is_token_ids is not None
            or payload.pooling_params is not None
            or payload.sampling_params is None
            or payload.sampling_params.structured_outputs is not None
            or payload.sampling_params.n != 1
            or payload.sampling_params.output_kind == RequestOutputKind.FINAL_ONLY
            or payload.sampling_params.stop
            or payload.abort_immediately
            or payload.stream_admission is not None
            or payload.data_parallel_rank not in (None, 0)
            or payload.current_wave != 0
        ):
            raise ValueError("Admission requires an empty resumable sampling request")
        if (
            self.scheduler.vllm_config.is_mm_encoder_only
            or self.scheduler.vllm_config.model_config.is_diffusion
        ):
            raise ValueError("This model does not support resumable requests")
        if (
            self.scheduler.connector is not None
            or self.scheduler.ec_connector is not None
            or self.scheduler.aux_output_connector is not None
            or (payload.sampling_params.extra_args or {}).get("kv_transfer_params")
            or (payload.sampling_params.extra_args or {}).get("ec_transfer_params")
        ):
            raise ValueError("Stream admission does not support transfer connectors")

    def _insert_parked_request(self, request: Request, record: AdmissionRecord) -> None:
        request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
        request.streaming_queue = deque()
        self.scheduler._enqueue_waiting_request(request)
        self.scheduler.requests[request.request_id] = request
        self.scheduler.num_waiting_for_streaming_input += 1
        record.deadline = self.clock() + self.config.never_fed_timeout
        record.deadline_reason = "never_fed"
        self.sessions[request.request_id] = record

    def _undo_insertion(self, request: Request, old_count: int) -> None:
        self.scheduler.waiting.remove_requests([request])
        self.scheduler.kv_holding_waiting.remove_requests([request])
        self.scheduler.deferred_waiting.discard(request)
        self.scheduler.requests.pop(request.request_id, None)
        self.scheduler.num_waiting_for_streaming_input = old_count
        self.sessions.pop(request.request_id, None)

    def _restore_records(
        self, namespace: OwnerNamespace, records: dict[int, AdmissionRecord], floor: int
    ) -> None:
        namespace.records = records
        namespace.replay_floor = floor

    def _remember(
        self, namespace: OwnerNamespace, key: int, record: AdmissionRecord
    ) -> None:
        if record.reason == "expired_id":
            namespace.replay_floor = min(namespace.records) + 1
        else:
            namespace.records[key] = record
            if len(namespace.records) <= self.max_records:
                return
            namespace.replay_floor = min(namespace.records) + 1
        namespace.records = {
            k: v for k, v in namespace.records.items() if k >= namespace.replay_floor
        }

    def _result(
        self, record: AdmissionRecord, replay_floor: int
    ) -> StreamAdmissionResult:
        if record.status == "awaiting_fence" and not any(
            r.handle == record.handle for r in self.scheduler.state_retirements
        ):
            record.status = "finished"
        return StreamAdmissionResult(
            record.accepted,
            record.request_id,
            record.status,
            record.reason,
            replay_floor,
        )

    def _refuse(
        self, request_id: str, reason: StreamAdmissionReason, replay_floor: int = 0
    ) -> StreamAdmissionResult:
        self.refusals[reason] += 1
        return StreamAdmissionResult(False, request_id, "refused", reason, replay_floor)

    def accept_update(self, request: Request) -> StreamUpdateRejection | None:
        record = self.sessions.get(request.request_id)
        context = request.stream_admission
        if record is None:
            return "unknown_admission" if context is not None else None
        if (
            context is None
            or context.admission_id != record.admission_id
            or context.owner != record.owner
        ):
            return "identity_mismatch"
        if request.resumable:
            if record.input_closed:
                return "input_closed"
            if not request.prompt_token_ids:
                return "empty_update"
            record.status = "live"
            record.deadline = None
            record.deadline_reason = None
        if context.input_closed or not request.resumable:
            if not record.input_closed:
                record.input_closed = True
                record.deadline = self.clock() + self.config.draining_timeout
                record.deadline_reason = "draining"
            record.status = "draining"
        return None

    def park(self, request_id: str) -> None:
        if (record := self.sessions.get(request_id)) and not record.input_closed:
            record.status = "parked"
            record.deadline = self.clock() + self.config.parked_idle_timeout
            record.deadline_reason = "parked_idle"

    def finish(self, request_id: str) -> None:
        if record := self.sessions.pop(request_id, None):
            record.status = "awaiting_fence"
            record.deadline = None
            record.deadline_reason = None

    def next_deadline(self) -> float | None:
        return min(
            (r.deadline for r in self.sessions.values() if r.deadline is not None),
            default=None,
        )

    def expire(self) -> None:
        now = self.clock()
        for request_id, record in list(self.sessions.items()):
            if record.deadline is not None and record.deadline <= now:
                assert record.deadline_reason is not None
                self.expiries[record.deadline_reason] += 1
                for request in self.scheduler.finish_requests(
                    request_id, RequestStatus.FINISHED_ABORTED
                ):
                    self.scheduler._queue_state_error(request)

    def occupied(self) -> dict[str, int]:
        counts = dict.fromkeys(
            ("never_fed", "live", "parked", "draining", "awaiting_fence"), 0
        )
        for request in self.scheduler.requests.values():
            if self.manager.get(request.request_id) is not None:
                record = self.sessions.get(request.request_id)
                status = (
                    record.status
                    if record
                    else (
                        "parked"
                        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
                        else "live"
                    )
                )
                counts[status] += 1
        counts["awaiting_fence"] = len(self.scheduler.state_retirements)
        return counts
