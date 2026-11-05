# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transactional, pre-input admission for synchronous state-only engines.

A request ID names one admitted session. A retry with the same ID and payload
returns the live session's result; once a session ends, its ID is not reused.
"""

import math
import time
from collections import Counter, deque
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Literal

import msgspec

from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import (
    EngineCoreRequest,
    StreamAdmissionContext,
    StreamAdmissionReason,
    StreamAdmissionResult,
    StreamAdmissionStatus,
    StreamUpdateRejection,
)
from vllm.v1.kv_cache_interface import StateSlotHandle
from vllm.v1.request import Request, RequestStatus, StreamingUpdate

if TYPE_CHECKING:
    from vllm.v1.core.sched.state_scheduler import StateScheduler


@dataclass
class StreamAdmissionConfig:
    never_fed_timeout: float = 15.0
    parked_idle_timeout: float = 60.0
    draining_timeout: float = 40.0

    def __post_init__(self) -> None:
        for value in (
            self.never_fed_timeout,
            self.parked_idle_timeout,
            self.draining_timeout,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Stream deadlines must be finite and positive")


@dataclass
class AdmissionRecord:
    """One admitted session, from admission until its request finishes."""

    fingerprint: bytes
    handle: StateSlotHandle
    status: StreamAdmissionStatus = "never_fed"
    deadline: float | None = None
    deadline_reason: Literal["never_fed", "parked_idle", "draining"] | None = None
    input_closed: bool = False


class StreamAdmissions:
    """Admitted sessions of a StateScheduler and their deadlines."""

    def __init__(self, scheduler: "StateScheduler", config: StreamAdmissionConfig):
        self.scheduler = scheduler
        self.config = config
        self.manager = next(
            iter(scheduler.kv_cache_manager.coordinator.state_managers.values())
        )
        self.clock = time.monotonic
        self.sessions: dict[str, AdmissionRecord] = {}
        self.refusals: Counter[str] = Counter()
        self.expiries: Counter[str] = Counter()

    def admit(
        self, payload: EngineCoreRequest, *, reject_new_requests: bool = False
    ) -> StreamAdmissionResult:
        """Reserve a slot and park an empty resumable request, atomically.

        A retry matching a live session's ID and payload returns its current
        result without moving its deadline.
        """
        self._validate(payload)
        # Frontends send no prompt, in-process callers an empty one.
        payload = msgspec.structs.replace(payload, prompt_token_ids=[])
        fingerprint = sha256(
            msgspec.msgpack.encode(payload, order="deterministic")
        ).digest()
        request_id = payload.request_id
        if (record := self.sessions.get(request_id)) is not None:
            if record.fingerprint != fingerprint:
                return self._refuse(request_id, "request_id_in_use")
            return StreamAdmissionResult(True, request_id, record.status)
        if reject_new_requests:
            return self._refuse(request_id, "engine_paused")
        if request_id in self.scheduler.requests or any(
            r.request_id == request_id for r in self.scheduler.state_retirements
        ):
            return self._refuse(request_id, "request_id_in_use")
        if not self.manager.num_free_slots:
            return self._refuse(request_id, "capacity")
        if len(self.scheduler.requests) >= self.scheduler.max_num_active_reqs:
            return self._refuse(request_id, "active_limit")
        # Only admission-local resources can mutate here: no preprocessing caches,
        # grammar tasks, connector callbacks, or worker dispatch.
        with ExitStack() as undo:
            request = Request.from_engine_core_request(deepcopy(payload), None)
            request.stream_admission = StreamAdmissionContext()
            handle = self.manager.allocate_for_admission(request_id, undo)
            assert handle is not None
            old_count = self.scheduler.num_waiting_for_streaming_input
            undo.callback(self._undo_insertion, request, old_count)
            self._insert_parked_request(request, AdmissionRecord(fingerprint, handle))
            undo.pop_all()
        return StreamAdmissionResult(True, request_id, "never_fed")

    def _validate(self, payload: EngineCoreRequest) -> None:
        if (
            not isinstance(payload.request_id, str)
            or not 1 <= len(payload.request_id) <= 256
        ):
            raise ValueError("Admission request_id must contain 1 to 256 characters")
        if (
            not payload.resumable
            or (
                payload.prompt_token_ids is not None
                and len(payload.prompt_token_ids) > 0
            )
            or payload.mm_features
            or self.unsupported_input(payload)
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
        ):
            raise ValueError("Stream admission does not support transfer connectors")

    @staticmethod
    def unsupported_input(request: EngineCoreRequest | Request) -> bool:
        """Whether a stream input uses features a state stream cannot carry."""
        return (
            request.pooling_params is not None
            or request.prompt_embeds is not None
            or request.prompt_is_token_ids is not None
            or StreamAdmissions._unsupported_params(request.sampling_params)
        )

    def misplaced_multimodal_input(
        self, request: EngineCoreRequest | Request | StreamingUpdate
    ) -> bool:
        """Whether a multimodal item falls outside the input's own prompt.

        Each stream input replaces the prompt, so positions are relative to it.
        A decoder-only item must fit within the prompt, or the scheduler could
        never cover it whole. An encoder-decoder item is encoder input whose
        length is independent of the decoder prompt, so only its start counts.
        """
        # A frontend's input carries an int32 ndarray, an in-process one a list.
        prompt = request.prompt_token_ids
        num_tokens = 0 if prompt is None else len(prompt)
        for feature in request.mm_features or ():
            position = feature.mm_position
            if not 0 <= position.offset < num_tokens:
                return True
            if (
                not self.scheduler.is_encoder_decoder
                and position.offset + position.length > num_tokens
            ):
                return True
        return False

    @staticmethod
    def _unsupported_params(params: SamplingParams | None) -> bool:
        """Sampling features a state stream cannot carry across inputs."""
        extra_args = (params.extra_args or {}) if params is not None else {}
        return (
            params is None
            or params.structured_outputs is not None
            or params.n != 1
            or params.output_kind == RequestOutputKind.FINAL_ONLY
            or bool(params.stop)
            or bool(extra_args.get("kv_transfer_params"))
            or bool(extra_args.get("ec_transfer_params"))
        )

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

    def _refuse(
        self, request_id: str, reason: StreamAdmissionReason
    ) -> StreamAdmissionResult:
        self.refusals[reason] += 1
        return StreamAdmissionResult(False, request_id, "refused", reason)

    def accept_update(self, request: Request) -> StreamUpdateRejection | None:
        record = self.sessions.get(request.request_id)
        context = request.stream_admission
        if record is None:
            if context is not None:
                return "unknown_admission"
            # A stream that was never admitted replaces its prompt the same way.
            if (
                request.resumable
                and request.request_id in self.scheduler.requests
                and self.misplaced_multimodal_input(request)
            ):
                return "validation_error"
            return None
        if context is None:
            return "identity_mismatch"
        if self.unsupported_input(request):
            return "unsupported_update"
        if request.resumable:
            if record.input_closed:
                return "input_closed"
            if not request.prompt_token_ids:
                return "empty_update"
            # Every refusal precedes the first mutation; the replacement itself
            # is applied later and must not be able to fail.
            if self.misplaced_multimodal_input(request):
                return "validation_error"
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
            ("never_fed", "live", "parked", "draining", "retiring"), 0
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
        counts["retiring"] = len(self.scheduler.state_retirements)
        return counts
