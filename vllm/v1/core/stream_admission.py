# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream admission for a StateScheduler: admitted sessions and their deadlines.

See docs/design/persistent_state.md, "Stream admission".
"""

from collections import Counter, deque
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Literal

import msgspec

from vllm.config import SchedulerConfig
from vllm.sampling_params import RequestOutputKind
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import (
    EngineCoreRequest,
    StreamAdmissionContext,
    StreamAdmissionReason,
    StreamAdmissionResult,
    StreamUpdateRejection,
)
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.sched.state_scheduler import StateScheduler

SessionStatus = Literal["never_fed", "live", "parked", "draining"]
DeadlineName = Literal["never_fed", "parked_idle", "draining"]


@dataclass
class StreamSession:
    """One admitted session, from admission until its request finishes."""

    status: SessionStatus = "never_fed"
    deadline: float | None = None
    deadline_name: DeadlineName | None = None
    input_closed: bool = False


class StreamAdmissions:
    """Admitted sessions of a StateScheduler and their deadlines."""

    def __init__(self, scheduler: "StateScheduler"):
        self.scheduler = scheduler
        self.manager = next(iter(scheduler.state_managers.values()))
        self.timeouts = {
            **SchedulerConfig.DEFAULT_STREAM_ADMISSION_TIMEOUTS,
            **scheduler.scheduler_config.stream_admission_timeouts,
        }
        self.sessions: dict[str, StreamSession] = {}
        self.refusals: Counter[str] = Counter()
        self.expiries: Counter[str] = Counter()

    def admit(self, payload: EngineCoreRequest) -> StreamAdmissionResult:
        """Reserve a slot and park an empty resumable request under the
        payload's request ID, or refuse without changing anything."""
        scheduler = self.scheduler
        request_id = payload.request_id
        # An ID in use is refused whatever the request's other fields. It
        # stays in use until its stream's retirement notice is delivered.
        if (
            request_id in scheduler.requests
            or any(e.request_id == request_id for e in scheduler.state_retirements)
            or any(request_id in ids for ids in scheduler.retired_streams.values())
        ):
            return self._refuse(request_id, "request_id_in_use")
        self._validate(payload)
        if scheduler.pause_state != PauseState.UNPAUSED:
            return self._refuse(request_id, "engine_paused")
        if not self.manager.num_free_slots:
            return self._refuse(request_id, "capacity")
        if len(scheduler.requests) >= scheduler.max_num_active_reqs:
            return self._refuse(request_id, "active_limit")
        # Frontends send no prompt, in-process callers may send an empty one.
        # Every step that can fail precedes the first change.
        request = Request.from_engine_core_request(
            msgspec.structs.replace(
                payload, prompt_token_ids=[], stream_admission=StreamAdmissionContext()
            ),
            None,
        )
        request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
        request.streaming_queue = deque()
        self.manager.allocate_new_blocks(request_id, 0, 0)
        scheduler._enqueue_waiting_request(request)
        scheduler.requests[request_id] = request
        scheduler.num_waiting_for_streaming_input += 1
        session = self.sessions[request_id] = StreamSession()
        self.start_deadline(session, "never_fed")
        return StreamAdmissionResult(True, request_id)

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
        scheduler = self.scheduler
        if (
            scheduler.vllm_config.is_mm_encoder_only
            or scheduler.vllm_config.model_config.is_diffusion
        ):
            raise ValueError("This model does not support resumable requests")
        if scheduler.connector is not None or scheduler.ec_connector is not None:
            raise ValueError("Stream admission does not support transfer connectors")

    @staticmethod
    def unsupported_input(request: EngineCoreRequest | Request) -> bool:
        """Whether a stream input uses features a state stream cannot carry."""
        params = request.sampling_params
        if (
            params is None
            or request.pooling_params is not None
            or request.prompt_embeds is not None
            or request.prompt_is_token_ids is not None
        ):
            return True
        extra_args = params.extra_args or {}
        return (
            params.structured_outputs is not None
            or params.n != 1
            or params.output_kind == RequestOutputKind.FINAL_ONLY
            or bool(params.stop)
            or bool(extra_args.get("kv_transfer_params"))
            or bool(extra_args.get("ec_transfer_params"))
        )

    def _refuse(
        self, request_id: str, reason: StreamAdmissionReason
    ) -> StreamAdmissionResult:
        self.refusals[reason] += 1
        return StreamAdmissionResult(False, request_id, reason)

    def accept_input(self, request: Request) -> StreamUpdateRejection | None:
        """The reason the scheduler refuses `request` as an input of an
        admitted session, or None after recording an accepted input or close.
        A request that concerns no session is accepted unchanged."""
        session = self.sessions.get(request.request_id)
        context = request.stream_admission
        if session is None:
            return None if context is None else "unknown_admission"
        if context is None:
            return "identity_mismatch"
        if self.unsupported_input(request):
            return "unsupported_update"
        if request.resumable:
            if session.input_closed:
                return "input_closed"
            if not request.prompt_token_ids:
                return "empty_update"
            # Every refusal precedes the first change; the replacement itself
            # is applied later and must not be able to fail.
            if self.scheduler._misplaced_multimodal_input(request):
                return "validation_error"
            session.status = "live"
            session.deadline = session.deadline_name = None
        if context.input_closed or not request.resumable:
            if not session.input_closed:
                session.input_closed = True
                self.start_deadline(session, "draining")
            session.status = "draining"
        return None

    def start_deadline(self, session: StreamSession, name: DeadlineName) -> None:
        session.deadline = monotonic() + self.timeouts[f"{name}_timeout"]
        session.deadline_name = name

    def next_deadline(self) -> float | None:
        """The earliest deadline that can expire; none while no deadline runs."""
        if self.scheduler.pause_state == PauseState.PAUSED_ALL:
            return None
        return min(
            (s.deadline for s in self.sessions.values() if s.deadline is not None),
            default=None,
        )

    def expire(self) -> None:
        """End every session whose deadline has passed, unless a pause keeps
        every request from running."""
        if self.scheduler.pause_state == PauseState.PAUSED_ALL:
            return
        now = monotonic()
        for request_id, session in list(self.sessions.items()):
            if session.deadline is not None and session.deadline <= now:
                assert session.deadline_name is not None
                self.expiries[session.deadline_name] += 1
                for request in self.scheduler.finish_requests(
                    request_id, RequestStatus.FINISHED_ABORTED
                ):
                    self.scheduler._queue_state_error(request)

    def occupied(self) -> dict[str, int]:
        counts = dict.fromkeys(
            ("never_fed", "live", "parked", "draining", "retiring"), 0
        )
        for request in self.scheduler.requests.values():
            if self.manager.get(request.request_id) is None:
                continue
            session = self.sessions.get(request.request_id)
            if session is not None:
                counts[session.status] += 1
            elif request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                counts["parked"] += 1
            else:
                counts["live"] += 1
        counts["retiring"] = len(self.scheduler.state_retirements)
        return counts
