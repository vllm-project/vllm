# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler for models with scheduler-owned persistent state.

See docs/design/persistent_state.md for the slot, release-fence and stream
admission contracts this scheduler implements.
"""

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

from vllm.config import SchedulerConfig, VllmConfig
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.state_slot_manager import StateSlotRetirement
from vllm.v1.core.stream_admission import StreamAdmissions
from vllm.v1.engine import (
    EngineCoreEventType,
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
    StreamAdmissionContext,
    StreamAdmissionResult,
    StreamOutput,
    StreamUpdateRejection,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    StateSlotHandle,
)
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate

# Cache clears that would end a live stream are refused.
RESET_HELD = (
    "persistent state: {n} state slot(s) are held, and resetting running "
    "requests would end their streams. Abort the streams and wait until they "
    "have retired, then reset again."
)
PAUSE_HELD = (
    "persistent state: {n} state slot(s) are held, and clearing the cache would "
    "end their streams. Pause with clear_cache=False; to clear the cache, abort "
    "the streams, wait until they have retired, then reset the prefix cache."
)
RETIRING = (
    "persistent state: {n} state slot(s) are still retiring; retry the reset "
    "once they have retired."
)
RELEASE_REFUSED = (
    "persistent state: the KV cache holds per-stream state, so it cannot be "
    "released while the engine runs."
)


def get_scheduler_cls(
    scheduler_config: SchedulerConfig, kv_cache_config: KVCacheConfig
) -> type[SchedulerInterface]:
    """The configured scheduler class, or a StateScheduler for a
    persistent-state KV cache layout."""
    if not kv_cache_config.state_groups:
        return scheduler_config.get_scheduler_cls()
    if scheduler_config.scheduler_cls is None:
        return StateScheduler
    scheduler_cls = scheduler_config.get_scheduler_cls()
    if not (
        isinstance(scheduler_cls, type)
        and issubclass(scheduler_cls, StateScheduler)
        and not issubclass(scheduler_cls, AsyncScheduler)
    ):
        raise ValueError(
            "Persistent state requires a StateScheduler subclass as "
            f"scheduler_cls, got {scheduler_cls!r}"
        )
    return scheduler_cls


class StateScheduler(Scheduler):
    """Owns per-request state slots: allocation through the KV cache manager,
    retirement behind the worker's release fence, and stream admission."""

    def __init__(
        self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, *args, **kwargs
    ) -> None:
        if not kv_cache_config.state_groups:
            raise ValueError("StateScheduler requires a persistent-state KV cache")
        super().__init__(vllm_config, kv_cache_config, *args, **kwargs)
        self.state_retirements: list[StateSlotRetirement] = []
        self.state_cleanup_misses: dict[StateSlotHandle, int] = {}
        self.state_step_seq = 0
        self.state_error_outputs: list[tuple[int, EngineCoreOutput]] = []
        self.state_outputs_held = False
        self.state_ended_streams: set[str] = set()
        # The context of the input each admitted stream's stopped turn ran on.
        self.state_turn_contexts: dict[str, StreamAdmissionContext] = {}
        # Released admitted streams not yet reported, by client index.
        self.retired_streams: dict[int, list[str]] = {}
        self.stream_admissions = StreamAdmissions(self)

    @property
    def state_managers(self):
        return self.kv_cache_manager.coordinator.state_managers

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.stream_admissions.expire()
        return super().schedule(throttle_prefills)

    def admit_stream(self, request: EngineCoreRequest) -> StreamAdmissionResult:
        """Admit a stream before it has any input, or refuse it with a reason."""
        self.stream_admissions.expire()
        return self.stream_admissions.admit(request)

    def set_pause_state(self, pause_state: PauseState) -> None:
        resumed = (
            self.pause_state == PauseState.PAUSED_ALL
            and pause_state != PauseState.PAUSED_ALL
        )
        super().set_pause_state(pause_state)
        if resumed:
            # Deadlines did not run during the pause, so each starts again.
            for session in self.stream_admissions.sessions.values():
                if session.deadline_name is not None:
                    self.stream_admissions.start_deadline(
                        session, session.deadline_name
                    )

    def _holds_kv_blocks(self, request: Request) -> bool:
        # A replacement prompt restarts at zero computed tokens while the
        # request keeps its slot, and an admitted stream holds its slot before
        # any token is computed. Slot ownership, not computed tokens, decides
        # the holding queue, so the stream drains ahead of requests that still
        # need a slot and under PAUSED_NEW.
        return any(
            manager.get(request.request_id) is not None
            for manager in self.state_managers.values()
        ) or super()._holds_kv_blocks(request)

    def _handle_blocked_waiting_request(self, request: Request) -> bool:
        # A request whose previous slot is still retiring waits for its fence.
        if any(
            entry.request_id == request.request_id for entry in self.state_retirements
        ):
            return False
        return super()._handle_blocked_waiting_request(request)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Snapshot handles before the base clears this step's finished ids.
        self.state_step_seq += 1
        scheduler_output.state_step_seq = self.state_step_seq
        # Retry unacknowledged cleanup without repeating removal callbacks.
        # Sync execution and the worker's stream joins fence earlier writes.
        cleanup: dict[str, dict[str, StateSlotHandle]] = {}
        for entry in self.state_retirements:
            cleanup.setdefault(entry.request_id, {})[entry.layer_name] = entry.handle
        scheduler_output.state_cleanup_handles = cleanup
        scheduler_output.state_slot_handles = {
            req_id: {
                name: handle
                for name, manager in self.state_managers.items()
                if (handle := manager.get(req_id)) is not None
            }
            for req_id in scheduler_output.num_scheduled_tokens
        }
        for req_id in scheduler_output.num_scheduled_tokens:
            for manager in self.state_managers.values():
                manager.last_scheduled[req_id] = self.state_step_seq
        super()._update_after_schedule(scheduler_output)

    def _preempt_request(
        self, request: Request, timestamp: float, drop_stale_output: bool = False
    ) -> None:
        # State cannot be rebuilt from tokens, so preemption ends the request.
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        request.status = RequestStatus.FINISHED_ERROR
        self._free_request(request)
        self._queue_state_error(request)

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """Replace the parked session's prompt; the state carries the history."""
        if session.status not in (
            RequestStatus.WAITING_FOR_STREAMING_REQ,
            RequestStatus.FINISHED_STOPPED,
        ):
            raise ValueError("state stream replacement requires a legal park")
        # add_request refuses misplaced multimodal items before a continuation is
        # applied or queued; the only callers are add_request and the queue it
        # fills, so this cannot fail.
        assert not self._misplaced_multimodal_input(update), (
            "multimodal item is outside replacement prompt"
        )
        prompt = update.prompt_token_ids or ()
        self.encoder_cache_manager.free(session)
        # Replace the token buffers in place: the all_token_ids view wraps one.
        assert session.prompt_token_ids is not None
        del session.prompt_token_ids[:]
        session.prompt_token_ids.extend(prompt)
        del session._all_token_ids[:]
        session._all_token_ids.extend(prompt)
        session._output_token_ids.clear()
        session.num_computed_tokens = 0
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.prompt_embeds = None
        session.prompt_is_token_ids = None
        session.mm_features = list(update.mm_features or ())
        session.block_hashes.clear()
        session.update_block_hashes()
        session.stop_reason = None
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        session.max_tokens = update.max_tokens
        if update.stream_admission is not None:
            session.stream_admission = update.stream_admission
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING
        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _misplaced_multimodal_input(
        self, request: EngineCoreRequest | Request | StreamingUpdate
    ) -> bool:
        """Whether a multimodal item falls outside the input's own prompt.

        Each stream input replaces the prompt, so positions are relative to it.
        A decoder-only item must fit within the prompt, or the scheduler could
        never cover it whole. An encoder-decoder item is encoder input whose
        length is independent of the decoder prompt, so only its start counts.
        """
        prompt = request.prompt_token_ids
        num_tokens = 0 if prompt is None else len(prompt)
        for feature in request.mm_features or ():
            position = feature.mm_position
            if not 0 <= position.offset < num_tokens:
                return True
            if (
                not self.is_encoder_decoder
                and position.offset + position.length > num_tokens
            ):
                return True
        return False

    def _get_stale_output_req_ids(
        self, scheduler_output: SchedulerOutput
    ) -> set[str] | None:
        # Outputs for a handle that is no longer the request's live slot.
        return {
            req_id
            for req_id, handles in (scheduler_output.state_slot_handles or {}).items()
            if any(
                self.state_managers[name].get(req_id) != handle
                for name, handle in handles.items()
            )
        }

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        self._complete_state_slots(scheduler_output, model_runner_output)
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        self._mark_stream_outputs(outputs)
        if not self.state_outputs_held:
            for client_index, request_ids in self.retired_streams.items():
                if (batch := outputs.get(client_index)) is None:
                    outputs[client_index] = batch = EngineCoreOutputs()
                batch.retired_streams = request_ids
            self.retired_streams = {}
        return outputs

    def _mark_stream_outputs(self, outputs: dict[int, EngineCoreOutputs]) -> None:
        """Mark the outputs that end a stream, and give each output of an
        admitted stream the context of the input it ran on."""
        for batch in outputs.values():
            for output in batch.outputs:
                request_id = output.request_id
                if output.stream_output is not None:
                    continue
                if request_id in self.state_ended_streams:
                    output.ends_stream = True
                # A stopped turn's context was read before stop handling
                # could apply the next queued input.
                context = self.state_turn_contexts.get(request_id)
                if context is None and (request := self.requests.get(request_id)):
                    context = request.stream_admission
                if context is not None:
                    output.stream_output = StreamOutput(context)
        self.state_ended_streams.clear()
        self.state_turn_contexts.clear()

    def _add_queued_outputs(self, outputs: dict[int, list[EngineCoreOutput]]) -> None:
        if self.state_outputs_held:
            return
        for client_index, error_output in self.state_error_outputs:
            outputs[client_index].append(error_output)
        self.state_error_outputs.clear()

    def _handle_stopped_request(self, request: Request) -> bool:
        # A state stream parks only at EOS; any other stop leaves model state
        # mid-turn, so the stream ends instead of resuming, and the turn's
        # output also ends the client's stream.
        # An input that closes input ends its session once it has run.
        context = request.stream_admission
        if context is not None:
            self.state_turn_contexts[request.request_id] = context
        if request.resumable and (
            request.status != RequestStatus.FINISHED_STOPPED
            or request.stop_reason is not None
            or (context is not None and context.input_closed)
        ):
            request.resumable = False
            self.state_ended_streams.add(request.request_id)
        finished = super()._handle_stopped_request(request)
        if finished and context is not None:
            self.state_ended_streams.add(request.request_id)
        session = self.stream_admissions.sessions.get(request.request_id)
        if (
            request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
            and session is not None
            and not session.input_closed
        ):
            session.status = "parked"
            self.stream_admissions.start_deadline(session, "parked_idle")
        return finished

    def add_request(self, request: Request) -> None:
        self.stream_admissions.expire()
        admitted = self.stream_admissions.sessions.get(request.request_id)
        closed = admitted is not None and admitted.input_closed
        rejection = self.stream_admissions.accept_input(request)
        if rejection is not None:
            # The engine core aborts a request that asks for it once it is
            # added; a refused input must leave its session running.
            request.abort_immediately = False
            self._queue_state_error(request, rejection)
            return
        if closed:
            # A close after the first changes nothing.
            return
        session = self.requests.get(request.request_id)
        if (
            request.stream_admission is not None
            and not request.resumable
            and session is not None
            and session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
        ):
            # A close with nothing left to run ends the session at once.
            session.stream_admission = request.stream_admission
            for ended_request in self.finish_requests(
                request.request_id, RequestStatus.FINISHED_STOPPED
            ):
                self._queue_state_error(ended_request)
            return
        if (
            request.stream_admission is None
            and request.resumable
            and self._misplaced_multimodal_input(request)
        ):
            # Every refusal precedes the first mutation; the replacement itself
            # is applied later and must not be able to fail. A continuation
            # ends its stream at once, even mid-turn, so its slot retires; a
            # first input is never queued.
            if request.request_id in self.requests:
                ended = self.finish_requests(
                    request.request_id, RequestStatus.FINISHED_ERROR
                )
            else:
                request.status = RequestStatus.FINISHED_ERROR
                ended = [request]
            for ended_request in ended:
                self._queue_state_error(ended_request)
            return
        super().add_request(request)

    def has_queued_requests(self) -> bool:
        """Whether a request waits or runs, whatever the pause state; a parked
        stream waits only for its next input."""
        return sum(self.get_request_counts()) > self.num_waiting_for_streaming_input

    # The hold does not nest: leaving any hold ends it.
    @contextmanager
    def holding_queued_outputs(self) -> Iterator[None]:
        """Hold queued error outputs, including those queued meanwhile, back
        from steps whose outputs the caller drops, so that the next step
        delivers them."""
        self.state_outputs_held = True
        try:
            yield
        finally:
            self.state_outputs_held = False

    def state_slot_counts(self) -> tuple[int, int]:
        """The bound or parked state slots, and the retiring ones."""
        held = sum(
            any(m.get(req_id) is not None for m in self.state_managers.values())
            for req_id in self.requests
        )
        return held, len(self.state_retirements)

    def check_cache_clear(self, held_message: str, refuse_retiring: bool) -> None:
        """Refuse a cache clear that would end a live stream, before any change;
        with `refuse_retiring`, also one that must wait for retiring slots."""
        held, retiring = self.state_slot_counts()
        if held:
            raise RuntimeError(held_message.format(n=held))
        if retiring and refuse_retiring:
            raise RuntimeError(RETIRING.format(n=retiring))

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # Without resetting running requests, held slots keep blocks in use, so
        # the base reset returns False and changes nothing.
        if reset_running_requests:
            self.check_cache_clear(RESET_HELD, refuse_retiring=True)
        return super().reset_prefix_cache(reset_running_requests, reset_connector)

    def _free_request_blocks(self, request: Request) -> None:
        # Retire the slot before the generic block free; it returns to the
        # pool only once the worker acknowledges its release fence.
        stream_client = (
            request.client_index
            if self.stream_admissions.sessions.pop(request.request_id, None)
            else None
        )
        for name, manager in self.state_managers.items():
            last_step = manager.last_scheduled.pop(request.request_id, 0)
            fence = self.state_step_seq + 1 if last_step else 0
            handle = manager.retire(request.request_id, fence)
            if handle is not None:
                self.state_retirements.append(
                    StateSlotRetirement(
                        manager, handle, fence, request.request_id, name, stream_client
                    )
                )
        self._drain_state_retirements()
        super()._free_request_blocks(request)

    def has_finished_requests(self) -> bool:
        # Held errors wait for the step after the hold, so they must not keep
        # the held steps going.
        deliverable = (
            self.state_error_outputs or self.retired_streams
        ) and not self.state_outputs_held
        return (
            bool(deliverable or self.state_retirements)
            or super().has_finished_requests()
        )

    def make_stats(self, *args: Any, **kwargs: Any) -> SchedulerStats | None:
        stats = super().make_stats(*args, **kwargs)
        if stats is not None:
            stats.stream_occupied = self.stream_admissions.occupied()
            stats.stream_refusals = dict(self.stream_admissions.refusals)
            stats.stream_expiries = dict(self.stream_admissions.expiries)
        return stats

    def _drain_state_retirements(self) -> None:
        released = []
        retiring = []
        for entry in self.state_retirements:
            if entry.allocator.release(entry.handle, entry.fence_step):
                released.append(entry)
            else:
                retiring.append(entry)
        self.state_retirements = retiring
        # A stream retires once its last state manager releases its slot.
        still_retiring = {entry.request_id for entry in retiring}
        for entry in released:
            if entry.stream_client is None or entry.request_id in still_retiring:
                continue
            request_ids = self.retired_streams.setdefault(entry.stream_client, [])
            if entry.request_id not in request_ids:
                request_ids.append(entry.request_id)

    def _complete_state_slots(
        self, scheduled: SchedulerOutput, output: ModelRunnerOutput
    ) -> None:
        completion = output.state_completion
        failed = set()
        for req_id, handles in (scheduled.state_slot_handles or {}).items():
            for name, handle in handles.items():
                manager = self.state_managers[name]
                if (
                    completion is not None
                    and completion.step_seq == scheduled.state_step_seq
                    and completion.handles.get(req_id, {}).get(name) == handle
                ):
                    with suppress(ValueError):
                        manager.complete(handle, completion.step_seq)
                    if (
                        req_id in completion.failed_req_ids
                        and manager.get(req_id) == handle
                    ):
                        failed.add(req_id)
                else:
                    with suppress(ValueError):
                        manager.quarantine(handle, scheduled.state_step_seq)
                    if manager.get(req_id) == handle:
                        failed.add(req_id)
        for req_id, handles in (scheduled.state_cleanup_handles or {}).items():
            for name, handle in handles.items():
                manager = self.state_managers[name]
                if (
                    completion is not None
                    and completion.step_seq == scheduled.state_step_seq
                    and completion.cleanup_handles.get(req_id, {}).get(name) == handle
                ):
                    self.state_cleanup_misses.pop(handle, None)
                    with suppress(ValueError):
                        manager.complete(handle, completion.step_seq)
                    continue
                with suppress(ValueError):
                    manager.quarantine(handle, scheduled.state_step_seq)
                # The initial cleanup step and one retry both missed: the worker
                # broke the state completion contract, so stop rather than poll.
                misses = self.state_cleanup_misses.get(handle, 0) + 1
                if misses > 1:
                    raise RuntimeError(
                        f"State slot {handle} of request {req_id!r} was not "
                        f"acknowledged after cleanup step "
                        f"{scheduled.state_step_seq}; its writes are unfenced"
                    )
                self.state_cleanup_misses[handle] = misses
        self._drain_state_retirements()
        if failed:
            for request in self.finish_requests(failed, RequestStatus.FINISHED_ERROR):
                self._queue_state_error(request)

    def _queue_state_error(
        self, request: Request, rejection: StreamUpdateRejection | None = None
    ) -> None:
        """Queue an output with no tokens: the rejection of an input, or
        the end of `request`, which ends the stream of an admitted session."""
        context = request.stream_admission
        stream_output = None
        if rejection is not None:
            stream_output = StreamOutput(context, rejection)
        elif context is not None:
            stream_output = StreamOutput(context)
        self.state_error_outputs.append(
            (
                request.client_index,
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=(
                        request.get_finished_reason()
                        if rejection is None and request.is_finished()
                        else FinishReason.ERROR
                    ),
                    ends_stream=rejection is None and context is not None,
                    stream_output=stream_output,
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                ),
            )
        )
