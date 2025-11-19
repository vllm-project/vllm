from collections.abc import Callable
from typing import Any

from overrides import overrides

from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.request import Request, RequestStatus
from vllm.v1.streaming.engine import (
    StreamingEngineCoreOutput,
    StreamingEngineCoreOutputs,
)
from vllm.v1.streaming.streaming_request import StreamingRequest


class StreamingScheduler(Scheduler):
    engine_core_output_cls: type[EngineCoreOutput] = StreamingEngineCoreOutput
    engine_core_outputs_cls: type[EngineCoreOutputs] = StreamingEngineCoreOutputs

    @overrides
    def add_request(self, request: Request) -> None:
        assert isinstance(request, StreamingRequest)
        if request.request_id not in self.requests:
            # TODO: add logic in vllm predictor to make sure the first request is 
            # processed before sending others.
            assert (
                request.streaming_sequence_id == 0
            ), f"Request with streaming_sequence_id=0 request must be sent before other\
            requests with streaming_sequence_id {request.streaming_sequence_id} > 1"
            if request.close_session:
                return
            self.requests[request.request_id] = request
        else:
            session_request = self.requests[request.request_id]
            assert request.streaming_sequence_id > session_request.streaming_sequence_id
            request.status = RequestStatus.WAITING_FOR_SESSION_REQ

        self.waiting.add_request(request)
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def _update_session_request(
        self, session_request: StreamingRequest, request: StreamingRequest
    ) -> None:
        """
        Updates the waiting session request with the new streaming request.

        Removes the last output token (which hasn't been scheduled yet) from
        `_all_token_ids`, as the new request's prompt tokens are replacing it. Typically
        decoding outputs are scheduled as the next input in autoregressive decoding.
        When we receive a new streaming request, the new prompt becomes our next input,
        so the last output token is no longer needed and will not join the kv cache.
        This also guarantees correct calculation of `num_new_tokens` in `schedule`.
        """
        num_new_tokens = (
            session_request.num_tokens - session_request.num_computed_tokens
        )
        assert num_new_tokens in {0, 1}
        if num_new_tokens == 1:
            assert (
                session_request._all_token_ids[-1]
                == session_request._output_token_ids[-1]
            )
            del session_request._all_token_ids[-1]

        if request.mm_features:
            base = session_request.num_tokens
            for mm_feature in request.mm_features:
                mm_feature.mm_position = PlaceholderRange(
                    offset=mm_feature.mm_position.offset + base,
                    length=mm_feature.mm_position.length,
                )
            session_request.mm_features.extend(request.mm_features)

        session_request._all_token_ids.extend(request.prompt_token_ids)
        session_request.prompt_token_ids.extend(request.prompt_token_ids)
        session_request.prompt_embeds = request.prompt_embeds
        session_request.max_tokens = (
            session_request.num_output_tokens + request.max_tokens
        )
        session_request.streaming_sequence_id = request.streaming_sequence_id
        session_request.arrival_time = request.arrival_time
        session_request.priority = request.priority
        session_request.sampling_params = request.sampling_params
        session_request.status = RequestStatus.WAITING
        assert not request.close_session

        if self.log_stats:
            session_request.record_event(EngineCoreEventType.QUEUED)

    def process_streaming_requests(self) -> RequestQueue:
        waiting_requests = create_request_queue(self.policy)
        while self.waiting:
            request = self.waiting.pop_request()
            if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                pass
            elif request.status == RequestStatus.WAITING_FOR_SESSION_REQ:
                session_request = self.requests[request.request_id]
                if (
                    session_request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
                    and request.streaming_sequence_id
                    == 1 + session_request.streaming_sequence_id
                ):
                    if request.close_session:
                        session_request.status = RequestStatus.FINISHED_STOPPED
                        session_request.close_session = True
                        session_request.stop_reason = "close_session"
                        self.finished_req_ids.add(session_request.request_id)
                    else:
                        self._update_session_request(session_request, request)
                    continue
            waiting_requests.prepend_request(request)

        skipped_waiting_requests = create_request_queue(self.policy)
        while waiting_requests:
            request = waiting_requests.pop_request()
            if (
                request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
                or request.status == RequestStatus.WAITING_FOR_SESSION_REQ
            ):
                skipped_waiting_requests.prepend_request(request)
            elif not RequestStatus.is_finished(request.status):
                self.waiting.prepend_request(request)

        return skipped_waiting_requests

    @overrides
    def _create_new_request_data(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> NewRequestData:
        """
        Creates NewRequestData for requests in waiting queue to be sent to
        ModelRunner via SchedulerOutput.scheduled_new_reqs.

        This applies to new or updated session requests, not for ongoing decodes. When
        updating session requests, we need to provide all tokens for the prompt field
        to include all past inputs, including decoded outputs. New and updated session
        requests create new entries in InputBatch, so we need the full input history to
        ensure alignment of mm offsets, kv cache, and token ids.

        Make sure that prompt_token_ids is a copy of the original request's
        _all_token_ids. Since the scheduler updates _all_token_ids each iteration, when
        continuing decoding for a text or special token an updated NewRequestData won't
        be sent from the scheduler, the corresponding prompt_token_ids reference in 
        SpeechGPUModelRunner will be mistakenly updated.
        """
        out = super()._create_new_request_data(request, block_ids)
        out.prompt_token_ids = request._all_token_ids.copy()
        return out

    @overrides
    def schedule(self) -> SchedulerOutput:
        # When new streaming request comes, the lifecycle is
        # WAITING_FOR_SESSION_REQ -> WAIT (becomes session request) -> 
        # RUNNING -> WAITING_FOR_STREAM_REQ
        skipped_waiting_requests = self.process_streaming_requests()
        scheduler_output = super().schedule()
        self.waiting.prepend_requests(skipped_waiting_requests)
        return scheduler_output

    @overrides
    def _make_engine_core_output(
        self,
        request: Request,
        **kwargs: Any,
    ) -> StreamingEngineCoreOutput:
        kwargs["close_session"] = request.close_session
        return self.engine_core_output_cls(**kwargs)

    @overrides
    def has_unfinished_requests(self) -> bool:
        """
        Returns True if there are unfinished requests in the scheduler's
        internal queue, excluding requests with status WAITING_FOR_STREAMING_REQ
        which are considered finished.
        """
        # Count running requests (none should have WAITING_FOR_STREAMING_REQ status)
        if len(self.running) > 0:
            return True

        # Return true if there are any request in waiting queue
        # not with WAITING_FOR_STREAMING_REQ status
        for request in self.waiting:
            if request.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                return True

        return False

    @overrides
    def _handle_stopped(
        self,
        request: Request,
        status_before_stop: RequestStatus,
        mark_running_stopped: Callable[[Request], None],
        mark_preempted_stopped: Callable[[Request], None],
    ) -> dict[str, Any] | None:
        assert not request.close_session, "session should already be closed"
        request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
        self.waiting.add_request(request)
        kv_transfer_params = None
        if status_before_stop == RequestStatus.RUNNING:
            mark_running_stopped(request)
        else:
            mark_preempted_stopped(request)
        return kv_transfer_params

    @overrides
    def _handle_finished(
        self,
        finished_req_ids: set[str],
        outputs: dict[int, list[EngineCoreOutput]],
    ) -> None:
        for req_id in finished_req_ids:
            request = self.requests.get(req_id)
            if request is None:
                # This can happen if the request was aborted.
                continue
            assert request.close_session, "session should be in close state"
            outputs[request.client_index].append(
                self._make_engine_core_output(
                    request,
                    new_token_ids=[],
                    request_id=req_id,
                    finish_reason=request.get_finished_reason(),
                    stop_reason=request.stop_reason,
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                    num_cached_tokens=request.num_cached_tokens,
                )
            )
            self._free_request(request)
