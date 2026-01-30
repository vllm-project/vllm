# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# EWSJF MODIFICATION: This file implements the EWSJF policy by inheriting from
# the default vLLm v1 Scheduler and overriding its waiting queue management.

from __future__ import annotations

import threading
import time
from typing import cast

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVEventBatch
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.ewsjf_scheduler.scoring import SimpleScoreCalculator
from vllm.v1.core.sched.ewsjf_scheduler.waiting_queue import (
    QueueInfo,
    WaitingQueue,
)
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

# EWSJF MODIFICATION: Import the parent Scheduler class to inherit from it.
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


class EWSJFScheduler(Scheduler):
    """
    EWSJF (Estimated Weighted Shortest Job First) Scheduler.

    This scheduler inherits from the default vLLM Scheduler and implements
    the EWSJF policy by overriding the request queuing and selection
    mechanism. It maintains multiple queues based on prompt lengths and
    selects requests based on estimated completion time and waiting time
    scores.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        # EWSJF MODIFICATION: Call the parent constructor FIRST to initialize everything
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            mm_registry,
            include_finished_set,
            log_stats,
        )

        # EWSJF MODIFICATION: Initialize with the new queue structure
        # FIX: Wrapped line to fix E501
        self.external_parameters = self.vllm_config.scheduler_config.external_parameters
        self.lock = threading.Lock()
        if self.external_parameters and "step_size" in self.external_parameters:
            self.step_size: int = self.external_parameters["step_size"]
        else:
            self.step_size = (
                200  # Default queue size range (removed redundant type hint)
            )

        self.empty_queue_threshold: int = 20  # Cycles before removing empty queue
        self.current_time: float | None = (
            None  # Current timestamp for score calculations
        )
        if self.external_parameters and "score_calculator" in self.external_parameters:
            self.score_calculator = self.external_parameters["score_calculator"]
        else:
            self.score_calculator = SimpleScoreCalculator(weighting_factor=0.5)

        # Core EWSJF data structures
        # FIX: Moved type ignore to the same line
        self.waiting = WaitingQueue(self.lock)  # type: ignore[assignment]
        self.request_partial_scores: dict[str, float] = {}  # Cache for partial scores

        # Helper to treat self.waiting as WaitingQueue for MyPy
        self._ewsjf_waiting = cast(WaitingQueue, self.waiting)

        # Initialize queues either from config or with defaults
        if self.external_parameters and "queues_config" in self.external_parameters:
            self._ewsjf_waiting.initialize_queues_by_config(
                self.external_parameters["queues_config"]
            )
        else:
            self._ewsjf_waiting.initialize_queues(num_queues=10)

        self.get_queues_boundary = None
        # EWSJF MODIFICATION: Start the background optimizer thread.
        self.update_event = threading.Event()  # Signal to start score update
        self.finish_update_event = threading.Event()  # Signal when update is done
        self.update_stop_event = threading.Event()  # Signal to stop the thread
        self.update_thread = threading.Thread(
            target=self._update_scores_loop, daemon=True
        )
        self.update_thread.start()

    # --- EWSJF MODIFICATION: Helper methods for queue management ---
    def schedule(self) -> SchedulerOutput:
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()
        self.current_time = time.time()

        self.update_event.set()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens,
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    *_,  # Ignore extra return values
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                )

            if num_new_tokens == 0:
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )

                if new_blocks is not None:
                    # The request can be scheduled.
                    break

                # The request cannot be scheduled.
                # Preempt the lowest-priority request.
                if self.policy == SchedulingPolicy.PRIORITY:
                    preempted_req = max(
                        self.running,
                        key=lambda r: (r.priority, r.arrival_time),
                    )
                    self.running.remove(preempted_req)
                    if preempted_req in scheduled_running_reqs:
                        scheduled_running_reqs.remove(preempted_req)
                        token_budget += num_scheduled_tokens[preempted_req.request_id]
                        req_to_new_blocks.pop(preempted_req.request_id)
                        num_scheduled_tokens.pop(preempted_req.request_id)
                        req_index -= 1
                else:
                    preempted_req = self.running.pop()

                self.kv_cache_manager.free(preempted_req)
                self.encoder_cache_manager.free(preempted_req)
                preempted_req.status = RequestStatus.PREEMPTED
                preempted_req.num_computed_tokens = 0
                preempted_req.num_preemptions += 1
                if self.log_stats:
                    preempted_req.record_event(
                        EngineCoreEventType.PREEMPTED, scheduled_timestamp
                    )

                self.waiting.prepend_request(preempted_req)
                preempted_reqs.append(preempted_req)
                if preempted_req == request:
                    # No more request to preempt. Cannot schedule this request.
                    break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens + request.num_computed_tokens - request.num_tokens
                )
                if num_scheduled_spec_tokens > 0:
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids
                    )

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule
                )
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            timeout_occurred = not self.finish_update_event.wait(
                timeout=0.1
            )  # 100ms timeout
            if not timeout_occurred:
                self.finish_update_event.clear()
            else:
                logger.warning("Score update timed out, using previous best queue")

            token_budget = self._schedule_waiting_requests_ewsjf(
                encoder_compute_budget,
                num_scheduled_tokens,
                req_index,
                req_to_new_blocks,
                scheduled_encoder_inputs,
                scheduled_loras,
                scheduled_new_reqs,
                scheduled_resumed_reqs,
                scheduled_timestamp,
                skipped_waiting_requests,
                token_budget,
            )

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request.request_id
                )
            )

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids()
            )
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )

        # NOTE: Structured output args removed as they are no longer supported
        # in this version of SchedulerOutput constructor or are optional/renamed.
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )

        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        events = self.kv_cache_manager.take_events()

        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _schedule_waiting_requests_ewsjf(
        self,
        encoder_compute_budget,
        num_scheduled_tokens,
        req_index,
        req_to_new_blocks,
        scheduled_encoder_inputs,
        scheduled_loras,
        scheduled_new_reqs,
        scheduled_resumed_reqs,
        scheduled_timestamp,
        skipped_waiting_requests,
        token_budget,
    ):
        # Use cast to access custom Queue properties
        waiting_queue = cast(WaitingQueue, self.waiting)
        if not waiting_queue.has_best_queue or waiting_queue.is_empty_best_queue:
            return token_budget

        with self.lock:
            while not waiting_queue.is_empty_best_queue and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = waiting_queue.peek_request()
                if not request:
                    break

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,
                        )
                        waiting_queue.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        waiting_queue.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    waiting_queue.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                # FIX: Explicit type hint to allow None reassignment
                num_external_computed_tokens: int | None = 0
                load_kv_async = False

                if request.num_computed_tokens == 0:
                    new_computed_blocks, num_new_local_computed_tokens, *_ = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async, *_ = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if num_external_computed_tokens is None:
                            waiting_queue.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                    # FIX: Assert not None to satisfy int | None type check
                    assert num_external_computed_tokens is not None

                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                else:
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # type check: int | None compared to int, ensure valid
                    assert (
                        num_external_computed_tokens is not None
                        and num_external_computed_tokens > 0
                    )
                    num_new_tokens = 0
                else:
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if (
                        0
                        < self.scheduler_config.long_prefill_token_threshold
                        < num_new_tokens
                    ):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold
                        )

                    if (
                        not self.scheduler_config.chunked_prefill_enabled
                        and num_new_tokens > token_budget
                    ):
                        waiting_queue.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    if request.has_encoder_inputs:
                        # FIX: Using direct unpacking via a temporary variable
                        # to avoid syntax errors with *_, inside parenthesis
                        # which sometimes confuses parsers/formatters
                        sched_result = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                        )
                        encoder_inputs_to_schedule = sched_result[0]
                        num_new_tokens = sched_result[1]
                        new_encoder_compute_budget = sched_result[2]

                        if num_new_tokens == 0:
                            break

                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                if self.is_encoder_decoder and request.has_encoder_inputs:
                    num_encoder_tokens = (
                        self.scheduler_config.max_num_encoder_input_tokens
                    )
                else:
                    num_encoder_tokens = 0

                # FIX: Explicit int casting/assertion for num_external_computed_tokens
                assert num_external_computed_tokens is not None
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    break

                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                request = waiting_queue.pop_request()
                # FIX: Assert request is not None to silence MyPy errors below
                assert request is not None

                if load_kv_async:
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule
                    )
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget

            if skipped_waiting_requests:
                self.waiting.prepend_requests(skipped_waiting_requests)

        return token_budget

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        total_requests_to_reschedule = 0
        total_tokens_to_reschedule = 0

        # Cast for access to get_all_queues
        waiting_queue = cast(WaitingQueue, self.waiting)

        # --- Handle async KV loads (WAITING_FOR_REMOTE_KVS) ---
        async_load_reqs = (
            req
            for queue in waiting_queue.get_all_queues()
            for req in queue.requests
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_affected_req_ids, num_tokens_to_reschedule, *_ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs, invalid_block_ids
            )
        )

        total_requests_to_reschedule += len(async_affected_req_ids)
        total_tokens_to_reschedule += num_tokens_to_reschedule

        self.failed_recving_kv_req_ids |= async_affected_req_ids

        # --- Handle sync KV loads (running requests) ---
        sync_affected_req_ids, num_tokens_to_reschedule, *_ = (
            self._update_requests_with_invalid_blocks(self.running, invalid_block_ids)
        )

        total_requests_to_reschedule += len(sync_affected_req_ids)
        total_tokens_to_reschedule += num_tokens_to_reschedule

        if total_requests_to_reschedule:
            logger.warning(
                "Recovered from KV load failure: "
                "%d request(s) rescheduled (%d tokens affected).",
                total_requests_to_reschedule,
                total_tokens_to_reschedule,
            )

        return sync_affected_req_ids

    def _update_scores_loop(self):
        while not self.update_stop_event.is_set():
            self.update_event.wait()
            self.update_event.clear()
            self._update_scores()
            self.finish_update_event.set()

    def _update_scores(self):
        new_best_queue = None
        new_best_score = -1.0
        queues_to_remove = []

        # Cast for access to get_all_queues and update_best_queue
        waiting_queue = cast(WaitingQueue, self.waiting)

        with self.lock:
            for queue in waiting_queue.get_all_queues(self.get_queues_boundary):
                if queue.is_empty:
                    if queue.removable:
                        queue.increment_empty_count()
                        if (
                            queue.empty_count >= self.empty_queue_threshold
                            and waiting_queue.queues_count > 1
                        ):
                            queues_to_remove.append(queue)
                    else:
                        queue.update_score(0.0)
                    continue

                queue.reset_empty_count()
                first_req = queue.peek_request()
                if not first_req:
                    continue

                partial_score = self.request_partial_scores.get(
                    first_req.request_id, 0.0
                )
                if partial_score == 0.0:
                    partial_score = self.score_calculator.get_partial_score(
                        first_req, self.step_size
                    )
                    self.request_partial_scores[first_req.request_id] = partial_score

                score = self.score_calculator.complete_score(
                    first_req, partial_score, self.current_time
                )
                queue.update_score(score)

                if score > new_best_score:
                    new_best_score = score
                    new_best_queue = queue

        waiting_queue.update_best_queue(new_best_queue)

        with self.lock:
            for queue in queues_to_remove:
                self._remove_queue(queue)

    def _remove_queue(self, queue_to_remove: QueueInfo):
        waiting_queue = cast(WaitingQueue, self.waiting)
        remaining_requests = waiting_queue.delete_queue(queue_to_remove)

        if remaining_requests:
            for req in remaining_requests:
                self.add_request(req)

    def shutdown(self):
        super().shutdown()
        self.update_stop_event.set()
        self.update_thread.join(timeout=1)
