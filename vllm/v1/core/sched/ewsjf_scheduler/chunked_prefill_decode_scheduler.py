# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# EWSJF MODIFICATION: This file implements the EWSJF policy by inheriting from
# the default vLLm v1 Scheduler and overriding its waiting queue management.

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple, Deque, Union
from sortedcontainers import SortedDict

from vllm.v1.core.sched.ewsjf_scheduler.scheduler import EWSJFScheduler
from vllm.v1.core.sched.ewsjf_scheduler.waiting_queue import WaitingQueue, QueueInfo
# EWSJF MODIFICATION: Import the parent Scheduler class to inherit from it.
from vllm.v1.core.sched.scheduler import Scheduler

from vllm.v1.core.sched.ewsjf_scheduler.scoring import SimpleScoreCalculator
import time
from collections import defaultdict
from collections.abc import Iterable

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVEventBatch

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


class ChunkedPrefillSchedulerDecode(EWSJFScheduler):
    """
    EWSJF (Estimated Weighted Shortest Job First) Scheduler.

    This scheduler inherits from the default vLLM Scheduler and implements the EWSJF policy
    by overriding the request queuing and selection mechanism. It maintains multiple queues
    based on prompt lengths and selects requests based on estimated completion time and
    waiting time scores.

    Key Features:
    - Multiple queues organized by prompt length ranges
    - Dynamic queue creation and removal
    - Score-based request selection using EWSJF algorithm
    - Background thread for continuous score updates
    - Preserves all advanced vLLM features (preemption, caching, LoRA, etc.)
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
        """
        Initialize the EWSJF Scheduler.

        Args:
            vllm_config (VllmConfig): vLLM configuration object
            kv_cache_config (KVCacheConfig): Key-value cache configuration
            structured_output_manager (StructuredOutputManager): Manager for structured outputs
            mm_registry (MultiModalRegistry, optional): Multimodal registry. Defaults to MULTIMODAL_REGISTRY.
            include_finished_set (bool, optional): Whether to track finished requests. Defaults to False.
            log_stats (bool, optional): Whether to log scheduling statistics. Defaults to False.
        """
        # EWSJF MODIFICATION: Call the parent constructor FIRST to initialize everything.
        print("&&&&&&&&&&&&&&&&&&&&&& init chunked prefill scheduler &&&&&&&&&&&&&&&&&&&")

        super().__init__(vllm_config, kv_cache_config, structured_output_manager,
                         block_size, mm_registry, include_finished_set, log_stats)
        self.priority_running: list[Request] = []

        self.skip_running_for_fairness = False
        self.consecutive_long_prefills = 0
        self.last_step_had_long_prefill = False
        self.num_skip_running = 0

    def _is_long_prefill(self, request: Request) -> bool:
        if request.num_computed_tokens >= request.num_tokens:
            return False

        remaining_tokens = request.num_tokens - request.num_computed_tokens
        return remaining_tokens > self.scheduler_config.long_prefill_token_threshold

    def _process_priority_running(self, req_index, token_budget, encoder_compute_budget, scheduled_running_reqs,
                                  num_scheduled_tokens, req_to_new_blocks, scheduled_timestamp, preempted_reqs,
                                  scheduled_spec_decode_tokens, scheduled_encoder_inputs):
        # First, schedule the RUNNING requests.
        while req_index < len(self.priority_running) and token_budget > 0:
            request = self.priority_running[req_index]

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
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
                        self.priority_running,
                        key=lambda r: (r.priority, r.arrival_time),
                    )
                    self.priority_running.remove(preempted_req)
                    if preempted_req in scheduled_running_reqs:
                        scheduled_running_reqs.remove(preempted_req)
                        token_budget += num_scheduled_tokens[preempted_req.request_id]
                        req_to_new_blocks.pop(preempted_req.request_id)
                        num_scheduled_tokens.pop(preempted_req.request_id)
                        req_index -= 1
                else:
                    preempted_req = self.priority_running.pop()

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

            if self._is_long_prefill(request):
                self.consecutive_long_prefills += 1
                self.last_step_had_long_prefill = True

                # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens + request.num_computed_tokens - request.num_tokens
                )
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids
                    )

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule
                )
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget

        # Record the LoRAs in scheduled_running_reqs
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        return token_budget

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

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

        req_index = 0
        scheduled_loras: set[int] = set()

        if self.priority_running: # and self.num_skip_running == 0:

            # print("&&&&&&&&&&&&&&&&&&&&&& run priority running &&&&&&&&&&&&&&&&&&&")
            token_budget = self._process_priority_running(req_index, token_budget, encoder_compute_budget, scheduled_running_reqs,
                                           num_scheduled_tokens, req_to_new_blocks, scheduled_timestamp, preempted_reqs,
                                           scheduled_spec_decode_tokens, scheduled_encoder_inputs)

            scheduler_output = self._get_scheduler_output(num_scheduled_tokens, req_to_new_blocks,
                                                          scheduled_encoder_inputs, scheduled_running_reqs,
                                                          scheduled_spec_decode_tokens, token_budget)

            return scheduler_output

        self.skip_running_for_fairness = (
                self.consecutive_long_prefills >= self.scheduler_config.max_long_partial_prefills
                and self.waiting
                and any(self._is_long_prefill(req) for req in self.running)
        )

        if self.skip_running_for_fairness and self.scheduler_config.long_prefill_token_threshold > 0:
            self.get_queues_boundary =  self.scheduler_config.long_prefill_token_threshold
        else:
            self.get_queues_boundary = None

        self.update_event.set()

        if not self.skip_running_for_fairness: # and self.num_skip_running == 0:
        # First, schedule the RUNNING requests.
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
                # This is necessary when using spec decoding.
                num_new_tokens = min(
                    num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
                )

                # Schedule encoder inputs.
                encoder_inputs_to_schedule = None
                new_encoder_compute_budget = encoder_compute_budget
                if request.has_encoder_inputs:
                    (
                        encoder_inputs_to_schedule,
                        num_new_tokens,
                        new_encoder_compute_budget,
                    ) = self._try_schedule_encoder_inputs(
                        request,
                        request.num_computed_tokens,
                        num_new_tokens,
                        encoder_compute_budget,
                    )

                if num_new_tokens == 0:
                    # The request cannot be scheduled because one of the following
                    # reasons:
                    # 1. No new tokens to schedule. This may happen when
                    #    (1) PP>1 and we have already scheduled all prompt tokens
                    #    but they are not finished yet.
                    #    (2) Async scheduling and the request has reached to either
                    #    its max_total_tokens or max_model_len.
                    # 2. The encoder budget is exhausted.
                    # 3. The encoder cache is exhausted.
                    # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                    # we do not strictly follow the FCFS scheduling policy and
                    # allow the lower-priority requests to be scheduled.
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

                if self._is_long_prefill(request):
                    self.consecutive_long_prefills += 1
                    self.last_step_had_long_prefill = True

                    # Speculative decode related.
                if request.spec_token_ids:
                    num_scheduled_spec_tokens = (
                        num_new_tokens + request.num_computed_tokens - request.num_tokens
                    )
                    if num_scheduled_spec_tokens > 0:
                        # Trim spec_token_ids list to num_scheduled_spec_tokens.
                        del request.spec_token_ids[num_scheduled_spec_tokens:]
                        scheduled_spec_decode_tokens[request.request_id] = (
                            request.spec_token_ids
                        )

                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule
                    )
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget

            # Record the LoRAs in scheduled_running_reqs
            if self.lora_config:
                scheduled_loras = set(
                    req.lora_request.lora_int_id
                    for req in scheduled_running_reqs
                    if req.lora_request and req.lora_request.lora_int_id > 0
                )
                assert len(scheduled_loras) <= self.lora_config.max_loras
        else:
            self.last_step_had_long_prefill = False
            self.consecutive_long_prefills = 0
            # self.num_skip_running += 1
            #
            # if self.num_skip_running == 3:
            #     self.num_skip_running = 0

            # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            # Wait for score update to complete
            timeout_occurred = not self.finish_update_event.wait(timeout=0.1)  # 100ms timeout
            if not timeout_occurred:
                self.finish_update_event.clear()
            else:
                # Use previous best_queue if update didn't complete
                logger.warning("Score update timed out, using previous best queue")

            token_budget = self._schedule_waiting_requests_ewsjf(encoder_compute_budget, num_scheduled_tokens, req_index,
                                                                 req_to_new_blocks, scheduled_encoder_inputs, scheduled_loras,
                                                                 scheduled_new_reqs, scheduled_resumed_reqs, scheduled_timestamp,
                                                                 skipped_waiting_requests, token_budget)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
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
        structured_output_request_ids, grammar_bitmask = self.get_grammar_bitmask(
            num_scheduled_tokens.keys(), scheduled_spec_decode_tokens
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _get_scheduler_output(self, num_scheduled_tokens: dict[str, int], req_to_new_blocks: dict[str, KVCacheBlocks],
                              scheduled_encoder_inputs: dict[str, list[int]], scheduled_running_reqs: list[Request],
                              scheduled_spec_decode_tokens: dict[str, list[int]], token_budget) -> SchedulerOutput:
        # Early return after processing priority requests

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0

        total_running = len(self.running) + len(self.priority_running)
        # assert total_running <= self.max_num_running_reqs


        structured_output_request_ids, grammar_bitmask = self.get_grammar_bitmask(
            num_scheduled_tokens.keys(), scheduled_spec_decode_tokens
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],  # No new requests from priority processing
            scheduled_cached_reqs=self._make_cached_request_data(
                scheduled_running_reqs, [], num_scheduled_tokens,
                scheduled_spec_decode_tokens, req_to_new_blocks
            ),
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=[0] * len(self.kv_cache_config.kv_cache_groups),
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        self._update_after_schedule(scheduler_output)

        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta
        return scheduler_output

    def _schedule_waiting_requests_ewsjf(self, encoder_compute_budget, num_scheduled_tokens, req_index, req_to_new_blocks,
                                         scheduled_encoder_inputs, scheduled_loras, scheduled_new_reqs, scheduled_resumed_reqs,
                                         scheduled_timestamp, skipped_waiting_requests, token_budget):
        if not self.waiting.best_queue or self.waiting.best_queue.is_empty:
            return token_budget

        with self.lock:
            while not self.waiting.best_queue.is_empty and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()
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
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                        self.lora_config
                        and request.lora_request
                        and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                )
                ):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if num_external_computed_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                            num_new_local_computed_tokens + num_external_computed_tokens
                    )
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                new_encoder_compute_budget = encoder_compute_budget

                # KVTransfer: loading remote KV, do not allocate for new work.
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # Number of tokens to be scheduled.
                else:
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if (
                            0
                            < self.scheduler_config.long_prefill_token_threshold
                            < num_new_tokens
                    ):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold
                        )

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                            not self.scheduler_config.chunked_prefill_enabled
                            and num_new_tokens > token_budget
                    ):
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                if self.is_encoder_decoder and request.has_encoder_inputs:
                    # TODO(russellb): For Whisper, we know that the input is
                    # always padded to the maximum length. If we support other
                    # encoder-decoder models, this will need to be updated if we
                    # want to only allocate what is needed.
                    num_encoder_tokens = (
                        self.scheduler_config.max_num_encoder_input_tokens
                    )
                else:
                    num_encoder_tokens = 0

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
                    # The request cannot be scheduled.
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                req_index += 1
                if self.skip_running_for_fairness:
                    self.priority_running.append(request)
                else:
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
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule
                    )
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget

            # Put back any skipped requests at the head of the waiting queue
            if skipped_waiting_requests:
                self.waiting.prepend_requests(skipped_waiting_requests)

        return token_budget

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            stats = self.connector.get_kv_connector_stats()
            if stats:
                kv_connector_stats = kv_connector_stats.aggregate(stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids
            )

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_priority_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # Skip requests that were recovered from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                request.num_computed_tokens -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len, pooler_output)

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    if request in self.priority_running:
                        stopped_priority_reqs.add(request)
                    else:
                        stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                struct_output_request.grammar.accept_tokens(req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_priority_reqs:
            self.priority_running = remove_all(self.priority_running, stopped_priority_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(spec_decoding_stats, kv_connector_stats)
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs
