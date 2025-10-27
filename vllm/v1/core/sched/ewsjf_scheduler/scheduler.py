# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# EWSJF MODIFICATION: This file implements the EWSJF policy by inheriting from
# the default vLLm v1 Scheduler and overriding its waiting queue management.

from __future__ import annotations

import time
import threading
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple, Deque, Union
from sortedcontainers import SortedDict

from vllm.v1.core.sched.ewsjf_scheduler.queue_info import QueueInfo
# EWSJF MODIFICATION: Import the parent Scheduler class to inherit from it.
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.distributed.kv_events import KVEventBatch
from vllm.v1.core.sched.ewsjf_scheduler.scoring import SimpleScoreCalculator
import itertools
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    compute_encoder_budget,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


class EWSJFScheduler(Scheduler):
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
        super().__init__(vllm_config, kv_cache_config, structured_output_manager,
                         block_size, mm_registry, include_finished_set, log_stats)

        # EWSJF MODIFICATION: Initialize with the new queue structure
        self.external_parameters = self.vllm_config.scheduler_config.external_parameters
        self.lock = threading.Lock()
        if self.external_parameters and 'step_size' in self.external_parameters:
            self.step_size: int = self.external_parameters['step_size']
        else:
            self.step_size: int = 200  # Default queue size range

        self.empty_queue_threshold: int = 20  # Cycles before removing empty queue
        self.current_time = None  # Current timestamp for score calculations
        if self.external_parameters and 'score_calculator' in self.external_parameters:
            self.score_calculator = self.external_parameters['score_calculator']
        else:
            self.score_calculator = SimpleScoreCalculator(weighting_factor=0.5)

        # Core EWSJF data structures
        self.queues: SortedDict = SortedDict()  # key (low_boundary) -> QueueInfo mapping
        self.best_queue: Optional[QueueInfo] = None  # Currently highest-scoring queue
        self.request_partial_scores: Dict[str, float] = {}  # Cache for partial scores

        # Initialize queues either from config or with defaults
        if self.external_parameters and 'queues_config' in self.external_parameters:
            self._initialize_queues_by_config(self.external_parameters['queues_config'])
        else:
            self._initialize_queues(num_queues=10)

        # EWSJF MODIFICATION: Start the background optimizer thread.
        self.update_event = threading.Event()  # Signal to start score update
        self.finish_update_event = threading.Event()  # Signal when update is done
        self.update_stop_event = threading.Event()  # Signal to stop the thread
        self.update_thread = threading.Thread(target=self._update_scores_loop, daemon=True)
        self.update_thread.start()

    def add_request(self, request: Request) -> None:
        """
        Override the parent add_request method to implement EWSJF queue assignment.

        Instead of adding to a single waiting queue, this method dispatches the request
        into one of the EWSJF queues based on the prompt length. If no suitable queue
        exists, a new one is created.

        Args:
            request (Request): The incoming request to be queued
        """
        # Add to the global requests dictionary (inherited from parent)
        self.requests[request.request_id] = request

        # Determine which queue should handle this request based on prompt length
        prompt_len = len(request.prompt_token_ids)

        with self.lock:

            queue = self._get_queue_by_length(prompt_len)

            # Create a new queue if none exists for this length
            if queue is None:
                queue = self._add_queue(prompt_len)

            # Add the request to the appropriate queue
            queue.add_request(request)

        # Log the queuing event if statistics are enabled
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    # --- EWSJF MODIFICATION: Helper methods for queue management ---
    def _preempt_request(self, request: Request):
        """
        Handle a preempted request by returning it to the appropriate queue.

        Preempted requests are added to the front of their queue to maintain
        priority over new requests.

        Args:
            request (Request): The preempted request to re-queue
        """
        prompt_len = len(request.prompt_token_ids)
        queue = self._get_queue_by_length(prompt_len)
        if queue is None:
            queue = self._add_queue(prompt_len)
        queue.add_request_front(request)

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

                self._preempt_request(preempted_req)  # Put back into our queues

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
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

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

    def _schedule_waiting_requests_ewsjf(self, encoder_compute_budget, num_scheduled_tokens, req_index, req_to_new_blocks,
                                         scheduled_encoder_inputs, scheduled_loras, scheduled_new_reqs, scheduled_resumed_reqs,
                                         scheduled_timestamp, skipped_waiting_requests, token_budget):
        if not self.best_queue or self.best_queue.is_empty:
            return token_budget

        with self.lock:
            while not self.best_queue.is_empty and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.best_queue.peek_request()
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
                        self.best_queue.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.best_queue.pop_request()
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
                    self.best_queue.pop_request()
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
                            self.best_queue.pop_request()
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
                        self.best_queue.pop_request()
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

                # Request was already popped from self.best_queue
                # unless it was re-added above due to new_blocks being None.
                request = self.best_queue.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
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
                self.best_queue.add_requests_front(skipped_waiting_requests)

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
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self._remove_waiting_requests(stopped_preempted_reqs)

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

    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self._remove_waiting_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        total_requests_to_reschedule = 0
        total_tokens_to_reschedule = 0

        # --- Handle async KV loads (WAITING_FOR_REMOTE_KVS) ---
        async_load_reqs = (
            req
            for queue in self.queues
            for req in queue.requests
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_affected_req_ids, num_tokens_to_reschedule = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs, invalid_block_ids
            )
        )

        total_requests_to_reschedule += len(async_affected_req_ids)
        total_tokens_to_reschedule += num_tokens_to_reschedule

        # Mark requests with async KV load failures; they will be rescheduled
        # once loading completes.
        self.failed_recving_kv_req_ids |= async_affected_req_ids

        # --- Handle sync KV loads (running requests) ---
        sync_affected_req_ids, num_tokens_to_reschedule = (
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

        # Return the IDs of affected running requests to skip in
        # update_from_output.
        return sync_affected_req_ids

    def _remove_waiting_requests(self, requests_to_remove):
        for req in requests_to_remove:
            found = False
            for queue in self.queues.values():
                if queue.remove_request(req):
                    found = True
                    break
            if not found:
                logger.warning(f"Request {req.request_id} not found in any waiting queue.")

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None

        running_len, waiting_len = self.get_request_counts()
        return SchedulerStats(
            num_running_reqs=running_len,
            num_waiting_reqs=waiting_len,
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            num_corrupted_reqs=sum(req.is_output_corrupted for req in self.running),
            kv_connector_stats=kv_connector_stats.data if kv_connector_stats else None,
        )

    def get_num_unfinished_requests(self) -> int:
        num_running, num_waiting = self.get_request_counts()
        return num_waiting + num_running

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        with self.lock:
            # Sum all requests across all EWSJF queues
            num_waiting = sum(queue.size for queue in self.queues.values())
        return len(self.running), num_waiting

    def _initialize_queues(self, num_queues: int = 4):
        """
        Initialize a default set of queues with equal-sized ranges.

        Args:
            num_queues (int, optional): Number of initial queues to create. Defaults to 4.
        """
        for i in range(num_queues):
            boundaries = (i * self.step_size, (i + 1) * self.step_size - 1)
            self._create_queue(boundaries, False)

    def _initialize_queues_by_config(self, queues_config: list):
        """
        Initialize queues based on configuration file.

        Args:
            queues_config (list): List of queue configuration dictionaries,
                                each containing 'boundaries' key
        """
        for q in queues_config:
            self._create_queue(q['boundaries'], False)

    def _update_scores_loop(self):
        """
        Background thread function that continuously updates queue scores.

        This method runs in a separate thread and waits for signals from the
        main scheduling thread to update scores for all queues.
        """
        while not self.update_stop_event.is_set():
            # Wait for update event from schedule()
            self.update_event.wait()
            self.update_event.clear()
            self._update_scores()
            self.finish_update_event.set()

    def _update_scores(self):
        """
        Update EWSJF scores for all queues and identify the best queue.

        This method calculates scores for each non-empty queue, handles empty
        queue removal, and updates the best_queue pointer to the highest-scoring queue.
        """
        new_best_queue = None
        new_best_score = -1.0
        queues_to_remove = []

        # Iterate through all queues and update their scores
        with self.lock:
            for queue in self.queues.values():
                if queue.is_empty:
                    if queue.removable:
                        queue.increment_empty_count()
                        # Mark for removal if empty too long
                        if queue.empty_count >= self.empty_queue_threshold and len(self.queues) > 1:
                            queues_to_remove.append(queue)
                    else:
                        # Non-removable empty queues get score 0
                        queue.update_score(0.0)
                    continue

                # Queue has requests - calculate score
                queue.reset_empty_count()
                first_req = queue.peek_request()
                if not first_req:
                    continue

                # Get or calculate partial score (cached for efficiency)
                partial_score = self.request_partial_scores.get(first_req.request_id, 0.0)
                if partial_score == 0.0:
                    partial_score = self.score_calculator.get_partial_score(first_req, self.step_size)
                    self.request_partial_scores[first_req.request_id] = partial_score

                # Calculate final EWSJF score
                score = self.score_calculator.complete_score(first_req, partial_score, self.current_time)
                queue.update_score(score)

                # Track the highest scoring queue
                if score > new_best_score:
                    new_best_score = score
                    new_best_queue = queue

        # Update the best queue pointer
        self.best_queue = new_best_queue

        # Remove queues that have been empty too long
        with self.lock:
            for queue in queues_to_remove:
                self._remove_queue(queue)

    def _get_queue_by_length(self, length: int) -> Optional[QueueInfo]:
        """
        Find the queue that should contain requests of the given prompt length.

        Uses binary search on the sorted queue boundaries for O(log n) complexity.

        Args:
            length (int): The prompt length to find a queue for

        Returns:
            Optional[QueueInfo]: The queue that handles this length, or None if not found
        """
        # Use binary search to find the appropriate queue
        idx = self.queues.bisect_right(length)

        if idx > 0:
            key = self.queues.iloc[idx - 1]
            queue = self.queues[key]
            if queue.contains_length(length):
                return queue

        return None

    def _add_queue(self, length: int) -> QueueInfo:
        """
        Add a new queue for the given prompt length.

        This method dynamically creates a new queue with appropriate boundaries
        based on existing queues and the specific length requirement.

        Args:
            length (int): The prompt length that triggered queue creation

        Returns:
            QueueInfo: The newly created queue
        """
        # Find the insertion position using SortedDict
        # bisect_right finds the position where length should be inserted
        insert_idx = self.queues.bisect_right(length)

        # Get the previous queue (if exists) using the SortedDict
        prev_queue = None
        if insert_idx > 0:
            prev_key = self.queues.iloc[insert_idx - 1]
            prev_queue = self.queues[prev_key]

        # Get the next queue (if exists) using the SortedDict
        next_queue = None
        if insert_idx < len(self.queues):
            next_key = self.queues.iloc[insert_idx]
            next_queue = self.queues[next_key]

        # Determine boundaries for the new queue based on adjacent queues
        if prev_queue is not None:
            prev_range = prev_queue.high_boundary - prev_queue.low_boundary
            half_range = prev_range // 2
            lower = max(prev_queue.high_boundary + 1, length - half_range)
            upper = lower + prev_range
        else:
            # No previous queue - this is the first queue
            prev_range = 100  # Default range
            lower = 0
            upper = 100

        # Adjust upper boundary if there's a next queue
        if next_queue is not None:
            upper = min(next_queue.low_boundary - 1, upper)

        # Ensure minimum range
        if upper - lower < prev_range:
            half_range = prev_range // 2
            lower = max(0, length - half_range)
            upper = lower + prev_range

        return self._create_queue((lower, upper))

    def _create_queue(self, boundaries: Tuple[int, int], removable: bool = True) -> QueueInfo:
        """
        Create a new queue with specified boundaries.

        Args:
            boundaries (Tuple[int, int]): (min_length, max_length) for the queue
            removable (bool, optional): Whether queue can be auto-removed. Defaults to True.

        Returns:
            QueueInfo: The newly created queue
        """
        print(f'add queue: {boundaries}')
        queue_id = str(boundaries[0])
        new_queue = QueueInfo(queue_id, boundaries, removable)
        # Add to the sorted dictionary using low boundary as key
        self.queues[boundaries[0]] = new_queue
        return new_queue

    def _remove_queue(self, queue_to_remove: QueueInfo):
        """
        Remove a queue and redistribute its requests to appropriate queues.

        Args:
            queue_to_remove (QueueInfo): The queue to be removed
        """
        if queue_to_remove.low_boundary in self.queues:
            # Get any remaining requests before removal
            remaining_requests = queue_to_remove.get_all_requests()

            # Remove from the sorted dictionary
            del self.queues[queue_to_remove.low_boundary]

            # Redistribute remaining requests to appropriate queues
            for req in remaining_requests:
                self.add_request(req)

    def shutdown(self):
        """
        Shutdown the scheduler and clean up resources.

        This method stops the background scoring thread and calls the parent's
        shutdown method to clean up inherited resources.
        """
        super().shutdown()
        # Stop the background thread gracefully
        self.update_stop_event.set()
        self.update_thread.join(timeout=1)
