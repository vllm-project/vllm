# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2025 Huawei Technologies Co., Ltd.
#
# This file was copied and modified from the vllm-ascend project (original at
# https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/core/scheduler.py @ bc69d7c)  # noqa: E501

import time
from collections import deque
from collections.abc import Iterable

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVEventBatch
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager


class AscendScheduler(Scheduler):
    """This Scheduler extends vllm's original v1 scheduler
    with prefill-first scheduling strategy."""

    def _initialize_common(self) -> None:
        """Initialize common attributes shared across all versions."""
        self.scheduled_req_ids: set[str] = set()
        self.running: list[Request] = []
        self.finished_prefill_reqs: deque[Request] = deque()

        enable_pd_transfer = getattr(self.scheduler_config, "enable_pd_transfer", False)
        decode_max_num_seqs = getattr(self.scheduler_config, "decode_max_num_seqs", 0)
        self.phase = "" if not enable_pd_transfer else "prefill"
        self.decode_max_num_running_reqs = max(
            self.max_num_running_reqs, decode_max_num_seqs
        )

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
        # Call the parent class's __init__ method
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            mm_registry,
            include_finished_set,
            log_stats,
        )

        # Initialize common attributes
        self._initialize_common()

        # Optional execution mode gate (None=auto, 1=prefill, 0=decode)
        self._forced_mode: int | None = None

    def set_forced_mode(self, mode: int | None) -> None:
        # mode must be None, 0 (decode) or 1 (prefill)
        assert mode in (None, 0, 1)
        self._forced_mode = mode

    def schedule(self) -> SchedulerOutput:
        if self.scheduler_config.chunked_prefill_enabled:
            return super().schedule()
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens

        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # Record scheduled LoRA requests.
        scheduled_loras: set[int] = set()

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later.
        skipped_waiting_requests = create_request_queue(self.policy)

        if self.phase == "prefill":
            remaining_running_reqs = []
            for request in self.running:
                # move request has finished prefill to finished_prefill_reqs
                if request.num_tokens > request.num_prompt_tokens:
                    self.finished_prefill_reqs.append(request)
                else:
                    remaining_running_reqs.append(request)
            self.running = remaining_running_reqs
            # all request prefilled, change phase to decode
            if not self.waiting and not self.running:
                self.phase = "decode"
        # Skip long prompt requests in prefill stage.
        # `long_prefill_budget` is `float("inf")` when long-prefill is disabled.
        scheduler_config = self.vllm_config.scheduler_config
        if scheduler_config.long_prefill_token_threshold == 0:
            long_prefill_budget = float("inf")
            long_prefill_token_threshold = float("inf")
        else:
            long_prefill_budget = scheduler_config.max_long_partial_prefills
            long_prefill_token_threshold = scheduler_config.long_prefill_token_threshold

        # Schedule prefill requests first (unless decode is forced).
        while self.waiting and token_budget > 0 and self._forced_mode != 0:
            if len(self.running) == (
                self.decode_max_num_running_reqs
                if self.phase == "decode"
                else self.max_num_running_reqs
            ):
                break

            request = self.waiting.peek_request()

            def skip_cur_request(request: Request = request) -> None:
                popped = self.waiting.pop_request()
                assert popped is request
                skipped_waiting_requests.prepend_request(popped)

            # P/D: skip request if still waiting for remote kvs.
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                is_ready = self._update_waiting_for_remote_kv(request)
                if is_ready:
                    request.status = RequestStatus.WAITING
                else:
                    skip_cur_request()
                    continue

            # Skip request if the structured output request is still waiting
            # for FSM compilation.
            if request.status == RequestStatus.WAITING_FOR_FSM:
                structured_output_req = request.structured_output_request
                if structured_output_req and structured_output_req.grammar:
                    # unblock request if FSM is now ready
                    request.status = RequestStatus.WAITING
                else:
                    # skip request if FSM is still not ready
                    skip_cur_request()
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
                skip_cur_request()
                continue

            num_external_computed_tokens = 0
            load_kv_async = False

            # Get already-cached tokens.
            if request.num_computed_tokens == 0:
                new_computed_blocks, num_new_local_computed_tokens = (
                    self.kv_cache_manager.get_computed_blocks(request)
                )

                # Get externally-cached tokens if using a KVConnector.
                if self.connector is not None:
                    ext_tokens, load_kv_async = (
                        self.connector.get_num_new_matched_tokens(
                            request, num_new_local_computed_tokens
                        )
                    )
                    if ext_tokens is None:
                        skip_cur_request()
                        continue
                    num_external_computed_tokens = ext_tokens

                # Total computed tokens (local + external).
                num_computed_tokens = (
                    num_new_local_computed_tokens + num_external_computed_tokens
                )
            else:
                # P/D: skip checking prefix cache if loaded from remote kvs.
                new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                num_new_local_computed_tokens = 0
                num_computed_tokens = request.num_computed_tokens

            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget

            # P/D: loading remote KV, do not allocate for new work.
            if load_kv_async:
                assert num_external_computed_tokens > 0
                num_new_tokens = 0
                blocks = None
            # Number of tokens to be scheduled.
            else:
                prompt_limit = self._get_prompt_limit(request)
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed
                # requests, which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                max_tokens_in_kvcache = (
                    self.kv_cache_config.num_blocks * self.block_size
                )
                prompt_limit = min(prompt_limit, max_tokens_in_kvcache)

                # Finish request that exceeds prompt_limit or kv cache size.
                if num_new_tokens > prompt_limit:
                    logger.warning(
                        "Input prompt (%d tokens) is too long and exceeds limit of %d",
                        num_new_tokens,
                        prompt_limit,
                    )
                    request.status = RequestStatus.FINISHED_IGNORED
                    self.finished_req_ids.add(  # type: ignore
                        request.request_id
                    )  # type: ignore
                    self.waiting.pop_request()
                    continue

                if num_new_tokens > token_budget:
                    # Scheduling would exceed token_budget, skip.
                    skip_cur_request()
                    continue
                assert num_new_tokens > 0
                blocks = new_computed_blocks.blocks[0]

                # Schedule encoder inputs.
                if request.has_encoder_inputs:
                    (
                        encoder_inputs_to_schedule,
                        num_new_tokens,
                        new_encoder_budget,
                        _,
                    ) = self._try_schedule_encoder_inputs(
                        request, num_computed_tokens, num_new_tokens, encoder_budget
                    )
                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break

            watermark = getattr(self.scheduler_config, "watermark", 0.01)
            if not self._check_watermark_for_prefill(
                request, num_new_tokens, blocks, watermark
            ):
                # Scheduling would exceed watermark, skip.
                skip_cur_request()
                continue

            if (
                num_new_tokens > long_prefill_token_threshold
                and long_prefill_budget <= 0
            ):
                skip_cur_request()
                continue

            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens + num_external_computed_tokens,
                num_new_local_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=self.num_lookahead_tokens,
                delay_cache_blocks=load_kv_async,
            )
            if new_blocks is None:
                # The request cannot be scheduled.
                break

            # KVConnector: update internal state after allocation.
            # This information is used to determine if a load is
            # needed for this request.
            if self.connector is not None:
                self.connector.update_state_after_alloc(
                    request,
                    new_computed_blocks + new_blocks,
                    num_external_computed_tokens,
                )

            self.waiting.pop_request()
            if load_kv_async:
                # If loading async, allocate memory and put request
                # into the WAITING_FOR_REMOTE_KV state.
                skipped_waiting_requests.prepend_request(request)
                request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                continue

            self.running.append(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
            self.scheduled_req_ids.add(request.request_id)
            # Check request status.
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(f"Invalid request status: {request.status}")

            if self.lora_config and request.lora_request:
                scheduled_loras.add(request.lora_request.lora_int_id)

            req_to_new_blocks[request.request_id] = self.kv_cache_manager.get_blocks(
                request.request_id
            )
            # Update request info.
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            if num_new_tokens > long_prefill_token_threshold:
                long_prefill_budget -= 1
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
                encoder_budget = new_encoder_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            for req in skipped_waiting_requests:
                self.waiting.prepend_request(req)

        if self.phase == "decode":
            while (
                len(self.running) < self.decode_max_num_running_reqs
                and self.finished_prefill_reqs
            ):
                request = self.finished_prefill_reqs.popleft()
                self.running.append(request)

        # If no prefill requests are scheduled (or prefill skipped),
        # schedule decode requests next (unless prefill is forced).
        if len(self.scheduled_req_ids) == 0 and self._forced_mode != 1:
            req_index = 0
            while req_index < len(self.running) and token_budget > 0:
                request = self.running[req_index]
                if request.request_id in self.scheduled_req_ids:
                    # This request has already been scheduled.
                    req_index += 1
                    continue

                num_new_tokens = (
                    request.num_tokens_with_spec - request.num_computed_tokens
                )
                assert (request.num_tokens - request.num_computed_tokens) == 1
                num_new_tokens = min(num_new_tokens, token_budget)
                # Make sure the input position does not exceed the max model len.
                # This is necessary when using spec decoding.
                num_new_tokens = min(
                    num_new_tokens, self.max_model_len - request.num_computed_tokens
                )

                # Schedule encoder inputs.
                encoder_inputs_to_schedule = None
                new_encoder_budget = encoder_budget
                if request.has_encoder_inputs:
                    (
                        encoder_inputs_to_schedule,
                        num_new_tokens,
                        new_encoder_budget,
                        _,
                    ) = self._try_schedule_encoder_inputs(
                        request,
                        request.num_computed_tokens,
                        num_new_tokens,
                        encoder_budget,
                    )

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
                    num_new_tokens = 0

                if num_new_tokens == 0:
                    # The request cannot be scheduled because one of the following
                    # reason:
                    # 1. No new tokens to schedule. This may happen when PP>1 and
                    #    we have already scheduled all prompt tokens but they are
                    #    not finished yet.
                    # 2. Adding the request exceeds the max_loras constraint.
                    # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                    # we do not strictly follow the FCFS scheduling policy and
                    # allow the lower-priority requests to be scheduled.
                    req_index += 1
                    continue

                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )
                    if new_blocks is None:
                        # The request cannot be scheduled.
                        # Preempt the lowest-priority request.
                        preempted_req = self.running.pop()
                        self.kv_cache_manager.free(preempted_req)
                        preempted_req.status = RequestStatus.PREEMPTED
                        preempted_req.num_computed_tokens = 0
                        if self.log_stats:
                            preempted_req.record_event(
                                EngineCoreEventType.PREEMPTED, scheduled_timestamp
                            )
                        self.waiting.prepend_request(preempted_req)
                        preempted_reqs.append(preempted_req)
                        if preempted_req == request:
                            # No more request to preempt.
                            can_schedule = False
                            break
                    else:
                        # The request can be scheduled.
                        can_schedule = True
                        break
                if not can_schedule:
                    break
                assert new_blocks is not None

                # Schedule the request.
                scheduled_running_reqs.append(request)
                self.scheduled_req_ids.add(request.request_id)
                req_to_new_blocks[request.request_id] = new_blocks
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                req_index += 1

                # Speculative decode related.
                if request.spec_token_ids:
                    num_scheduled_spec_tokens = (
                        num_new_tokens
                        + request.num_computed_tokens
                        - request.num_tokens
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
                    encoder_budget = new_encoder_budget

                # Record scheduled LoRA requests.
                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert (
            len(self.running) <= self.decode_max_num_running_reqs
            if self.phase == "decode"
            else self.max_num_running_reqs
        )
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
        scheduled_cached_reqs = cached_reqs_data
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=scheduled_cached_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,  # type: ignore
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )
        grammar_output = self.get_grammar_bitmask(scheduler_output)
        if grammar_output is not None:
            scheduler_output.structured_output_request_ids = (
                grammar_output.structured_output_request_ids
            )
            scheduler_output.grammar_bitmask = grammar_output.grammar_bitmask
        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        events = self.kv_cache_manager.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_scheduled_token

        self.finished_req_ids = set()  # type: ignore
        return scheduler_output

    def _check_watermark_for_prefill(
        self, request, num_new_tokens, computed_blocks, watermark=0.01
    ):
        computed_blocks = computed_blocks or []
        watermark_blocks = self.kv_cache_config.num_blocks * watermark
        num_computed_tokens = (
            request.num_computed_tokens + len(computed_blocks) * self.block_size
        )
        num_required_blocks = cdiv(
            num_new_tokens + num_computed_tokens, self.block_size
        )
        req_blocks = self.kv_cache_manager.coordinator.get_blocks(request.request_id)
        num_new_blocks = num_required_blocks - len(req_blocks[0]) - len(computed_blocks)
        num_evictable_computed_blocks = sum(
            1 for blk in computed_blocks if blk.ref_cnt == 0
        )
        # Don't allocate if free blocks would fall below watermark.
        free_after_alloc = (
            self.kv_cache_manager.block_pool.get_num_free_blocks()
            - num_evictable_computed_blocks
            - num_new_blocks
        )
        return free_after_alloc >= watermark_blocks

    def _get_prompt_limit(self, request: Request) -> int:
        if self.scheduler_config.chunked_prefill_enabled and not getattr(
            self.scheduler_config, "is_multi_step", False
        ):
            prompt_limit = self.vllm_config.model_config.max_model_len
        else:
            prompt_limit = min(
                self.vllm_config.model_config.max_model_len,
                self.scheduler_config.max_num_batched_tokens,
            )

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if request.lora_request and request.lora_request.long_lora_max_len:
            assert prompt_limit <= request.lora_request.long_lora_max_len
            return request.lora_request.long_lora_max_len
        else:
            return prompt_limit

    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue
            if request.status == RequestStatus.RUNNING:
                self.scheduled_req_ids.discard(request.request_id)
        super().finish_requests(request_ids, finished_status)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self.running:
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                continue
            if req_id in self.scheduled_req_ids:
                self.scheduled_req_ids.remove(req_id)

        return super().update_from_output(scheduler_output, model_runner_output)
