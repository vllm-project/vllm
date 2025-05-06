# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import time
from collections import deque

from vllm import envs
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.logger import init_logger
from vllm.sampling_params import KVTransferParams
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class DisaggregatedScheduler(Scheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # NOTE(rob): there is no reason to believe these are not
        # supported. However, I would like to test them first
        # before enabling them with P/D.
        if self.use_eagle or self.vllm_config.speculative_config:
            raise NotImplementedError(
                "Speculative Decoding is not yet supported with "
                "KV Disaggregation.")
        if self.lora_config:
            raise NotImplementedError(
                "LoRA is not yet supported with KV Disaggregation.")

        # Create KVConnector for the Scheduler.
        if self.vllm_config.kv_transfer_config is None:
            raise ValueError("Using Disaggregated Scheduler but found unset "
                             "kv_transfer_config.")
        self.connector = KVConnectorFactory.create_connector_v1(
            config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        # Requests in states for tracking KV transfers.
        self.finished_recving_kv_req_ids: set[str] = set()

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

        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to the running request index.
        # This will helps us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}

        req_to_new_block_ids: dict[str, list[int]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens

        # For logging.
        scheduled_timestamp = time.monotonic()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = (request.num_tokens_with_spec -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when PP>1 and
                #    we have already scheduled all prompt tokens but they are
                #    not finished yet.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    preempted_req = self.running.pop()
                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.appendleft(preempted_req)
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
            if request.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[request.request_id] = req_index
            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Encoder-related.
            if not request.do_remote_prefill and encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later (e.g. for FSM
        # or KVCacheSending).
        skipped_waiting_requests: deque[Request] = deque()

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting[0]

                # Skip request if the remote KV recv is still waiting
                # for the requests to arrive.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    if request.request_id in self.finished_recving_kv_req_ids:
                        assert self.kv_cache_manager.enable_caching
                        # Now that the KVs have been recved, we can cache
                        # them and set num_computed_tokens.
                        self.kv_cache_manager.cache_blocks(
                            request,
                            num_tokens=0,
                            num_computed_tokens=(len(request.all_token_ids) -
                                                 1))
                        self.finished_recving_kv_req_ids.remove(
                            request.request_id)
                        request.status = RequestStatus.WAITING
                        self.kv_cache_manager.free(request)
                    else:
                        self.waiting.popleft()
                        skipped_waiting_requests.appendleft(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.popleft()
                        skipped_waiting_requests.appendleft(request)
                        continue

                # Get already-cached tokens.
                computed_blocks, num_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(
                        request)

                # Get externally-cached tokens if using a KVConnector.
                num_external_tokens = (
                    0 if self.connector is None else
                    self.connector.get_num_new_matched_tokens(
                        request, num_computed_tokens))

                # Total computed tokens (local + external).
                num_computed_tokens += num_external_tokens

                if request.do_remote_prefill and num_external_tokens > 0:
                    # Allocate slots for the external tokens, but skip
                    # caching until after the KV transfer is done.
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_external_tokens,
                        computed_blocks,
                        skip_cache_blocks=True)
                    if new_blocks is None:
                        # Requests cannot be scheduled
                        break

                    self.waiting.popleft()
                    skipped_waiting_requests.appendleft(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS

                    # KVConnector: update internal state after allocation.
                    # This information is used to determine if a load is
                    # needed for this request.
                    if self.connector is not None:
                        self.connector.update_state_after_alloc(
                            request,
                            [
                                b.block_id for b in itertools.chain(
                                    computed_blocks, new_blocks)
                            ],
                            num_external_tokens,
                        )
                        # We should only trigger a KV transfer once per request.
                        request.do_remote_prefill = False
                    continue

                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed request,
                # which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                if (0 < self.scheduler_config.long_prefill_token_threshold <
                        num_new_tokens):
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # Schedule encoder inputs.
                if request.has_encoder_inputs:
                    (encoder_inputs_to_schedule, num_new_tokens,
                     new_encoder_budget) = self._try_schedule_encoder_inputs(
                         request, num_computed_tokens, num_new_tokens,
                         encoder_budget)
                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break
                else:
                    encoder_inputs_to_schedule = None
                    new_encoder_budget = encoder_budget

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_tokens,
                    computed_blocks,
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
                        [
                            b.block_id for b in itertools.chain(
                                computed_blocks, new_blocks)
                        ],
                        num_external_tokens,
                    )

                self.waiting.popleft()
                if request.use_structured_output:
                    structured_output_request_ids[
                        request.request_id] = req_index
                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in computed_blocks + new_blocks
                ]
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.extendleft(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = 0
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduled_spec_decode_tokens={})
        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self._make_cached_request_data(
                request=req,
                num_scheduled_tokens=num_scheduled_tokens[req.request_id],
                num_scheduled_spec_tokens=0,
                new_block_ids=req_to_new_block_ids[req.request_id],
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_cached_request_data(
                request=req,
                num_scheduled_tokens=num_scheduled_tokens[req.request_id],
                num_scheduled_spec_tokens=0,
                new_block_ids=req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
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
            if req := self.requests.get(req_id):
                req.num_computed_tokens += num_scheduled_token

        self.finished_req_ids = set()
        return scheduler_output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        new_running: list[Request] = []
        outputs: list[EngineCoreOutput] = []
        send_kv_no_op: list[str] = []

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self.running:
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                new_running.append(request)
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index]

            cached_encoder_input_ids = (
                self.encoder_cache_manager.get_cached_input_ids(request))
            # OPTIMIZATION: Avoid list(set) if the set is empty.
            if cached_encoder_input_ids:
                for input_id in list(cached_encoder_input_ids):
                    mm_positions = request.mm_positions[input_id]
                    start_pos = mm_positions.offset
                    num_tokens = mm_positions.length
                    if start_pos + num_tokens <= request.num_computed_tokens:
                        # The encoder output is already processed and stored
                        # in the decoder's KV cache.
                        self.encoder_cache_manager.free_encoder_input(
                            request, input_id)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids

            # Append generated tokens and check for stop. Note that if
            # a request is still being prefilled, we expect the model runner
            # to return empty token ids for the request.
            for num_new, output_token_id in enumerate(new_token_ids, 1):
                request.append_output_token_ids(output_token_id)

                # Check for stop and update request state.
                # This must be called before we make the EngineCoreOutput.
                stopped = check_stop(request, self.max_model_len)
                if stopped:
                    self._free_request(request)
                    del new_token_ids[num_new:]  # Trim new tokens if needed.
                    break

            # Extract sample logprobs if needed.
            if request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and request.use_structured_output:
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # check above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            # Add newly generated spec token ids to the request.
            if spec_token_ids is not None:
                if request.use_structured_output:
                    metadata = request.structured_output_request
                    assert metadata is not None and metadata.grammar is not None
                    # Needs to happen after new_token_ids are accepted.
                    request.spec_token_ids = metadata.grammar.validate_tokens(
                        spec_token_ids[req_index])
                else:
                    request.spec_token_ids = spec_token_ids[req_index]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids:
                # Stop request after the first token if doing a remote_decode.
                # NOTE(rob): req is not freed (or preempted) in the EngineCore
                # until the xfer is done to ensure we do not free the KV blocks.
                kv_transfer_params = None
                if request.do_remote_decode and not stopped:
                    request.status = RequestStatus.FINISHED_REMOTE_DECODE
                    self._free_request(request, skip_free_blocks=True)
                    stopped = True

                    # TODO(rob): do this on a per-Connector basis.
                    remote_blocks = [
                        block.block_id for block in
                        self.kv_cache_manager.req_to_blocks[request.request_id]
                        if block._block_hash is not None
                    ]
                    # If prompt < block_size, then there will be no KV xfer.
                    # Free these requests so we don't have a mem leak.
                    if len(remote_blocks) == 0:
                        send_kv_no_op.append(request.request_id)

                    engine_id = self.vllm_config.kv_transfer_config.engine_id
                    kv_transfer_params = KVTransferParams(
                        do_remote_prefill=True,
                        remote_block_ids=remote_blocks,
                        remote_engine_id=engine_id,
                        remote_host=envs.VLLM_NIXL_SIDE_CHANNEL_HOST,
                        remote_port=envs.VLLM_NIXL_SIDE_CHANNEL_PORT,
                    )

                # Add EngineCoreOutput for this Request.
                outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                    ))

            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

            if not stopped:
                new_running.append(request)

        # P/D: update recv and send status from last step.
        for req_id in (model_runner_output.finished_recving or ()):
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in (model_runner_output.finished_sending or ()):
            logger.debug("Finished sending KV transfer for request %s", req_id)
            self._free_blocks(self.requests[req_id])
        for req_id in send_kv_no_op:
            logger.debug("No op sending KV transfer for request %s", req_id)
            self._free_blocks(self.requests[req_id])

        # Return the cached request data to the queue so they can
        # be reused.
        for req_data in scheduler_output.scheduled_cached_reqs:
            # NOTE(rob): since we free stopped reqs above, adding stopped reqs
            # to _cached_reqs_data will cause a memory leak.
            if req_data.req_id not in self.finished_req_ids:
                self._cached_reqs_data[req_data.req_id].append(req_data)

        self.running = new_running
        engine_core_outputs = EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(),
        )
        if self.include_finished_set:
            #TODO currently sending duplicates here, improve this
            engine_core_outputs.finished_requests = (
                scheduler_output.finished_req_ids | self.finished_req_ids)

        return engine_core_outputs

    def _free_request(self,
                      request: Request,
                      skip_free_blocks: bool = False) -> None:
        assert request.is_finished()
        self.encoder_cache_manager.free(request)
        self._cached_reqs_data.pop(request.request_id, None)
        self.finished_req_ids.add(request.request_id)

        if not skip_free_blocks:
            self._free_blocks(request)

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        assert request.request_id not in self._cached_reqs_data
        self.kv_cache_manager.free(request)
        self.kv_cache_manager.free_block_hashes(request)
        del self.requests[request.request_id]
