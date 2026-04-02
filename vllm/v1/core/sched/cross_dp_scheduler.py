from ast import Set
import itertools
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any
from itertools import chain

from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.sched.dynamic_bucket_load_balancer import NoStandardBucketLoadBalancer
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.cross_dp_kv_cache_manager import CrossDPKVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue, LongShortRequestQueue
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import (
    PrefixCacheStats,
    SchedulerStats,
)
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

from vllm.v1.core.sched.scheduler import Scheduler


logger = init_logger(__name__)

class RequestManager:
    def __init__(
        self,
        cp_world_size: int,
        max_num_seqs: int,
        dynamic_cp_threshold: int,
    ):
        self.cp_world_size = cp_world_size
        self.max_num_seqs = max_num_seqs
        self.num_long_req_per_domain = 0
        self.num_req_per_dp = [0] * self.cp_world_size

        self.balancer = NoStandardBucketLoadBalancer(
            num_buckets=self.cp_world_size,
            max_length=dynamic_cp_threshold)

    def select_dp(self, request: Request, is_long: bool, num_new_tokens: int) -> list[int] | None:
        if len(request.cp_ranks) > 0:
            if all([self.num_req_per_dp[rank] < self.max_num_seqs for rank in request.cp_ranks]):
                return request.cp_ranks
            else:
                return None

        if is_long:
            return [
                i for i in range(self.cp_world_size)
            ]
        else:
            # Get the the dp with the least number of requests
            # best_dp = min(range(len(self.num_req_per_dp)), key=lambda i: self.num_req_per_dp[i])
            best_dp = self.balancer.dispatch_task_without_id(num_new_tokens)
            return [best_dp]

    def add_req(self, request: Request) -> None:
        if len(request.cp_ranks) > 1:
            self.num_long_req_per_domain += 1

        for rank in request.cp_ranks:
            self.num_req_per_dp[rank] += 1

    def free_req(self, request: Request) -> None:
        if len(request.cp_ranks) > 1:
            self.num_long_req_per_domain -= 1

        for rank in request.cp_ranks:
            self.num_req_per_dp[rank] -= 1

    def get_num_req_per_dp(self, dp_rank: int) -> int:
        return self.num_req_per_dp[dp_rank]

    def get_num_cp_req_per_domain(self) -> int:
        return self.num_long_req_per_domain

    def get_total_num_req(self) -> int:
        return sum(self.num_req_per_dp) - self.num_long_req_per_domain * (self.cp_world_size - 1)

    def has_slot_for_long_request(self) -> bool:
        return all(self.num_req_per_dp[i] < self.max_num_seqs for i in range(self.cp_world_size))

    def __repr__(self) -> str:
        return (f"RequestManager(cp_world_size={self.cp_world_size}"
                + f"max_num_seqs={self.max_num_seqs}"
                + f"max_num_seqs={self.max_num_seqs}, "
                + f"num_long_req_per_domain={self.num_long_req_per_domain}, "
                + f"num_req_per_dp={self.num_req_per_dp})")

class CrossDPScheduler(Scheduler):
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
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        self.cp_world_size = vllm_config.parallel_config.dp_per_domain
        self.finished_req_ids: list[set[str]] = [set() for _ in range(self.cp_world_size)]

        self.kv_cache_manager = CrossDPKVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            cp_world_size=self.cp_world_size,
            hash_block_size=self.block_size,
            metrics_collector=self.kv_metrics_collector,
        )
        self.max_cp_tokens = self.vllm_config.scheduler_config.num_cp_seqs
        # self.graph_size_for_cp = self.vllm_config.compilation_config.cudagraph_capture_sizes_for_cp
        self.graph_size_for_cp = self.scheduler_config.num_cp_seqs
        assert self.max_cp_tokens >= self.graph_size_for_cp, "max_cp_tokens should be greater than or equal to graph_size_for_cp"
        self.dynamic_cp_threshold = 1 * 1024
        # Request queue control the token threshold for long requests.
        self.waiting = LongShortRequestQueue(
            long_request_threshold=self.dynamic_cp_threshold,
            max_long_requests=self.max_cp_tokens,
        )
        self.request_manager = RequestManager(
            cp_world_size=self.cp_world_size,
            max_num_seqs=self.max_num_running_reqs,
            dynamic_cp_threshold=self.dynamic_cp_threshold,
        )
        self.requests_to_free_blocks: set[Request] = set()

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            if (
                len(request.cp_ranks) == 1
                or (len(request.cp_ranks) > 1 and scheduler_output.cp_rank == 0)
            ):
                request.num_computed_tokens += num_scheduled_token

            # [vllm added]
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output
            )

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids[scheduler_output.cp_rank] = set()

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()

        """
        TODO(AoChen): If the req is removed from the running queue,
        1. the running_long_count should be decremented.
        2. the request manager should be updated.
        3. the has_slot_for_long_request should be updated.
        """
        self.waiting.running_long_count -= 1 if self.waiting.is_long_request(request) else 0
        self.request_manager.free_req(request)
        self.waiting.has_slot_for_long_request = self.request_manager.has_slot_for_long_request()

        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        for cp_rank in request.cp_ranks:
            self.finished_req_ids[cp_rank].add(request_id)

        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        # [vllm added]
        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def has_finished_requests(self) -> bool:
        return sum(len(sub_ids) for sub_ids in self.finished_req_ids) > 0

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # [vllm added]
        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        # 等后续调优后再释放
        # self.kv_cache_manager.remove_skipped_blocks(
        #     request_id=request.request_id,
        #     total_computed_tokens=request.num_tokens,
        # )

        block_ids = self.kv_cache_manager.get_block_ids(request)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, [block_id[0] for block_id in block_ids])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            num_computed_tokens = 0
            for (block_ids,) in self.kv_cache_manager.get_block_ids(request):
                num_computed_tokens += len(block_ids) * self.block_size
            # Handle the case where num request tokens less than one block.
            num_computed_tokens = min(num_computed_tokens, request.num_tokens)
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

            # Update the request state for scheduling.
            request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self.requests_to_free_blocks.add(self.requests[req_id])
                # self._free_blocks(self.requests[req_id])

        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self.requests_to_free_blocks.add(self.requests[req_id])
            # self._free_blocks(self.requests[req_id])

    def update_from_output(
        self,
        scheduler_outputs: list[SchedulerOutput],
        model_runner_outputs: list[ModelRunnerOutput],
    ) -> dict[int, EngineCoreOutputs]:

        """
        Due to we use example connector now, many stats are None.
        So, the scheduler_stats only contain the last dp stats.
        """
        processed_request: list[str] = []
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

        for scheduler_output, model_runner_output in zip(scheduler_outputs, model_runner_outputs):
            if model_runner_output is False:
                continue

            sampled_token_ids = model_runner_output.sampled_token_ids
            logprobs = model_runner_output.logprobs
            prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            pooler_outputs = model_runner_output.pooler_output
            num_nans_in_logits = model_runner_output.num_nans_in_logits
            kv_connector_output = model_runner_output.kv_connector_output
            cudagraph_stats = model_runner_output.cudagraph_stats

            # [vllm add]
            perf_stats: PerfStats | None = None
            if self.perf_metrics and self.perf_metrics.is_enabled():
                perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

            # outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
            spec_decoding_stats: SpecDecodingStats | None = None
            kv_connector_stats: KVConnectorStats | None = (
                kv_connector_output.kv_connector_stats if kv_connector_output else None
            )
            if kv_connector_stats and self.connector:
                kv_stats = self.connector.get_kv_connector_stats()
                assert kv_stats is None, "Where example connector kv_stats is None, if not, implemented it"
                if kv_stats:
                    kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

            failed_kv_load_req_ids = None
            if kv_connector_output and kv_connector_output.invalid_block_ids:
                assert False, "This is unreachable"
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
                """
                Remove the duplicate model output
                """
                if req_id in processed_request:
                    continue

                assert num_tokens_scheduled > 0
                if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                    # skip failed or rescheduled requests from KV load failure
                    continue
                request = self.requests.get(req_id)
                # [vllm add]
                if request is None or request.is_finished():
                    # The request is already finished. This can happen if the
                    # request is aborted while the model is executing it (e.g.,
                    # in pipeline parallelism or in async scheduling).
                    # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                    # cache transfer in KV connector), the aborted request will not
                    # be set to None (in order to finish async KV transfer).
                    # In this case, we use is_finished() to check.
                    continue

                req_index = model_runner_output.req_id_to_index[req_id]
                generated_token_ids = (
                    sampled_token_ids[req_index] if sampled_token_ids else []
                )

                scheduled_spec_token_ids = (
                    scheduler_output.scheduled_spec_decode_tokens.get(req_id)
                )
                # [vllm add]
                if scheduled_spec_token_ids and generated_token_ids:
                    num_draft_tokens = len(scheduled_spec_token_ids)
                    num_accepted = len(generated_token_ids) - 1
                    num_rejected = num_draft_tokens - num_accepted
                    # num_computed_tokens represents the number of tokens
                    # processed in the current step, considering scheduled
                    # tokens and rejections. If some tokens are rejected,
                    # num_computed_tokens is decreased by the number of rejected
                    # tokens.
                    if request.num_computed_tokens > 0:
                        request.num_computed_tokens -= num_rejected
                    # If async scheduling, num_output_placeholders also includes
                    # the scheduled spec tokens count and so is similarly adjusted.
                    if request.num_output_placeholders > 0:
                        request.num_output_placeholders -= num_rejected

                    # [vllm add]
                    spec_decoding_stats = self.make_spec_decoding_stats(
                        spec_decoding_stats,
                        num_draft_tokens=num_draft_tokens,
                        num_accepted_tokens=num_accepted,
                        num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                        request_id=req_id,
                    )

                stopped = False
                new_logprobs = None
                new_token_ids = generated_token_ids
                pooler_output = pooler_outputs[req_index] if pooler_outputs else None
                kv_transfer_params = None
                status_before_stop = request.status

                # Check for stop and update request status.
                if new_token_ids:
                    new_token_ids, stopped = self._update_request_with_output(
                        request, new_token_ids
                    )
                elif request.pooling_params and pooler_output is not None:
                    # Pooling stops as soon as there is output.
                    request.status = RequestStatus.FINISHED_STOPPED
                    stopped = True

                # [vllm add]
                routed_experts = None
                finish_reason = None
                if stopped:
                    routed_experts = self._get_routed_experts(request)

                    # Capture finish_reason BEFORE _handle_stopped_request, which may
                    # reset the status to WAITING for streaming requests that continue.
                    finish_reason = request.get_finished_reason()
                    finished = self._handle_stopped_request(request)
                    if finished:
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
                    new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

                if new_token_ids and self.structured_output_manager.should_advance(request):
                    struct_output_request = request.structured_output_request
                    assert struct_output_request is not None
                    assert struct_output_request.grammar is not None

                    # [vllm add]
                    ok = struct_output_request.grammar.accept_tokens(req_id, new_token_ids)
                    if not ok:
                        logger.warning(
                            "Unexpected: grammar rejected tokens %s for request %s.",
                            new_token_ids,
                            req_id,
                        )

                if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                    request.num_nans_in_logits = num_nans_in_logits[req_id]

                # Get prompt logprobs for this request.
                prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)

                # [vllm add]
                if (
                    new_token_ids
                    or pooler_output is not None
                    or kv_transfer_params
                    or stopped
                ):
                    # Add EngineCoreOutput for this Request.
                    outputs[request.client_index].append(
                        EngineCoreOutput(
                            request_id=req_id,
                            new_token_ids=new_token_ids,
                            finish_reason=finish_reason,
                            new_logprobs=new_logprobs,
                            new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                            pooling_output=pooler_output,
                            stop_reason=request.stop_reason,
                            events=request.take_events(),
                            kv_transfer_params=kv_transfer_params,
                            trace_headers=request.trace_headers,
                            num_cached_tokens=request.num_cached_tokens,
                            num_external_computed_tokens=request.num_external_computed_tokens,
                            routed_experts=routed_experts,
                            num_nans_in_logits=request.num_nans_in_logits,
                        )
                    )
                else:
                    # Invariant: EngineCore returns no partial prefill outputs.
                    assert not prompt_logprobs_tensors

                processed_request.append(req_id)

            # Remove the stopped requests from the running and waiting queues.
            if stopped_running_reqs:
                self.running = remove_all(self.running, stopped_running_reqs)
            if stopped_preempted_reqs:
                # This is a rare case and unlikely to impact performance.
                self.waiting.remove_requests(stopped_preempted_reqs)

            if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
                requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
                self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
                for request in requests:
                    outputs[request.client_index].append(
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[],
                            finish_reason=request.get_finished_reason(),
                            events=request.take_events(),
                            trace_headers=request.trace_headers,
                            num_cached_tokens=request.num_cached_tokens,
                        )
                    )

            # KV Connector: update state for finished KV Transfers.
            if kv_connector_output:
                self._update_from_kv_xfer_finished(kv_connector_output)

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

            # Create EngineCoreOutputs for all clients that have requests with
            # outputs in this step.
            engine_core_outputs = {
                client_index: EngineCoreOutputs(outputs=outs)
                for client_index, outs in outputs.items()
            }

        # free blocks
        for request_to_free in self.requests_to_free_blocks:
            self._free_blocks(request_to_free)
        self.requests_to_free_blocks.clear()

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

        # [vllm add]
        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def schedule(self) -> list[SchedulerOutput]:
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

        scheduled_new_reqs: list[list[Request]] = [[] for _ in range(self.cp_world_size)]
        scheduled_resumed_reqs: list[list[Request]] = [[] for _ in range(self.cp_world_size)]
        scheduled_running_reqs: list[list[Request]] = [[] for _ in range(self.cp_world_size)]
        preempted_reqs: list[list[Request]] = [[] for _ in range(self.cp_world_size)]

        req_to_new_blocks: list[dict[str, KVCacheBlocks]] = [{} for _ in range(self.cp_world_size)]
        num_scheduled_tokens: list[dict[str, int]] = [{} for _ in range(self.cp_world_size)]
        cp_rank_scheduled_tokens: list[dict[str, int]] = [{} for _ in range(self.cp_world_size)]

        """
        TODO(AoChen): Token budget for each DCP rank is not implemented yet.
        """
        token_budget = self.max_num_scheduled_tokens

        # # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )

            """
            TODO(AoChen): Long prefill token threshold is not implemented yet. We temparily ignore this for decode instance.
            """
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            """
            TODO(AoChen): Encoder inputs scheduling is not implemented yet.
            """

            # [vllm add]
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
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
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request.cp_ranks,
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )
                    logger.debug(f"new_blocks: {new_blocks}, request.cp_ranks: {request.cp_ranks}, num_new_tokens: {num_new_tokens}")
                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    """
                    TODO(AoChen): PRIORITY is not implemented yet.
                    """
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        raise NotImplementedError
                    else:
                        preempted_req = self.running.pop()
                        """
                        TODO(AoChen): Preempted request is also need to be removed from the request manager.
                        """
                        self.request_manager.free_req(preempted_req)
                        self.waiting.running_long_count -= 1 if self.waiting.is_long_request(preempted_req) else 0
                        self.waiting.has_slot_for_long_request = self.request_manager.has_slot_for_long_request()

                    self._preempt_request(preempted_req, scheduled_timestamp)

                    if len(preempted_req.cp_ranks) > 1:
                        raise RuntimeError("Preempted request has multiple CP ranks is not supported now.")

                    for rank in preempted_req.cp_ranks:
                        preempted_reqs[rank].append(preempted_req)

                    # preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            assert len(request.cp_ranks) == len(new_blocks)
            # Schedule the request.
            for i, rank in enumerate(request.cp_ranks):
                scheduled_running_reqs[rank].append(request)
                request_id = request.request_id
                req_to_new_blocks[rank][request_id] = new_blocks[i]
                num_scheduled_tokens[rank][request_id] = num_new_tokens
                cp_rank_scheduled_tokens[rank][request_id] = len(request.cp_ranks)

            token_budget -= num_new_tokens
            req_index += 1

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        if not any(preempted_reqs):
            while self.waiting and token_budget > 0:
                if len(self.running) == (
                    (self.max_num_running_reqs - self.waiting.running_long_count) * self.cp_world_size + self.waiting.running_long_count
                ):
                    break
                request = self.waiting.peek_request()
                if request is None:
                    break

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        # [vllm add]
                        if request.num_preemptions:
                            # We must be loading for a resumed preemption
                            # rather than a new request.
                            request.status = RequestStatus.PREEMPTED
                        else:
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
                # if request.status == RequestStatus.WAITING_FOR_FSM:
                #     structured_output_req = request.structured_output_request
                #     if structured_output_req and structured_output_req.grammar:
                #         request.status = RequestStatus.WAITING
                #     else:
                #         self.waiting.pop_request()
                #         skipped_waiting_requests.prepend_request(request)
                #         continue

                # Streaming: skip request if still waiting for next streaming req.
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    assert not request.streaming_queue
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
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
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                    # [vllm add]
                    connector_prefix_cache_queries = (
                        request.num_tokens - num_new_local_computed_tokens
                    )
                    connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                # encoder_inputs_to_schedule = None
                # external_load_encoder_input = []
                # new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                # [vllm add]
                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                num_encoder_tokens = (
                    self._num_encoder_max_input_tokens
                    if self.is_encoder_decoder and request.has_encoder_inputs
                    else 0
                )

                if len(request.cp_ranks) == 0:
                    selected_dp = self.request_manager.select_dp(
                        request,
                        self.waiting.is_long_request(request),
                        num_new_tokens,
                    )
                else:
                    selected_dp = self.request_manager.select_dp(
                        request,
                        self.waiting.is_long_request(request),
                        num_new_tokens,
                    )
                    if selected_dp is None:
                        break

                if len(selected_dp) > 1:
                    logger.info(f"It's a cp req, selected_dp: {selected_dp}, request id: {request.request_id}")
                else:
                    logger.info(f"It's a short req, selected_dp: {selected_dp}, request id: {request.request_id}")

                # [vllm add]
                new_blocks = self.kv_cache_manager.allocate_slots(
                    selected_dp,
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )
                logger.debug(f"new_blocks -- 2: {new_blocks}, request.cp_ranks: {request.cp_ranks}, num_new_tokens: {num_new_tokens}")
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                request.cp_ranks = selected_dp

                """
                TODO(AoChen): update_state_after_alloc(PD disagg) is not implemented yet.
                """
                if self.connector is not None:
                    """
                        In the example connector, new_computed_blocks + new_blocks is not used,
                        So, temparily ignore it.
                    """
                    self.connector.update_state_after_alloc(
                        request=request,
                        # new_computed_blocks + new_blocks,
                        blocks=self.kv_cache_manager.get_blocks(request),
                        num_external_tokens=num_external_computed_tokens,
                    )

                    # [vllm add]
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
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

                self.running.append(request)
                self.waiting.running_long_count += 1 if self.waiting.is_long_request(request) else 0
                self.request_manager.add_req(request)
                self.waiting.has_slot_for_long_request = self.request_manager.has_slot_for_long_request()

                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )

                blocks = self.kv_cache_manager.get_blocks(request)

                for idx, rank in enumerate(request.cp_ranks):
                    if request.status == RequestStatus.WAITING:
                        scheduled_new_reqs[rank].append(request)
                    elif request.status == RequestStatus.PREEMPTED:
                        scheduled_resumed_reqs[rank].append(request)
                    else:
                        raise RuntimeError(f"Invalid request status: {request.status}")

                    req_to_new_blocks[rank][request.request_id] = blocks[idx]

                    num_scheduled_tokens[rank][request.request_id] = num_new_tokens
                    cp_rank_scheduled_tokens[rank][request.request_id] = len(request.cp_ranks)

                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        """
        TODO(AoChen): total_num_scheduled_tokens scheduling constraints are not implemented yet.
        """
        total_num_scheduled_tokens = sum([sum(scheduled_tokens.values()) for scheduled_tokens in num_scheduled_tokens])
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens * self.cp_world_size

        assert token_budget >= 0
        assert len(self.running) <= (
            (self.max_num_running_reqs - self.waiting.running_long_count) * self.cp_world_size + self.waiting.running_long_count
        )

        total_scheduled = (
            len(list(chain.from_iterable(scheduled_new_reqs)))
            + len(list(chain.from_iterable(scheduled_resumed_reqs)))
            + len(list(chain.from_iterable(scheduled_running_reqs)))
        )
        assert total_scheduled <= self.max_num_running_reqs * self.cp_world_size

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)

        assert sum(len(sub) for sub in scheduled_resumed_reqs) == 0, "Scheduled resumed requests are not supported now."

        self.request_manager.balancer.release_all_tasks()

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            raise NotImplementedError
        else:
            total_new_reqs_data = [
                [
                    NewRequestData.from_request(
                        req, req_to_new_blocks[idx][req.request_id].get_block_ids()
                    )
                    for req in scheduled_new_reqs[idx]
                ] for idx in range(self.cp_world_size)
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            total_cached_reqs_data = [
                    self._make_cached_request_data(
                        scheduled_running_reqs[idx],
                        scheduled_resumed_reqs[idx],
                        num_scheduled_tokens[idx], # num_scheduled_tokens should be refined num_scheduled_tokens[idx]
                        scheduled_spec_decode_tokens,
                        req_to_new_blocks[idx], # req_to_new_blocks should be refined req_to_new_blocks[idx]
                    ) for idx in range(self.cp_world_size)
            ]

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        for scheduled_tokens in num_scheduled_tokens:
            self.prev_step_scheduled_req_ids.update(scheduled_tokens.keys())

        total_scheduler_output = []

        none_tokens_in_peer_sched = all([sum(num_scheduled_tokens[idx].values()) == 0 for idx in range(self.cp_world_size)])

        for idx in range(self.cp_world_size):

            if sum(num_scheduled_tokens[idx].values()) == 0 and len(preempted_reqs[idx]) == 0 and len(self.finished_req_ids[idx]) == 0:
                scheduler_output = SchedulerOutput.make_empty()
                scheduler_output.none_tokens_in_peer_sched = none_tokens_in_peer_sched
                total_scheduler_output.append(scheduler_output)
            else:
                total_scheduler_output.append(
                    SchedulerOutput(
                        scheduled_new_reqs=total_new_reqs_data[idx],
                        scheduled_cached_reqs=total_cached_reqs_data[idx],
                        num_scheduled_tokens=num_scheduled_tokens[idx], # num_scheduled_tokens should be refined
                        total_num_scheduled_tokens=sum(num_scheduled_tokens[idx].values()), # num_scheduled_tokens should be refined
                        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens, # num_scheduled_tokens should be refined
                        scheduled_encoder_inputs=scheduled_encoder_inputs, # num_scheduled_tokens should be refined
                        num_common_prefix_blocks=num_common_prefix_blocks, # num_scheduled_tokens should be refined
                        preempted_req_ids={req.request_id for req in preempted_reqs[idx]},
                        # finished_req_ids is an existing state in the scheduler,
                        # instead of being newly scheduled in this step.
                        # It contains the request IDs that are finished in between
                        # the previous and the current steps.
                        finished_req_ids=self.finished_req_ids[idx],
                        free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
                        cp_rank=idx,
                        cp_rank_scheduled_tokens=cp_rank_scheduled_tokens[idx],
                        num_cp_request=sum([1 if cp_size > 1 else 0 for cp_size in  cp_rank_scheduled_tokens[idx].values()]),
                        none_tokens_in_peer_sched=none_tokens_in_peer_sched
                    )
                )
        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            for scheduler_output in total_scheduler_output:
                if scheduler_output is None:
                    continue
                meta: KVConnectorMetadata = self.connector.build_connector_meta(
                    scheduler_output
                )
                scheduler_output.kv_connector_metadata = meta
            self.connector.clear_reqs_need_recv() # debug

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            # self._update_after_schedule(scheduler_output)
            for scheduler_output in total_scheduler_output:
                if scheduler_output is None:
                    continue
                self._update_after_schedule(scheduler_output)

        return total_scheduler_output


class AsyncCrossDPScheduler(CrossDPScheduler):
    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        super()._update_after_schedule(scheduler_output)
        pending_structured_output_tokens = False
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            if (
                request.num_computed_tokens
                == request.num_tokens
                + request.num_output_placeholders
                + cur_num_spec_tokens
            ):
                # The request will generate a new token plus num_spec_tokens
                # in this scheduling step.
                request.num_output_placeholders += 1 + cur_num_spec_tokens
                # Add placeholders for the new tokens in spec_token_ids.
                # We will update the actual spec token ids in the worker process.
                request.spec_token_ids = [-1] * self.num_spec_tokens

        scheduler_output.pending_structured_output_tokens = (
            pending_structured_output_tokens
        )

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        if request.discard_latest_async_tokens:
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped