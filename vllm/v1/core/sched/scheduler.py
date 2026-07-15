# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np

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
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsReader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, ForkInfo, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
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
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # Fork-related configuration
        # Read from HuggingFace config (config.json), not vLLM's ModelConfig
        hf_config = getattr(vllm_config.model_config, 'hf_config', None)
        if hf_config is not None:
            self.fork_token_id: int | None = getattr(hf_config, 'fork_token_id', None)
            self.child_token_id: int | None = getattr(hf_config, 'child_token_id', None)
        else:
            self.fork_token_id = None
            self.child_token_id = None
        

        self.enable_dynamic_fork = (self.fork_token_id is not None and 
                                     self.child_token_id is not None)
        # self.enable_dynamic_fork = False
        # Fork debug switch - controlled by environment variable VLLM_FORK_DEBUG
        import os
        self.fork_debug = os.environ.get("VLLM_FORK_DEBUG", "0").lower() in ("1", "true", "yes")
        
        if self.enable_dynamic_fork:
            logger.info(f"Dynamic fork enabled: fork_token_id={self.fork_token_id}, "
                       f"child_token_id={self.child_token_id}, debug={self.fork_debug}")
            # Only load tokenizer when debug is enabled (performance optimization)
            if self.fork_debug:
                from transformers import AutoTokenizer
                model_path = vllm_config.model_config.model
                self.fork_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                logger.info(f"Loaded tokenizer from {model_path} for fork debugging")
            else:
                self.fork_tokenizer = None
        else:
            self.fork_tokenizer = None
        
        # Track parent-child relationships for result aggregation
        self.fork_relationships: dict[str, list[str]] = {}  # parent_id -> [child_ids]
        self.pending_fork_requests: list[Request] = []  # Requests to add after current step
        
        # Fork output aggregation
        # Stores accumulated token IDs for each fork branch
        self.fork_outputs: dict[str, dict[str, list[int]]] = {}  # parent_id -> {req_id: [token_ids]}
        # Tracks which branches have finished
        self.fork_finished: dict[str, set[str]] = {}  # parent_id -> set of finished req_ids
        # Stores the original parent request's client_index for returning aggregated results
        self.fork_client_index: dict[str, int] = {}  # parent_id -> client_index
        
        # Track fork child->parent mapping for Mamba state copy
        # This is populated in _process_pending_fork_requests and cleared after each schedule step
        self.fork_child_to_parent_this_step: dict[str, str] = {}  # child_id -> parent_id
        
        # Dirty flag for incremental waiting queue sorting:
        # Only re-sort when fork events actually change the queue composition.
        self._fork_queue_dirty: bool = False

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens
            else self.scheduler_config.max_num_batched_tokens
        )
        print(self.max_num_scheduled_tokens)
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        if self.vllm_config.kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = (
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        self.mm_budget = mm_budget = (
            MultiModalBudget(vllm_config, mm_registry)
            if self.supports_mm_inputs
            else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            hash_block_size=self.block_size,
            metrics_collector=self.kv_metrics_collector,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        if self.vllm_config.model_config.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_reader = RoutedExpertsReader.create()

            assert len(kv_cache_config.kv_cache_groups) > 0, (
                "enable_return_routed_experts requires at least one kv cache group"
            )
            self.max_num_kv_tokens = (
                kv_cache_config.num_blocks // len(kv_cache_config.kv_cache_groups) + 1
            ) * self.block_size

            self.routed_experts_reader.attach_buffer(
                max_num_kv_tokens=self.max_num_kv_tokens,
                vllm_config=self.vllm_config,
            )

        self._pause_state: PauseState = PauseState.UNPAUSED

        # Fork-aware scheduling configuration
        # KV cache usage threshold above which parent requests are not admitted
        import os
        self.fork_parent_kv_cache_threshold = float(
            os.environ.get("VLLM_FORK_PARENT_KV_CACHE_THRESHOLD", "0.2"))
        # Allow up to N fork-children to be admitted to running beyond
        # max_num_running_reqs. This guarantees that a freshly-created child
        # can be co-batched with its parent in the very next forward pass,
        # preserving APAR's weight-load amortization benefit (Option A).
        # Children share most of the parent's KV via fork_kv_cache so they
        # only consume <=1 partial block + spec slots of new KV.
        self.fork_child_running_overflow = int(
            os.environ.get("VLLM_FORK_CHILD_RUNNING_OVERFLOW", "0"))
        if self.enable_dynamic_fork:
            logger.info(
                "Fork-aware scheduling enabled: "
                "parent KV cache admission threshold=%.2f, "
                "child running overflow slots=%d",
                self.fork_parent_kv_cache_threshold,
                self.fork_child_running_overflow)

    def _get_fork_waiting_priority(
        self,
        request: Request,
        family_waiting_count: dict[str, int],
        family_running_count: dict[str, int],
    ) -> tuple:
        """Compute a sort key for fork-aware waiting queue ordering.

        Priority rules (FCFS-by-family):
        - Child requests always have higher priority than parent requests.
        - Among children: families whose parent arrived earlier get higher
          priority (FCFS by family). Within the same family, more siblings
          already running/finished = higher priority.
        - Among parents/non-fork: earlier arrival = higher priority (FCFS).

        Returns a tuple for sorting (lower = higher priority):
            (type_order, ..., arrival_time, request_id)
        """
        if request.is_fork_child:
            # Child request: type_order=0 (higher priority than parents)
            # Primary sub-key: parent's arrival time (earlier parent = higher priority)
            # Secondary sub-key: prefer families closer to completion
            parent_id = request.parent_request_id
            total_children = len(self.fork_relationships.get(parent_id, []))
            waiting = family_waiting_count.get(parent_id, 0)
            progress = total_children - waiting  # running + finished siblings

            parent_req = self.requests.get(parent_id)
            parent_arrival = parent_req.arrival_time if parent_req else request.arrival_time

            return (0, parent_arrival, -progress, request.arrival_time, request.request_id)
        else:
            # Parent request (or non-fork request): type_order=1 (lower priority)
            # Pure FCFS: earlier arrival = higher priority
            return (1, request.arrival_time, request.request_id)

    def _sort_waiting_queue_for_fork(self) -> None:
        """Re-sort the waiting queue using fork-aware priority rules.

        Called at the beginning of schedule() when dynamic fork is enabled.
        Uses a dirty flag to skip re-sorting when no fork events have occurred
        since the last sort, reducing CPU overhead from O(n log n) to O(1) for
        the common case where the queue composition hasn't changed.

        The dirty flag is set by:
        - _process_pending_fork_requests: new child requests added to waiting
        - _collect_fork_output: a branch finished, changing family statistics
        """
        if not self.waiting or not self._fork_queue_dirty:
            return

        # Pre-compute family counts in O(n) instead of O(n²)
        family_waiting_count: dict[str, int] = defaultdict(int)
        for r in self.waiting:
            if r.is_fork_child:
                family_waiting_count[r.parent_request_id] += 1

        family_running_count: dict[str, int] = defaultdict(int)
        for r in self.running:
            if r.is_fork_child:
                family_running_count[r.parent_request_id] += 1

        # Extract all requests, sort by fork-aware priority, rebuild queue
        all_requests = list(self.waiting)
        # Clear the existing queue
        if isinstance(self.waiting, FCFSRequestQueue):
            self.waiting.clear()
        else:
            # For PriorityRequestQueue
            self.waiting._heap.clear()

        all_requests.sort(
            key=lambda r: self._get_fork_waiting_priority(
                r, family_waiting_count, family_running_count))
        for req in all_requests:
            if isinstance(self.waiting, FCFSRequestQueue):
                self.waiting.append(req)
            else:
                self.waiting.add_request(req)

        self._fork_queue_dirty = False

    def _select_preempt_request_for_fork(self) -> Request:
        """Select a request to preempt from the running queue using fork-aware rules.

        Preemption rules (FCFS-by-family, consistent with scheduling priority):
        - Always prefer to preempt parent requests over child requests.
        - Among parents: later arrival = preempted first (FCFS protection).
        - Among non-fork requests: later arrival = preempted first.
        - Among children (only if no parents/others): later parent arrival =
          preempted first; ties broken by default priority/arrival_time.

        Returns the request to preempt.
        """
        # Separate parents and children in running queue
        parent_requests = []
        child_requests = []
        other_requests = []

        for req in self.running:
            if req.is_fork_child:
                child_requests.append(req)
            elif req.fork_count > 0:
                # This is a parent that has produced children
                parent_requests.append(req)
            else:
                other_requests.append(req)

        if parent_requests:
            # Preempt the parent that arrived latest (FCFS: earlier = protected)
            return max(parent_requests, key=lambda r: r.arrival_time)
        elif other_requests:
            # Non-fork requests: preempt latest arrival
            return max(other_requests, key=lambda r: r.arrival_time)
        else:
            # Only children remain: preempt children from later-arriving families first
            def _child_preempt_key(r):
                parent_req = self.requests.get(r.parent_request_id)
                parent_arrival = parent_req.arrival_time if parent_req else r.arrival_time
                return (-parent_arrival, r.priority, r.arrival_time)
            return max(child_requests, key=_child_preempt_key)

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        assert num_external_computed_tokens == 0, (
            "External KV connector is not verified yet"
        )
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Perform block-aligned splitting at prefill phase, including:
        # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0
        # * resumed requests: num_computed_tokens < (
        #                       num_prompt_tokens + num_output_tokens
        #                     )
        # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # To enable block-aligned caching of the Mamba state, `num_new_tokens`
            # must be a multiple of `block_size`.
            # As an exception, if `num_new_tokens` is less than `block_size`, the
            # state is simply not cached, requiring no special handling.
            # Additionally, when Eagle mode is enabled, FullAttn prunes the last
            # matching block. To prevent this from causing a Mamba cache miss, the
            # last chunk must be not smaller than `block_size`.
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            # eagle prune
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # align to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # force to cache the last chunk
                num_new_tokens = last_cache_position - num_computed_tokens
            else:
                # prefill the last few tokens
                pass
        return num_new_tokens

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
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # Fork-aware: re-sort waiting queue before scheduling
        if self.enable_dynamic_fork:
            self._sort_waiting_queue_for_fork()

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
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

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
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break
                    logger.info("preempt triggered: req=%s, kv_usage=%.3f",
                        request.request_id, self.kv_cache_manager.usage)
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    # P0: preserve already-scheduled draft (spec) tokens of the
                    # preempted request so the MTP investment is not wasted.
                    # We re-attach them to `spec_token_ids` AFTER
                    # `_preempt_request` (which clears that field), so the
                    # next schedule step can verify them again.
                    preempted_spec_token_ids: list[int] | None = None
                    if self.enable_dynamic_fork:
                        # Fork-aware preemption: prefer parents over children
                        preempted_req = self._select_preempt_request_for_fork()
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            preempted_spec_token_ids = (
                                scheduled_spec_decode_tokens.pop(
                                    preempted_req_id, None
                                )
                            )
                            logger.info("popped spec_tokens for %s: %s",
                                preempted_req_id, preempted_spec_token_ids)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    elif self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            preempted_spec_token_ids = (
                                scheduled_spec_decode_tokens.pop(
                                    preempted_req_id, None
                                )
                            )
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    # Restore the preserved draft tokens AFTER _preempt_request
                    # (which clears `spec_token_ids` internally). They remain
                    # valid as next-token predictions because preemption keeps
                    # the request's output tokens; only KV is recomputed.
                    '''if preempted_spec_token_ids:
                        print("preempted_spec_token_ids: ", preempted_spec_token_ids)
                        preempted_req.spec_token_ids = list(
                            preempted_spec_token_ids
                        )'''
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests.
        # Fork-aware (Option A): even if preemption happened during the running
        # loop, still allow fork-children to be admitted so that parent and
        # child stay co-batched in the same forward pass. Non-child requests
        # remain gated by the original `not preempted_reqs` rule.
        fork_only_admission = (
            self.enable_dynamic_fork
            and bool(preempted_reqs)
        )
        if (
            (not preempted_reqs or fork_only_admission)
            and self._pause_state == PauseState.UNPAUSED
        ):
            # Use a temporary RequestQueue to collect requests that need to be
            # skipped and put back at the head of the waiting queue later
            skipped_waiting_requests = create_request_queue(self.policy)

            # Fork-aware effective running cap: children may overflow up to
            # `fork_child_running_overflow` slots above max_num_running_reqs.
            # Non-children must still respect max_num_running_reqs.
            effective_max_running = self.max_num_running_reqs
            if self.enable_dynamic_fork:
                effective_max_running += self.fork_child_running_overflow

            while self.waiting and token_budget > 0:
                if len(self.running) >= effective_max_running:
                    break

                request = self.waiting.peek_request()
                request_id = request.request_id

                # Fork-only admission mode: skip non-child requests (they will
                # be retried next step when preemption pressure has cleared).
                if fork_only_admission and not request.is_fork_child:
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                # Respect the original max_num_running_reqs cap for non-children
                # even when overflow is enabled. Only children may use the
                # overflow slots.
                if (
                    not request.is_fork_child
                    and len(self.running) >= self.max_num_running_reqs
                ):
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                # Fork-aware: block parent requests when KV cache usage is high
                if (self.enable_dynamic_fork
                        and not request.is_fork_child
                        and self.kv_cache_manager.usage
                        > self.fork_parent_kv_cache_threshold):
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        if request.num_preemptions:
                            # We must be loading for a resumed preemption
                            # rather than a new request.
                            request.status = RequestStatus.PREEMPTED
                        else:
                            request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
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

                # Streaming: skip request if still waiting for next streaming req.
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    assert not request.streaming_queue
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

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                else:
                    # num_computed_tokens > 0 for two cases:
                    # 1. KVTransfer: async KV recvs completed.
                    # 2. Partial preemption: prompt blocks still allocated.
                    # In both cases, blocks are already in req_to_blocks,
                    # so we only need to allocate the remaining tokens.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

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
                    if num_new_tokens <= 0:
                        # 必须保留 ≥1 个 token 重算，否则没有 logits 给采样
                        block_size = self.block_size
                        # 把 cached 长度对齐到上一个完整块边界，确保严格 < num_tokens
                        capped = ((request.num_tokens - 1) // block_size) * block_size
                        # 同步退还多出来的整块（关键！）
                        excess_blocks = (num_new_local_computed_tokens - capped) // block_size
                        if excess_blocks > 0:
                            # 具体 API 看你的 KVCacheBlocks 实现，常见的是切片或 .pop_last()
                            new_computed_blocks = new_computed_blocks[:-excess_blocks] \
                                if hasattr(new_computed_blocks, '__getitem__') \
                                else new_computed_blocks.truncate(len(new_computed_blocks) - excess_blocks)
                        num_new_local_computed_tokens = capped
                        num_computed_tokens = capped
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

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

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

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
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
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            # Put back any skipped requests at the head of the waiting queue
            if skipped_waiting_requests:
                self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        # Fork-aware: children may overflow up to fork_child_running_overflow
        # beyond max_num_running_reqs (Option A co-batch guarantee).
        effective_running_cap = self.max_num_running_reqs
        if self.enable_dynamic_fork:
            effective_running_cap += self.fork_child_running_overflow
        assert len(self.running) <= effective_running_cap
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
            # Fork information for Mamba state copy
            fork_child_to_parent=self.fork_child_to_parent_this_step.copy() if self.fork_child_to_parent_this_step else None,
        )
        
        # Clear fork mapping after building scheduler output
        self.fork_child_to_parent_this_step.clear()

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta: KVConnectorMetadata = self.connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.

        When dynamic fork is enabled, uses partial preemption: keeps the
        prompt's KV cache blocks allocated and only frees output blocks.
        This avoids expensive prompt re-computation on reschedule.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )

        # Ensure that if this request was scheduled in the previous step,
        # we no longer treat it as such. Otherwise, when the same step both
        # preempts and resumes the request (e.g., fork-aware admission), the
        # resumed-path assertion in `_make_cached_request_data` will fail.
        self.prev_step_scheduled_req_ids.discard(request.request_id)
        
        # Partial preemption keeps the prompt-aligned KV cache prefix (and,
        # for multimodal requests, also keeps the encoder cache reference so
        # the GPU-side encoder outputs are not evicted), advancing
        # num_computed_tokens to that boundary. This avoids expensive prompt
        # re-computation on resume — including image encoder re-runs.
        #
        # Why partial preempt is now safe for multimodal + fork-child:
        #   - The kept KV prefix [0, tokens_kept) is a strict subset of
        #     [0, request.num_computed_tokens) at preempt time, whose KV is
        #     correctly populated.
        #   - For fork-child: the resume path goes through
        #     waiting.prepend_request and does NOT touch
        #     _process_pending_fork_requests (only fed by FORK-token emission
        #     into pending_fork_requests), so no double fork_kv_cache call /
        #     refcount corruption occurs on resume.
        #   - For multimodal: by NOT calling encoder_cache_manager.free here,
        #     the request keeps its reference on cached encoder outputs
        #     (mm_hash stays in `cached`, not in `freeable`), so they cannot
        #     be evicted by other requests' `can_allocate` while preempted.
        #     On resume, `check_and_update_cache` returns True for those
        #     mm_hashes, encoder is not rescheduled, and
        #     `_gather_mm_embeddings` finds the cached encoder outputs —
        #     avoiding the previously fatal `Encoder cache miss` assertion.
        #
        # Cap the keep target at num_computed_tokens to handle the rare case
        # of preemption mid-prefill, where the request's allocated blocks may
        # not yet cover the full num_prompt_tokens.
        use_partial_preempt = self.enable_dynamic_fork

        if use_partial_preempt:
            # Partial preemption: keep prompt KV blocks (or as many as are
            # actually computed) and keep encoder cache references alive.
            block_size = self.block_size
            tokens_to_keep = min(
                request.num_prompt_tokens,
                request.num_computed_tokens,
            )
            # Block-align down so this scheduler-side computation matches
            # what kv_cache_manager.free_partial will actually keep.
            tokens_to_keep = (tokens_to_keep // block_size) * block_size

            # Critical: tokens_to_keep MUST NOT split a multimodal item.
            # If [image_start, image_start+image_len) straddles
            # tokens_to_keep (image_start < tokens_to_keep < image_end),
            # then on resume the scheduler decides the image is not yet in
            # KV (line ~1469), goes to check_and_update_cache, finds it in
            # `cached` (since we don't call free here) → skips rescheduling
            # encoding → but the worker's `encoder_cache` may have already
            # evicted that tensor (it was made freeable when the request
            # passed the image earlier, and another request's can_allocate
            # could have evicted it before this preempt). The worker would
            # then hit `assert encoder_output is not None` (Encoder cache
            # miss). Pulling tokens_to_keep down to image_start makes the
            # image "not started" on resume, which forces a clean re-encode
            # path through can_allocate/allocate.
            if request.has_encoder_inputs and request.mm_features:
                while tokens_to_keep > 0:
                    straddler_start: int | None = None
                    for mm_feature in request.mm_features:
                        s = mm_feature.mm_position.offset
                        e = s + mm_feature.mm_position.length
                        if s < tokens_to_keep < e:
                            if straddler_start is None or s < straddler_start:
                                straddler_start = s
                    if straddler_start is None:
                        break
                    tokens_to_keep = (straddler_start // block_size) * block_size

            if tokens_to_keep <= 0:
                # Nothing safe to keep — fall back to full preempt.
                self.kv_cache_manager.free(request)
                request.num_computed_tokens = 0
                self.encoder_cache_manager.free(request)
            else:
                tokens_kept = self.kv_cache_manager.free_partial(
                    request, tokens_to_keep)
                request.num_computed_tokens = tokens_kept
                # NOTE: deliberately do NOT call
                # self.encoder_cache_manager.free here — see docstring
                # above.
        else:
            # Full preemption (original behavior, used when dynamic fork is
            # disabled). Frees both KV cache and encoder cache references.
            self.kv_cache_manager.free(request)
            request.num_computed_tokens = 0
            self.encoder_cache_manager.free(request)

        request.status = RequestStatus.PREEMPTED
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)

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
            request.num_computed_tokens += num_scheduled_token
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
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
        self.finished_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids
            if idx >= num_running_reqs:
                assert not scheduled_in_prev_step
                resumed_req_ids.add(req_id)
            if not scheduled_in_prev_step:
                all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if (
                start_pos
                >= num_computed_tokens + num_new_tokens + shift_computed_tokens
            ):
                # The encoder input is not needed in this step.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

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
        cudagraph_stats = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

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
                # skip failed or rescheduled requests from KV load failure
                continue
            request = self.requests.get(req_id)
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
            # Also check for FORK tokens and create child requests.
            forked = False
            if new_token_ids:
                new_token_ids, stopped, forked = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            routed_experts = None
            finish_reason = None
            if stopped:
                # Check if this request is part of a fork family
                parent_id = self._get_fork_parent_id(req_id)
                if parent_id is not None:
                    # This is a fork-related request
                    # Let it stop normally, but track for aggregation
                    if self.fork_debug:
                        logger.info(f"[FORK] Branch {req_id} stopped, waiting for aggregation")
                
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
            
            # Check if this request is part of a fork family
            parent_id = self._get_fork_parent_id(req_id)
            is_fork_request = parent_id is not None
            
            if (
                new_token_ids
                or pooler_output is not None
                or kv_transfer_params
                or stopped
            ):
                if is_fork_request:
                    # This is a fork-related request, collect output for aggregation
                    self._collect_fork_output(parent_id, req_id, new_token_ids, stopped)
                    
                    # Check if all branches are finished
                    if self._all_fork_branches_finished(parent_id):
                        # Aggregate and return the final output
                        aggregated_output = self._aggregate_fork_outputs(parent_id)
                        client_idx = self.fork_client_index[parent_id]
                        outputs[client_idx].append(
                            EngineCoreOutput(
                                request_id=parent_id,  # Use parent's request_id
                                new_token_ids=aggregated_output,
                                finish_reason=finish_reason,
                                new_logprobs=None,  # Logprobs not supported for aggregated output
                                new_prompt_logprobs_tensors=None,
                                pooling_output=pooler_output,
                                stop_reason=request.stop_reason,
                                events=request.take_events(),
                                kv_transfer_params=kv_transfer_params,
                                trace_headers=request.trace_headers,
                                num_cached_tokens=request.num_cached_tokens,
                            ))
                        # Clean up fork tracking data
                        self._cleanup_fork_data(parent_id)
                else:
                    # Normal request, add output directly
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

        # Process any pending fork requests created during this step
        self._process_pending_fork_requests()

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

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self.waiting.add_request(request)
        return False

    def _get_routed_experts(self, request: Request) -> np.ndarray | None:
        if not self.vllm_config.model_config.enable_return_routed_experts:
            return None

        kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
        block_ids = kv_blocks.get_block_ids()[0]
        num_tokens = request.num_tokens - 1

        # compute slot mapping
        block_ids_array = np.array(block_ids, dtype=np.int32)
        num_blocks = len(block_ids)
        block_size = self.block_size

        # generate block offsets
        block_offsets = np.arange(0, block_size)

        # compute slot mapping: slot = block_id * block_size + offset
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_array.reshape((num_blocks, 1)) * block_size
        ).flatten()[:num_tokens]

        return self.routed_experts_reader.get_routed_experts(indices=slot_mapping)

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool, bool]:
        """
        Update request with new output tokens.
        
        Returns:
            tuple: (new_token_ids, stopped, forked)
            - new_token_ids: The token IDs to include in output
            - stopped: Whether the request should stop
            - forked: Whether a fork was detected and child request created
        """

        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        forked = False
        # Convert to list if needed so we can modify it
        new_token_ids = list(new_token_ids)
        # Track which indices had FORK tokens (for output replacement later)
        fork_indices = []
        
        for idx, output_token_id in enumerate(new_token_ids):
            # Check for FORK token before appending
            is_fork_token = (self.enable_dynamic_fork and 
                             output_token_id == self.fork_token_id)

            # Prevent child branches from forking again (check if "_fork_" in request_id)
            # If a child request tries to fork again, end its prediction early.
            if is_fork_token and "_fork_" in request.request_id:
                logger.info(f"[FORK] Child request {request.request_id} cannot fork again, "
                           f"ending prediction early at token index {idx}")
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True
                # Drop this fork token and anything after it from the output
                new_token_ids = new_token_ids[:idx+1]
                fork_indices = [i for i in fork_indices if i < idx]
                break
            
            if is_fork_token:
                # Debug logging for fork (only decode when debug=True)
                if self.fork_debug:
                    output_tokens = list(request.output_token_ids) if request.output_token_ids else []
                    output_text = ""
                    if self.fork_tokenizer is not None and output_tokens:
                        try:
                            output_text = self.fork_tokenizer.decode(output_tokens, skip_special_tokens=False)
                        except Exception as e:
                            output_text = f"[decode error: {e}]"
                    is_child = getattr(request, 'is_fork_child', False)
                    logger.info(f"[FORK] request_id={request.request_id}, "
                           f"is_fork_child={is_child}, "
                           f"output_token_ids={output_tokens}, "
                           f"output_text={output_text}")
                # time.sleep(10)
                # Create a forked child request
                # Note: Don't pass block_hasher=None - let it inherit from parent
                # The parent's _block_hasher will be used for computing block_hashes
                forked = True
                child_request = request.fork_request(
                    child_token_id=self.child_token_id,
                    # block_hasher will be inherited from parent's _block_hasher
                )
                #child_request.sampling_params.max_tokens = 8000
                #child_request.max_tokens = 8000
                
                # Debug logging for child (only decode when debug=True)
                if self.fork_debug:
                    child_prompt_text = ""
                    if self.fork_tokenizer is not None:
                        try:
                            child_prompt_text = self.fork_tokenizer.decode(
                                child_request.prompt_token_ids, skip_special_tokens=False
                            )
                        except Exception as e:
                            child_prompt_text = f"[decode error: {e}]"
                    logger.info(f"[FORK] Child created: request_id={child_request.request_id}, "
                           f"num_prompt_tokens={child_request.num_prompt_tokens}, "
                           f"prompt_token_ids[-20:]={child_request.prompt_token_ids[-20:] if len(child_request.prompt_token_ids) > 20 else child_request.prompt_token_ids}, "
                           f"prompt_text_suffix={child_prompt_text[-200:] if len(child_prompt_text) > 200 else child_prompt_text}")
              
                # Queue the child request to be added after current step
                self.pending_fork_requests.append(child_request)
                # Track parent-child relationship
                if request.request_id not in self.fork_relationships:
                    self.fork_relationships[request.request_id] = []
                    # Initialize fork output aggregation for this parent
                    self.fork_outputs[request.request_id] = {request.request_id: []}
                    self.fork_finished[request.request_id] = set()
                    self.fork_client_index[request.request_id] = request.client_index
                self.fork_relationships[request.request_id].append(
                    child_request.request_id)
                # Initialize output storage for child
                self.fork_outputs[request.request_id][child_request.request_id] = []
                # IMPORTANT: Keep FORK in parent's output_token_ids to maintain model context
                # Only replace in the returned new_token_ids for display purposes
                fork_indices.append(idx)
            
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            # IMPORTANT: Skip stop check for FORK tokens to ensure parent continues
            # generating more FORK tokens for other branches
            if is_fork_token:
                # For FORK tokens, only check length limits, not stop tokens
                if (request.num_tokens >= self.max_model_len or
                    request.num_output_tokens >= request.max_tokens):
                    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                    stopped = True
                    new_token_ids = new_token_ids[:idx+1]
                    fork_indices = [i for i in fork_indices if i <= idx]
                    break
            else:
                stopped = check_stop(request, self.max_model_len)
                if stopped:
                    new_token_ids = new_token_ids[:idx+1]
                    # Update fork_indices to match trimmed list
                    fork_indices = [i for i in fork_indices if i <= idx]
                    break
        
        # Replace FORK with CHILD in the returned tokens (for display only)
        for idx in fork_indices:
            new_token_ids[idx] = self.child_token_id
        
        return new_token_ids, stopped, forked

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue
           
            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            self.waiting.add_request(request)
            self.requests[request.request_id] = request
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            Tuple of (req_id, client_index) for requests that were aborted. Will not
            include any that were already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

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
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return [(r.request_id, r.client_index) for r in valid_requests]

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()

        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):

        assert request.is_finished()
        request_id = request.request_id
        
        # If this is a fork child request, notify parent
        if "_fork_" in request_id:
            parent_id = request_id.rsplit("_fork_", 1)[0]
            if parent_id in self.fork_relationships:
                child_list = self.fork_relationships[parent_id]
                if request_id in child_list:
                    child_list.remove(request_id)
                    if self.fork_debug:
                        logger.info(f"[Fork] Removed child {request_id} from parent's fork_relationships")
        
        self.kv_cache_manager.free(request)
        del self.requests[request_id]

    # =========================================================================
    # Fork-related helper methods
    # =========================================================================
    
    def _get_fork_parent_id(self, req_id: str) -> str | None:
        """Get the root parent request ID if this request is part of a fork family.
        
        For nested forks like chatcmpl-xxx_fork_1_fork_1, this returns chatcmpl-xxx
        (the original parent), not chatcmpl-xxx_fork_1.
        """
        # Check if this is a parent request itself
        if req_id in self.fork_outputs:
            return req_id
        
        # Check if this is a child request (format: parent_id_fork_N or nested)
        if "_fork_" not in req_id:
            return None
        
        # For nested forks, we need to find the root parent
        # chatcmpl-xxx_fork_1_fork_1 -> chatcmpl-xxx (split on first _fork_)
        root_parent_id = req_id.split("_fork_", 1)[0]
        if root_parent_id in self.fork_outputs:
            return root_parent_id
        
        # Also check immediate parent (for compatibility with existing nested fork tracking)
        immediate_parent_id = req_id.rsplit("_fork_", 1)[0]
        if immediate_parent_id in self.fork_outputs:
            return immediate_parent_id
        
        return None

    def _collect_fork_output(self, parent_id: str, req_id: str, 
                             new_token_ids: list[int], stopped: bool) -> None:
        """Collect output tokens from a fork branch."""
        if parent_id not in self.fork_outputs:
            return

        # Ensure this request has an output list
        if req_id not in self.fork_outputs[parent_id]:
            self.fork_outputs[parent_id][req_id] = []
        
        # Append new tokens
        self.fork_outputs[parent_id][req_id].extend(new_token_ids)
        
        # Mark as finished if stopped
        if stopped:
            self.fork_finished[parent_id].add(req_id)
            # Family statistics changed — remaining siblings' priority may shift
            self._fork_queue_dirty = True
            if self.fork_debug:
                logger.info(f"[FORK] Branch {req_id} finished, "
                       f"total tokens: {len(self.fork_outputs[parent_id][req_id])}")

    def _all_fork_branches_finished(self, parent_id: str) -> bool:
        """Check if all fork branches (parent + children) have finished."""
        if parent_id not in self.fork_outputs:
            return False
        
        # Get all branch IDs (parent + children)
        all_branch_ids = set(self.fork_outputs[parent_id].keys())
        finished_ids = self.fork_finished.get(parent_id, set())
        
        all_finished = all_branch_ids == finished_ids
        if all_finished:
            logger.info(f"[FORK] All {len(all_branch_ids)} branches finished for parent {parent_id}")
        
        return all_finished

    def _aggregate_fork_outputs(self, parent_id: str) -> list[int]:
        """Aggregate outputs from all fork branches into a single output.
        
        The aggregation logic:
        1. Iterate through parent's tokens
        2. When encountering a CHILD token, insert the corresponding child branch's output after it
        3. Child branches are matched by their fork number (_fork_1, _fork_2, ...)
        
        Result format: <BLOCK>...<CHILD>[child_1_content]<BLOCK>...<CHILD>[child_2_content]...
        """

        if parent_id not in self.fork_outputs:
            return []
        
        branch_outputs = self.fork_outputs[parent_id]
        parent_tokens = branch_outputs.get(parent_id, [])
        
        # Get child branches sorted by fork number
        # Format: parent_id_fork_1, parent_id_fork_2, ...
        child_branches = [(bid, branch_outputs[bid]) 
                          for bid in branch_outputs.keys() 
                          if bid != parent_id]
        # Sort by fork number (extract the number after "_fork_")
        child_branches.sort(key=lambda x: int(x[0].rsplit("_fork_", 1)[1]))
        
        logger.info(f"[FORK] Aggregating: parent has {len(parent_tokens)} tokens, "
                   f"{len(child_branches)} child branches")
        
        # Interleave parent tokens with child outputs
        aggregated: list[int] = []
        child_idx = 0
        
        for token in parent_tokens:
            aggregated.append(token)
            # When we see a CHILD token, insert the corresponding child's output after it
            if token == self.child_token_id and child_idx < len(child_branches):
                child_id, child_tokens = child_branches[child_idx]
                aggregated.extend(child_tokens)
                if self.fork_debug:
                    logger.info(f"[FORK] Inserted child {child_idx + 1} ({child_id}): "
                           f"{len(child_tokens)} tokens after CHILD token")
                child_idx += 1
        
        # Log warning if not all children were inserted
        if child_idx < len(child_branches):
            if self.fork_debug:
                logger.warning(f"[FORK] Not all children inserted! "
                          f"Expected {len(child_branches)}, inserted {child_idx}")
        if self.fork_debug:
            logger.info(f"[FORK] Total aggregated output: {len(aggregated)} tokens")
        return aggregated

    def _cleanup_fork_data(self, parent_id: str) -> None:
        """Clean up fork tracking data after aggregation.
        
        Also handles nested fork cleanup - if parent_id itself contains '_fork_',
        it means this is a nested fork family that also needs cleanup.
        """

        # Clean up the direct fork data
        if parent_id in self.fork_outputs:
            del self.fork_outputs[parent_id]
        if parent_id in self.fork_finished:
            del self.fork_finished[parent_id]
        if parent_id in self.fork_client_index:
            del self.fork_client_index[parent_id]
        if parent_id in self.fork_relationships:
            del self.fork_relationships[parent_id]
        
        # Also clean up any nested fork data that may have been created
        # by child requests that also generated FORK tokens
        nested_ids_to_clean = []
        for fork_parent in list(self.fork_outputs.keys()):
            if fork_parent.startswith(parent_id + "_fork_"):
                nested_ids_to_clean.append(fork_parent)
        
        for nested_id in nested_ids_to_clean:
            if nested_id in self.fork_outputs:
                del self.fork_outputs[nested_id]
            if nested_id in self.fork_finished:
                del self.fork_finished[nested_id]
            if nested_id in self.fork_client_index:
                del self.fork_client_index[nested_id]
            if nested_id in self.fork_relationships:
                del self.fork_relationships[nested_id]
            if self.fork_debug:
                logger.info(f"[FORK] Cleaned up nested fork data for {nested_id}")
        if self.fork_debug:
            logger.info(f"[FORK] Cleaned up fork data for parent {parent_id}")

    def _process_pending_fork_requests(self) -> None:
        """Add pending fork requests to the scheduler."""
        if not self.pending_fork_requests:
            return

        if self.fork_debug:
            logger.info(f"[FORK] Processing {len(self.pending_fork_requests)} pending fork requests")

        # Get block_size for calculating num_computed_tokens
        # IMPORTANT: Use the actual KV cache block size from kv_cache_config, NOT self.block_size
        # self.block_size is max_model_len (e.g., 16000), but the actual KV cache block size
        # is from kv_cache_spec (e.g., 544). Using the wrong block_size causes:
        # - num_full_blocks = fork_position // 16000 = 0 (no blocks shared)
        # - Child branches don't inherit any KV cache, causing garbled output
        kv_cache_groups = self.kv_cache_config.kv_cache_groups
        if kv_cache_groups:
            block_size = kv_cache_groups[0].kv_cache_spec.block_size
        else:
            block_size = self.block_size

        for child_request in self.pending_fork_requests:
            # DEBUG: Print detailed info
            if self.fork_debug:
                print(f"[FORK PROCESS DEBUG] child_request.request_id={child_request.request_id}")
                print(f"[FORK PROCESS DEBUG] child_request.num_prompt_tokens={child_request.num_prompt_tokens}")
                print(f"[FORK PROCESS DEBUG] child_request.num_output_tokens={child_request.num_output_tokens}")
                print(f"[FORK PROCESS DEBUG] child_request.num_tokens={child_request.num_tokens}")
                print(f"[FORK PROCESS DEBUG] len(prompt_token_ids)={len(child_request.prompt_token_ids) if child_request.prompt_token_ids else 0}")
                print(f"[FORK PROCESS DEBUG] block_size={block_size}")
                
            # In method B: child inherits parent's output_token_ids
            # fork_position = child's total tokens = prompt + inherited output (including CHILD token)
            # This is the total sequence length that needs KV cache
            fork_position = child_request.num_tokens  # = num_prompt_tokens + num_output_tokens

            # Initialize num_shared_tokens to 0 (will be updated based on actual shared blocks)
            num_shared_tokens = 0

            if child_request.parent_request_id:
                # Try to fork KV cache (share parent's FULL blocks only)
                try:
                    shared_blocks = self.kv_cache_manager.fork_kv_cache(
                        parent_request_id=child_request.parent_request_id,
                        child_request_id=child_request.request_id,
                        fork_position=fork_position - 1,  # Position before CHILD token
                        block_size=block_size,  # Pass scheduler's block_size
                    )
                    # IMPORTANT: For hybrid models (Attention + Mamba), we need to find
                    # the first non-empty block group. blocks[0] might be MambaManager's
                    # empty list, while blocks[1] is AttentionManager's shared blocks.
                    num_shared_blocks = 0
                    if shared_blocks.blocks:
                        for block_group in shared_blocks.blocks:
                            if block_group:  # Find first non-empty block group
                                num_shared_blocks = len(block_group)
                                break
                    # IMPORTANT: Calculate num_shared_tokens based on ACTUAL shared blocks
                    # Use the block_size from kv_cache_manager, not scheduler
                    if num_shared_blocks > 0:
                        num_shared_tokens = num_shared_blocks * block_size
                    else:
                        num_shared_tokens = 0  # No blocks shared, recompute everything
                    if self.fork_debug:
                        logger.info(f"[FORK] Shared {num_shared_blocks} full blocks from parent "
                               f"({num_shared_tokens} tokens are shared)")
                except Exception as e:
                    logger.warning(f"Failed to fork KV cache: {e}")
                    num_shared_tokens = 0  # Fallback: recompute everything

            # Set num_computed_tokens = number of tokens covered by shared FULL blocks
            # Child will recompute tokens from num_shared_tokens to fork_position
            # This includes the partial block tokens AND the CHILD token
            #child_request.num_computed_tokens = num_shared_tokens
            child_request.num_computed_tokens = min(num_shared_tokens, child_request.num_tokens - 1)

            if self.fork_debug:
                logger.info(f"[FORK] Child {child_request.request_id}: "
                       f"num_prompt_tokens={child_request.num_prompt_tokens}, "
                       f"num_output_tokens={child_request.num_output_tokens}, "
                       f"fork_position={fork_position}, "
                       f"num_shared_tokens={num_shared_tokens}, "
                       f"num_computed_tokens={child_request.num_computed_tokens}, "
                       f"tokens_to_recompute={fork_position - num_shared_tokens}")

            # Add the child request to the scheduler
            self.add_request(child_request)
            if self.fork_debug:
                logger.info(f"[FORK] Added child request {child_request.request_id} to scheduler")
            
            # Track fork child->parent mapping for Mamba state copy
            # This will be used by GPU model runner to copy Mamba state from parent
            if child_request.parent_request_id:
                self.fork_child_to_parent_this_step[child_request.request_id] = child_request.parent_request_id

        # Mark waiting queue dirty so next schedule() re-sorts with new children
        self._fork_queue_dirty = True
        # Clear the pending list
        self.pending_fork_requests.clear()

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = len(self.waiting) - self.num_waiting_for_streaming_input
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp)
                # NOTE(zhuohan): For async scheduling, we need to discard the latest
                # output token on the fly to avoid a redundant repetitive output token.
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            logger.warning("reset_connector called but no KV connector is configured.")
            return False

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            encoder_cache_usage=self._get_encoder_cache_usage(),
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def _get_encoder_cache_usage(self) -> float:
        """Get encoder cache usage as a fraction (0.0 to 1.0)."""
        ecm = self.encoder_cache_manager
        if ecm.cache_size == 0:
            return 0.0
        used_slots = ecm.cache_size - ecm.num_free_slots
        return used_slots / ecm.cache_size

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

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

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

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
            (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
            num_computed_tokens = len(block_ids) * self.block_size
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
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # Async loading. If num_computed_tokens is set it implies we
                # already processed some block failures for it in a prior step
                req_num_computed_tokens = (
                    request.num_computed_tokens
                    if req_id in self.failed_recving_kv_req_ids
                    else len(req_block_ids) * self.block_size
                )
            else:
                # Sync loading. num_computed_tokens includes new tokens
                req_num_computed_tokens = request.num_cached_tokens

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens
                request.num_external_computed_tokens -= num_affected_tokens
                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - request.num_cached_tokens
                    )
                    request.num_computed_tokens = request.num_cached_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs, invalid_block_ids, evict_blocks=False
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
