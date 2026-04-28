# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler subclass with profiling-based dynamic chunk sizing.

When enabled via additional_config, this scheduler profiles prefill latency
during initialization and uses a quadratic model to predict optimal chunk
sizes at runtime, equalizing execution time across pipeline stages.

Usage::

    vllm serve <model> --additional-config \
        '{"profiling_chunk_config": {"enabled": true}}'
"""

import inspect
import time

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import (
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.profiling_chunk_predictor import ProfilingChunkManager
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


def _get_profiling_chunk_config(vllm_config: VllmConfig) -> dict:
    """Extract profiling_chunk_config from additional_config."""
    additional_config = vllm_config.additional_config
    if additional_config is None:
        return {}
    if isinstance(additional_config, dict):
        return additional_config.get("profiling_chunk_config", {})
    return {}


class ProfilingChunkScheduler(Scheduler):
    """Scheduler with profiling-based dynamic chunk sizing.

    During initialization, the scheduler profiles prefill latency at various
    chunk sizes by calling ``profile_prefill_latency`` on each worker via
    ``collective_rpc``.  A quadratic latency model is then fitted, and during
    scheduling the model predicts the optimal chunk size for each waiting
    request based on its ``num_computed_tokens``.
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
        **kwargs,
    ) -> None:
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
            **kwargs,
        )

        profiling_cfg = _get_profiling_chunk_config(vllm_config)
        base_chunk = self.max_num_scheduled_tokens

        smooth_factor = float(profiling_cfg.get("smooth_factor", 0.8))
        min_chunk = int(profiling_cfg.get("min_chunk", 4096))

        self.profiling_chunk_manager = ProfilingChunkManager(
            base_chunk_size=base_chunk,
            page_size=self.cache_config.block_size,
            smooth_factor=smooth_factor,
            min_chunk=min_chunk,
        )
        self._profiling_initialized = False

        logger.info(
            "[ProfilingChunk] Scheduler initialized. base_chunk=%d, "
            "page_size=%d, smooth_factor=%.2f, min_chunk=%d",
            base_chunk,
            self.cache_config.block_size,
            smooth_factor,
            min_chunk,
        )

    # ------------------------------------------------------------------
    # Profiling initialization
    # ------------------------------------------------------------------

    def run_profiling_chunk_init(self, model_executor) -> None:
        """Profile prefill latency using real model forward passes.

        Called by EngineCore after model_executor is ready.  Collects latency
        samples at different chunk sizes and fits the quadratic model.
        """
        if self._profiling_initialized:
            return
        self._profiling_initialized = True

        if model_executor is None:
            logger.warning(
                "[ProfilingChunk] No model_executor provided, "
                "skipping profiling"
            )
            return

        logger.info(
            "[ProfilingChunk] Running startup profiling "
            "with real model forward..."
        )

        seq_lens: list[int] = []
        latencies: list[float] = []

        base_chunk_size = self.profiling_chunk_manager.base_chunk_size
        num_samples = 64

        rpc_kwargs = self._build_rpc_kwargs(model_executor)

        total_steps = num_samples + 1
        log_interval = max(1, total_steps // 10)
        t_start = time.perf_counter()

        for i in range(total_steps):
            chunk_size = int(
                base_chunk_size - (i - 1) * (base_chunk_size / num_samples)
            )
            if chunk_size <= 0:
                break

            if i % log_interval == 0 or i == total_steps - 1:
                elapsed = time.perf_counter() - t_start
                logger.info(
                    "[ProfilingChunk] Profiling prefill latency: "
                    "%d/%d samples done (chunk=%d, elapsed=%.1fs)",
                    max(i - 1, 0),
                    num_samples,
                    chunk_size,
                    elapsed,
                )

            try:
                result = model_executor.collective_rpc(
                    "profile_prefill_latency",
                    args=(chunk_size,),
                    **rpc_kwargs,
                )

                # First iteration is warm-up
                if i == 0:
                    continue

                latency_ms = self._extract_latency(result)
                if latency_ms is None:
                    continue

                seq_lens.append(chunk_size)
                latencies.append(latency_ms)

            except Exception as e:
                logger.debug(
                    "[ProfilingChunk] Forward failed for chunk=%d: %s",
                    chunk_size,
                    e,
                )
                continue

        if len(seq_lens) < 8:
            logger.warning(
                "[ProfilingChunk] Profiling failed: only %d samples collected",
                len(seq_lens),
            )
            return

        logger.info(
            "[ProfilingChunk] Collected %d samples. "
            "Latency range: [%.2f, %.2f] ms",
            len(seq_lens),
            min(latencies),
            max(latencies),
        )

        predictor = self.profiling_chunk_manager.predictor
        if not predictor.fit(seq_lens, latencies):
            return

        predictor.set_target_latency(base_chunk_size)
        predictor.is_ready = True
        self.profiling_chunk_manager._profiling_done = True

        logger.info("[ProfilingChunk] Profiling completed successfully")

    @staticmethod
    def _build_rpc_kwargs(model_executor) -> dict:
        """Build kwargs for collective_rpc, handling PP unique_reply_rank."""
        kwargs: dict = {}
        if not hasattr(model_executor, "collective_rpc"):
            return kwargs

        sig = inspect.signature(model_executor.collective_rpc)
        if "unique_reply_rank" not in sig.parameters:
            return kwargs

        try:
            pc = model_executor.vllm_config.parallel_config
            output_rank = (
                pc.world_size
                - pc.tensor_parallel_size
                * pc.prefill_context_parallel_size
            )
            kwargs["unique_reply_rank"] = output_rank
        except AttributeError:
            pass

        return kwargs

    @staticmethod
    def _extract_latency(result) -> float | None:
        """Extract latency value from collective_rpc result."""
        if isinstance(result, (int, float)):
            return float(result)
        if isinstance(result, list) and len(result) > 0:
            return float(result[0])
        return None

    # ------------------------------------------------------------------
    # schedule() override
    # ------------------------------------------------------------------
    # The method below is based on the upstream Scheduler.schedule()
    # with profiling-based chunk sizing applied to both RUNNING requests
    # (chunked prefill continuation) and WAITING requests (new prefill).
    # Modified sections are marked with ">>> PROFILING CHUNK" comments.
    # ------------------------------------------------------------------

    def schedule(self) -> SchedulerOutput:  # noqa: C901
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        # >>> PROFILING CHUNK >>>
        # NOTE: time_budget feature is temporarily disabled due to FIA operator
        # performance issues with multiple request groups. It will be enabled
        # after the issues are resolved.
        # time_budget = self.profiling_chunk_manager.predictor.target_latency
        time_budget = 0.01
        # <<< PROFILING CHUNK <<<
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # First, schedule the RUNNING requests.
        req_index = 0
        # >>> PROFILING CHUNK >>>
        while (
            req_index < len(self.running)
            and token_budget > 0
            and time_budget > 0
        ):
            # <<< PROFILING CHUNK <<<
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens
                + 2
                - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if (
                0
                < self.scheduler_config.long_prefill_token_threshold
                < num_new_tokens
            ):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold
                )
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens,
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

            # >>> PROFILING CHUNK: dynamic chunk sizing for RUNNING >>>
            if (
                self.profiling_chunk_manager is not None
                and self.profiling_chunk_manager.is_ready
                and num_new_tokens > 1
                and request.num_computed_tokens > 0
            ):
                predicted_chunk = (
                    self.profiling_chunk_manager.predict_chunk_size(
                        num_computed_tokens=request.num_computed_tokens,
                        target_time=time_budget,
                    )
                )
                if predicted_chunk is not None and predicted_chunk > 0:
                    num_new_tokens = min(predicted_chunk, num_new_tokens)
            # <<< PROFILING CHUNK <<<

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
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
                        break

                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(
                                preempted_req_id
                            )
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(
                                preempted_req_id, None
                            )
                            preempted_encoder_inputs = (
                                scheduled_encoder_inputs.pop(
                                    preempted_req_id, None
                                )
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(
                        preempted_req, scheduled_timestamp
                    )
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        break

            if new_blocks is None:
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            time_budget -= self.profiling_chunk_manager.predict_time(
                num_new_tokens, request.num_computed_tokens
            )
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
                        spec_token_ids = spec_token_ids[
                            :num_scheduled_spec_tokens
                        ]
                    scheduled_spec_decode_tokens[request_id] = spec_token_ids

                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = (
                    encoder_inputs_to_schedule
                )
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(
                            request, i
                        )
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(
                            request, i
                        )

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
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)

            # >>> PROFILING CHUNK >>>
            while (
                (self.waiting or self.skipped_waiting)
                and token_budget > 0
                and time_budget > 0
            ):
                # <<< PROFILING CHUNK <<<
                if len(self.running) == self.max_num_running_reqs:
                    break

                request_queue = (
                    self._select_waiting_queue_for_scheduling()
                )
                assert request_queue is not None

                request = request_queue.peek_request()
                request_id = request.request_id

                # Try to promote blocked statuses while traversing
                # skipped queue.
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(
                    request
                ):
                    if (
                        request.status
                        == RequestStatus.WAITING_FOR_REMOTE_KVS
                    ):
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Check that adding the request still respects the
                # max_loras constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras)
                        == self.lora_config.max_loras
                        and request.lora_request.lora_int_id
                        not in scheduled_loras
                    )
                ):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries = 0
                connector_prefix_cache_hits = 0

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request,
                                num_new_local_computed_tokens,
                            )
                        )

                        if ext_tokens is None:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens
                            - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = (
                            num_external_computed_tokens
                        )

                    num_computed_tokens = (
                        num_new_local_computed_tokens
                        + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.empty_kv_cache_blocks
                    )
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    num_new_tokens = (
                        request.num_tokens - num_computed_tokens
                    )
                    threshold = (
                        self.scheduler_config.long_prefill_token_threshold
                    )
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # >>> PROFILING CHUNK: dynamic chunk sizing >>>
                    if (
                        self.profiling_chunk_manager is not None
                        and self.profiling_chunk_manager.is_ready
                        and num_new_tokens > 1
                        and request.num_computed_tokens > 0
                    ):
                        predicted_chunk = (
                            self.profiling_chunk_manager.predict_chunk_size(
                                num_computed_tokens=num_computed_tokens,
                                target_time=time_budget,
                            )
                        )
                        if (
                            predicted_chunk is not None
                            and predicted_chunk > 0
                        ):
                            num_new_tokens = min(
                                num_new_tokens, predicted_chunk
                            )
                    # <<< PROFILING CHUNK <<<

                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
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
                            shift_computed_tokens=(
                                1 if self.use_eagle else 0
                            ),
                        )
                        if num_new_tokens == 0:
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

                effective_lookahead_tokens = (
                    0
                    if request.num_computed_tokens == 0
                    else self.num_lookahead_tokens
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

                if (
                    self.scheduler_reserve_full_isl
                    and not self.kv_cache_manager.can_fit_full_sequence(
                        request,
                        num_new_computed_tokens=num_new_local_computed_tokens,
                        new_computed_blocks=new_computed_blocks,
                        num_external_computed_tokens=(
                            num_external_computed_tokens
                        ),
                        num_encoder_tokens=num_encoder_tokens,
                    )
                ):
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=(
                        num_external_computed_tokens
                    ),
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

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

                request = request_queue.pop_request()
                if load_kv_async:
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    request.num_computed_tokens = num_computed_tokens
                    continue

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED,
                        scheduled_timestamp,
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}"
                    )

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(
                        request.lora_request.lora_int_id
                    )
                req_to_new_blocks[request_id] = (
                    self.kv_cache_manager.get_blocks(request_id)
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                time_budget -= self.profiling_chunk_manager.predict_time(
                    num_new_tokens, request.num_computed_tokens
                )
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = (
                        encoder_inputs_to_schedule
                    )
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(
                                request, i
                            )
                    encoder_compute_budget = new_encoder_compute_budget
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(
                                request, i
                            )

            # Re-queue requests skipped in this pass ahead of older
            # skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert (
            len(scheduled_new_reqs)
            + len(scheduled_resumed_reqs)
            + len(scheduled_running_reqs)
            <= len(self.running)
        )

        # Get the longest common prefix among all requests in the
        # running queue.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups
        )
        with record_function_or_nullcontext(
            "schedule: get_num_common_prefix_blocks"
        ):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(
                        any_request_id
                    )
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs = (
                scheduled_new_reqs + scheduled_resumed_reqs
            )
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
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext(
            "schedule: make_cached_request_data"
        ):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(
            num_scheduled_tokens.keys()
        )

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
            preempted_req_ids={
                req.request_id for req in preempted_reqs
            },
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=(
                self.encoder_cache_manager.get_freed_mm_hashes()
            ),
            new_block_ids_to_zero=new_block_ids_to_zero,
        )

        if self.connector is not None:
            meta = self._build_kv_connector_meta(
                self.connector, scheduler_output
            )
            scheduler_output.kv_connector_metadata = meta

        if self.ec_connector is not None:
            ec_meta = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext(
            "schedule: update_after_schedule"
        ):
            self._update_after_schedule(scheduler_output)
        return scheduler_output
