# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import time
from collections.abc import Iterable
from contextlib import AbstractContextManager
from copy import deepcopy
from typing import Any, NamedTuple

import numpy as np
import torch

import vllm.envs as envs
from vllm.compilation.wrapper import compile_model_with_stock_torch
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.reload.inplace import reload_weights
from vllm.model_executor.models.interfaces import requires_raw_input_tokens
from vllm.model_executor.offloader import (
    create_offloader,
    get_offloader,
    set_offloader,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import ECConnectorOutput, ModelRunnerOutput
from vllm.v1.watermarking import create_watermarker
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.cpu.attn_utils import (
    FastPrefillHelper,
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.cpu.block_table import CPUBlockTables
from vllm.v1.worker.cpu.ec_connector import get_ec_connector
from vllm.v1.worker.cpu.encoder_cache import EncoderCache
from vllm.v1.worker.cpu.input_batch import (
    BatchReqState,
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    post_update,
    post_update_num_computed_tokens,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.cpu.kv_connector import (
    NO_OP_KV_CONNECTOR,
    KVConnector,
    get_kv_connector,
)
from vllm.v1.worker.cpu.logprob import CPUPromptLogprobsWorker
from vllm.v1.worker.cpu.lora_utils import LoraState
from vllm.v1.worker.cpu.model_states import init_model_state
from vllm.v1.worker.cpu.pooling_runner import PoolingRunner
from vllm.v1.worker.cpu.pp_utils import PPHandler
from vllm.v1.worker.cpu.sampler import (
    CPUSampler,
    CPUWatermarkSampler,
    SamplerOutput,
)
from vllm.v1.worker.cpu.states import CPURequestState
from vllm.v1.worker.cpu.structured_outputs import CPUStructuredOutputsWorker
from vllm.v1.worker.cpu.utils import CPUKVBlockZeroer, free_before_shutdown
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.utils import (
    clear_layer_kv_caches,
    copy_kv_cache_blocks_inplace,
    get_uniform_decode_token_count,
    raise_if_nan_logits,
)

logger = init_logger(__name__)


class ExecuteModelState(NamedTuple):
    input_batch: InputBatch
    hidden_states: torch.Tensor | None
    finished_req_ids: set[str]
    ec_connector_output: ECConnectorOutput | None


def sort_batch_req_ids(
    num_tokens_per_req: dict[str, int], decode_query_len: int
) -> list[str]:
    key = lambda r: (
        (num := num_tokens_per_req[r]) != decode_query_len,
        num,
    )
    return sorted(num_tokens_per_req, key=key)


class StandaloneCPUModelRunner(LoRAModelRunnerMixin):
    """Standalone MRV2 implementation for CPUs without mature Triton support.

    It follows the GPU MRV2 architecture with a smaller feature set. Batch
    preparation, request bookkeeping, KV-cache metadata, and sampling use C++
    and PyTorch operations instead of Triton.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.jit_warmup_registry = JitWarmupRegistry(vllm_config)

        self.device = device
        self.model: torch.nn.Module
        self.dtype = self.model_config.dtype
        self.kv_cache_dtype = self.dtype
        if self.cache_config.cache_dtype != "auto":
            # Quantized KV cache.
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype
            ]

        # Lazily initialized in _init_kv_zero_meta() when the KV cache needs
        # zeroing (e.g. hybrid models with fp8 KV cache).
        self.kv_block_zeroer: CPUKVBlockZeroer | None = None

        self.vocab_size = self.model_config.get_vocab_size()
        self.max_model_len = self.model_config.max_model_len
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.is_encoder_decoder = self.model_config.is_encoder_decoder

        # Pipeline parallelism.
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.is_first_pp_rank = get_pp_group().is_first_rank
        self.is_last_pp_rank = get_pp_group().is_last_rank

        # PP broadcast/receive helper.
        self.pp_handler: PPHandler | None = None

        # Persistent buffer for intermediate tensors (non-first PP ranks).
        self.intermediate_tensors: IntermediateTensors | None = None

        # Multimodal
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.model_config.supports_multimodal_inputs
        self.uses_inputs_embeds = self.supports_mm_inputs
        self.encoder_cache: EncoderCache | None = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            self.encoder_cache = EncoderCache()
        self.ec_connector = get_ec_connector(vllm_config, self.encoder_cache)

        # General request states.
        self.req_states = CPURequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            vocab_size=self.vocab_size,
            device=self.device,
        )

        # Pooling models.
        self.is_pooling_model = self.model_config.runner_type == "pooling"
        self.pooling_runner: PoolingRunner | None = None

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        self.fast_prefill: FastPrefillHelper | None = None
        if self.use_pp:
            self.pp_handler = PPHandler(
                max_num_reqs=self.max_num_reqs,
                device=device,
            )

        # Created in load_model() after the model state is available.
        self.sampler: CPUSampler | None = None
        self.prompt_logprobs_worker: CPUPromptLogprobsWorker | None = None
        self.structured_outputs_worker: CPUStructuredOutputsWorker | None = None

        # LoRA-related workers.
        self.lora_state = LoraState(max_num_reqs=self.max_num_reqs)

        # KV Connector if configured.
        self.kv_connector: KVConnector = NO_OP_KV_CONNECTOR
        self.execute_model_state: ExecuteModelState | None = None

        set_offloader(create_offloader(self.vllm_config.offload_config))

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        self.req_states.max_model_len = max_model_len

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks: list[SupportedTask] = []
        if self.model_config.runner_type == "generate":
            tasks.extend(self.model_state.get_supported_generation_tasks())
        if self.is_pooling_model:
            tasks.extend(PoolingRunner.get_supported_tasks(self.model))
        return tuple(tasks)

    def load_model(
        self,
        load_dummy_weights: bool = False,
        *args,
        model: torch.nn.Module | None = None,
        **kwargs,
    ) -> None:
        time_before_load = time.perf_counter()
        if model is None and load_dummy_weights:
            self.load_config.load_format = "dummy"

        with DeviceMemoryProfiler() as m:
            model_loader = get_model_loader(self.vllm_config.load_config)
            logger.info_once("Loading model from scratch...")

            # Capture warmup providers selected while constructing the model.
            with self.jit_warmup_registry.activate():
                self.model = model or model_loader.load_model(
                    vllm_config=self.vllm_config,
                    model_config=self.vllm_config.model_config,
                )
            if self.lora_config:
                self.model = self.load_lora_model(
                    self.model, self.vllm_config, self.device
                )
        time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info(
            "Model loading took %s GiB memory and %.6f seconds",
            format_gib(m.consumed_memory),
            time_after_load - time_before_load,
        )

        self.model_state = init_model_state(
            self.vllm_config, self.model, self.encoder_cache, self.device
        )
        self.decode_query_len = self.model_state.num_new_sampled_tokens_per_step
        if self.decode_query_len != 1:
            raise NotImplementedError(
                "CPU Model Runner V2 without Triton supports one sampled "
                "token per decode step."
            )

        if self.is_last_pp_rank and not self.is_pooling_model:
            sampler_kwargs: dict[str, Any] = {
                "vllm_config": self.vllm_config,
                "max_num_reqs": self.max_num_reqs,
                "vocab_size": self.vocab_size,
                "device": self.device,
                "req_states": self.req_states,
                "logprobs_mode": self.model_config.logprobs_mode,
                "use_fp64_gumbel": self.model_config.use_fp64_gumbel,
                "enable_trace_replay": self.model_config.enable_trace_replay,
                "return_sampling_mask": self.model_config.return_sampling_mask,
            }
            if self.vllm_config.watermark_config is None:
                self.sampler = CPUSampler(**sampler_kwargs)
            else:
                wm_config = self.vllm_config.watermark_config
                watermarker = create_watermarker(wm_config)
                self.sampler = CPUWatermarkSampler(
                    watermarker,
                    deduplicate_contexts=wm_config.deduplicate_contexts,
                    deduplicate_contexts_max_history=(
                        wm_config.deduplicate_contexts_max_history
                    ),
                    **sampler_kwargs,
                )
            self.prompt_logprobs_worker = CPUPromptLogprobsWorker(
                self.max_num_reqs,
                self.device,
                logprobs_mode=self.model_config.logprobs_mode,
            )
            self.structured_outputs_worker = CPUStructuredOutputsWorker(
                max_num_logits=self.max_num_reqs * self.decode_query_len,
                vocab_size=self.vocab_size,
                device=self.device,
                mask_stride=self.decode_query_len,
            )

        if self.is_pooling_model and self.is_last_pp_rank:
            self.pooling_runner = PoolingRunner(self.model, self.vllm_config)

        if not self.is_first_pp_rank:
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.model_config.dtype,
                device=self.device,
            )

        get_offloader().post_init()

        if self.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
            compile_model_with_stock_torch(self.model, self.vllm_config)

    @property
    def dp_size(self) -> int:
        return self.parallel_config.data_parallel_size

    @property
    def dp_rank(self) -> int:
        return self.parallel_config.data_parallel_rank

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_draft_model(self) -> torch.nn.Module | None:
        return None

    def reload_weights(
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
    ) -> None:
        """Reload weights in place."""
        vllm_config = self.vllm_config
        model = self.get_model()
        reload_weights(
            vllm_config, model, weights_iterator, weights_path, is_checkpoint_format
        )
        self.reset_lora_state()
        self.reset_encoder_cache()
        self.reset_mm_cache()

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        encoder_runner = getattr(self.model_state, "encoder_runner", None)
        return encoder_runner.get_encoder_timing_stats() if encoder_runner else {}

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return get_kv_cache_spec(self.vllm_config)

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        block_table_max_model_len = self.max_model_len
        if self.is_encoder_decoder:
            block_table_max_model_len = max(
                block_table_max_model_len,
                self.scheduler_config.max_num_encoder_input_tokens,
                getattr(self.model_config.hf_config, "max_source_positions", 0),
            )

        block_sizes = []
        max_num_blocks_per_group = []
        slot_mapping_enabled = []
        sliding_window_reach = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            layer_spec = (
                spec.first_spec if isinstance(spec, UniformTypeKVCacheSpecs) else spec
            )
            slot_mapping_enabled.append(layer_spec.uses_slot_mapping)
            sliding_window_reach.append(
                layer_spec.num_retained_tokens
                if isinstance(layer_spec, SlidingWindowSpec)
                else 0
            )
            max_num_blocks = spec.max_num_blocks_per_req(
                self.vllm_config, block_table_max_model_len
            )
            max_num_blocks = get_block_table_width(
                max_num_blocks,
                spec.block_size,
                token_alignment=spec.block_table_token_alignment,
            )
            max_num_blocks_per_group.append(max_num_blocks)

        with self.jit_warmup_registry.activate():
            self.attn_groups, self.kernel_block_sizes = init_attn_backend(
                self.kv_cache_config,
                self.vllm_config,
                self.device,
            )

        self.block_tables = CPUBlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            slot_mapping_enabled=slot_mapping_enabled,
            sliding_window_reach=sliding_window_reach,
        )
        initialize_mamba_ssu_backend(
            self.vllm_config.mamba_config,
            self.kv_cache_config,
            use_replayssm=self.cache_config.use_replayssm,
        )
        if self.cache_config.kv_sharing_fast_prefill:
            self.fast_prefill = FastPrefillHelper()

        with self.jit_warmup_registry.activate():
            kv_caches_dict = init_kv_cache(
                self.compilation_config.static_forward_context,
                self.kv_cache_config,
                self.device,
                self.kernel_block_sizes,
                self.vllm_config,
                kv_cache_allocation_context=kv_cache_allocation_context,
                attn_groups=(group for groups in self.attn_groups for group in groups),
            )
        self.kv_caches = [
            cache for cache in kv_caches_dict.values() if cache.device == self.device
        ]
        self.model_state.initialize_kv_cache(self.kv_cache_config, self.block_tables)
        if is_profiling:
            self.kv_connector = NO_OP_KV_CONNECTOR
        else:
            self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)

    def _init_kv_zero_meta(self) -> None:
        self.kv_block_zeroer = CPUKVBlockZeroer(
            self.device,
            attn_groups_iter=(group for groups in self.attn_groups for group in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            static_forward_context=self.compilation_config.static_forward_context,
            num_blocks=self.kv_cache_config.num_blocks,
        )

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        *,
        skip_attn: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        randomize_inputs: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if skip_attn and not is_profile:
            raise ValueError(
                "skip_attn must only be True for initial memory profiling."
            )

        # Create a dummy scheduler output.
        num_reqs = min(num_tokens, self.max_num_reqs)
        if uniform_decode:
            num_tokens = max(num_tokens, self.decode_query_len)
            num_reqs = num_tokens // self.decode_query_len
            assert num_tokens % self.decode_query_len == 0

        assert num_tokens <= self.max_num_tokens, (
            f"Dummy run with {num_tokens} tokens exceeds max_num_batched_tokens "
            f"({self.max_num_tokens})"
        )

        # Distribute the remainder evenly so no dummy request exceeds
        # ceil(num_tokens / num_reqs) <= max_model_len tokens.
        num_tokens_per_request = [
            num_tokens // num_reqs + (i >= num_reqs - num_tokens % num_reqs)
            for i in range(num_reqs)
        ]

        assert sum(num_tokens_per_request) == num_tokens
        num_scheduled_tokens = {
            f"_dummy_req_{i}": n for i, n in enumerate(num_tokens_per_request)
        }
        dummy_scheduler_output = SchedulerOutput.make_empty()
        dummy_scheduler_output.total_num_scheduled_tokens = num_tokens
        dummy_scheduler_output.num_scheduled_tokens = num_scheduled_tokens

        # Disable any use of KVConnector for dummy runs.
        self.kv_connector.set_disabled(True)

        # Get the intermediate tensors for the dummy run.
        intermediate_tensors = None
        if not self.is_first_pp_rank:
            assert self.intermediate_tensors is not None
            intermediate_tensors = self.intermediate_tensors[:num_tokens]

        max_loras = self.lora_config.max_loras if self.lora_config is not None else 0
        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens=np.array(num_tokens_per_request, dtype=np.int32),
            num_sampled_tokens=None,
            remove_lora=True,
            num_active_loras=max_loras,
        ):
            # Execute the model.
            self.execute_model(
                dummy_scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=True,
                skip_attn_for_dummy_run=skip_attn,
                is_profile=is_profile,
                randomize_inputs=randomize_inputs,
            )
        self.kv_connector.set_disabled(False)

        # Non-last PP ranks don't produce output for sampling.
        if not self.is_last_pp_rank:
            return None, None

        assert self.execute_model_state is not None
        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        self.execute_model_state = None

        assert hidden_states is not None  # Last PP rank always has hidden_states
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        return hidden_states, sample_hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        num_reqs = hidden_states.shape[0]
        logits = self.model.compute_logits(hidden_states)
        input_batch = InputBatch.make_dummy(num_reqs, num_reqs, self.input_buffers)
        assert self.sampler is not None
        self.sampler(logits, input_batch)

    @torch.inference_mode()
    def _dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        assert self.pooling_runner is not None
        self.pooling_runner.dummy_pooler_run(hidden_states)

    @torch.inference_mode()
    def profile_run(self, randomize_inputs: bool = False) -> None:
        if self.supports_mm_inputs and self.is_first_pp_rank:
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and not mm_config.skip_mm_profiling:
                mm_budget = MultiModalBudget(
                    self.vllm_config, self.mm_registry, enable_cache=False
                )
                dummy_mm_inputs = mm_budget.get_dummy_encoder_profile_inputs()
                self.model_state.encoder_runner.profile_encoder_cache(
                    dummy_mm_inputs, mm_budget
                )

        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens,
            skip_attn=True,
            is_profile=True,
            randomize_inputs=randomize_inputs,
        )

        if self.is_last_pp_rank:
            assert sample_hidden_states is not None
            if self.pooling_runner is None:
                self._dummy_sampler_run(sample_hidden_states)
            else:
                assert hidden_states is not None
                self._dummy_pooler_run(hidden_states)

        del hidden_states, sample_hidden_states
        self.reset_encoder_cache()
        gc.collect()

    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        self.profile_run()
        logger.info("Warming up done.")

    def reset_mm_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_encoder_cache()
        if self.pooling_runner is not None:
            self.pooling_runner.clear()

    def release_late_interaction_query_cache(self, query_keys: list[str]) -> None:
        if self.pooling_runner is not None:
            self.pooling_runner.release_late_interaction_query_cache(query_keys)

    def _remove_request(self, req_id: str) -> bool:
        self.model_state.remove_request(req_id)
        req_idx = self.req_states.remove_request(req_id)
        if req_idx is None:
            return False
        if self.pooling_runner is not None:
            self.pooling_runner.remove_request(req_idx)
        if self.pp_handler is not None:
            self.pp_handler.on_req_idx_freed(req_idx)
        if self.encoder_cache is not None:
            self.encoder_cache.remove_request(req_id)
        if self.prompt_logprobs_worker is not None:
            self.prompt_logprobs_worker.remove_request(req_id)
        self.lora_state.remove_request(req_id)
        return True

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        finished_req_ids = scheduler_output.finished_req_ids
        if self.pooling_runner is not None:
            # Preempted docs keep their query-use reservation until rescheduled.
            self.pooling_runner.on_requests_finished(finished_req_ids)
        preempted_req_ids = scheduler_output.preempted_req_ids
        if preempted_req_ids:
            finished_req_ids = finished_req_ids.union(preempted_req_ids)
        # Sorted so every TP rank frees request slots in the same order.
        # Features like batch-sharded sampling derive rank request ownership
        # from the slot index.
        for req_id in sorted(finished_req_ids):
            self._remove_request(req_id)

    def free_states(self, scheduler_output: SchedulerOutput) -> None:
        if self.encoder_cache is not None:
            for mm_hash in scheduler_output.free_encoder_mm_hashes:
                self.encoder_cache.free_encoder_cache(mm_hash)

    def update_pp_decode_requests(self) -> None:
        if self.pp_handler is not None:
            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
                self.postprocess_sampled(**outputs)

    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.prefill_token_ids is not None
            req_id = new_req_data.req_id

            # Streaming input update: request already exists from a prior
            # chunk. Remove old state so it can be cleanly re-added below
            # with the updated prompt_token_ids and mm_features.
            self._remove_request(req_id)

            prompt_len = new_req_data.prompt_len
            sampling_params = new_req_data.sampling_params
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                max_tokens=sampling_params.max_tokens if sampling_params else 1,  # type: ignore[arg-type]
            )
            req_index = self.req_states.req_id_to_index[req_id]

            if self.pooling_runner is not None:
                assert new_req_data.pooling_params is not None
                self.pooling_runner.add_request(
                    req_id,
                    req_index,
                    new_req_data.pooling_params,
                    new_req_data.prompt_token_ids,
                )

            if self.encoder_cache is not None:
                self.encoder_cache.add_request(req_id, new_req_data.mm_features)

            self.model_state.add_request(req_index, new_req_data)
            self.block_tables.append_block_ids(
                req_index, new_req_data.block_ids, overwrite=True
            )
            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)

            if self.is_last_pp_rank and new_req_data.sampling_params is not None:
                assert self.sampler is not None
                self.sampler.add_request(req_index, new_req_data.sampling_params)
                assert self.prompt_logprobs_worker is not None
                self.prompt_logprobs_worker.add_request(
                    req_id, req_index, new_req_data.sampling_params
                )

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        # Add new blocks and update num_computed_tokens for the existing requests.
        reqs = scheduler_output.scheduled_cached_reqs
        num_computed_tokens_np = self.req_states.num_computed_tokens_np
        for req_id, num_computed_tokens, req_new_block_ids in zip(
            reqs.req_ids, reqs.num_computed_tokens, reqs.new_block_ids
        ):
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens_np[req_index] = num_computed_tokens
            if req_new_block_ids is not None:
                self.block_tables.append_block_ids(
                    req_index, req_new_block_ids, overwrite=False
                )

        # Update CPU num_computed_prefill_tokens.
        np.minimum(
            self.req_states.num_computed_tokens_np,
            self.req_states.prefill_len_np,
            out=self.req_states.num_computed_prefill_tokens,
        )

        # Zero memory for freshly allocated cache blocks to prevent stale
        # NaN/data from corrupting attention or SSM computation.
        if scheduler_output.new_block_ids_to_zero:
            assert self.kv_block_zeroer is not None
            self.kv_block_zeroer.zero_block_ids(scheduler_output.new_block_ids_to_zero)

        # Apply copy-on-write block copies for partial prefix-cache hits, after
        # zeroing new blocks and before the forward pass reads them.
        if scheduler_output.kv_cache_block_copies:
            copy_kv_cache_blocks_inplace(
                self.kv_caches,
                self.kv_cache_config.num_blocks,
                scheduler_output.kv_cache_block_copies,
            )

    def gather_batch_req_state(
        self, scheduler_output: SchedulerOutput, dummy_run: bool
    ) -> tuple[BatchReqState | None, int | None]:
        """Gather CPU request state for the scheduled batch, in batch order.
        Returns (batch_state, uniform_decode_token_count)
        """
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())

        if dummy_run:
            # Dummy batches are uniform by construction.
            return None, get_uniform_decode_token_count(
                num_reqs, num_toks, max_query_len, decode_graph_eligible=True
            )

        # batch_idx -> req_id
        req_ids = sort_batch_req_ids(num_tokens_per_req, self.decode_query_len)

        numtoks_iter = map(num_tokens_per_req.__getitem__, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

        idx_mapping_iter = map(self.req_states.req_id_to_index.__getitem__, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.intp, count=num_reqs)
        prefill_len_np = self.req_states.prefill_len_np[idx_mapping_np]
        num_computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens[
            idx_mapping_np
        ]
        is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np
        has_prefill = bool(is_prefilling_np.any())
        max_seq_len_np = None
        if self.use_pp and has_prefill:
            # consumed by the PP `compute_need_sampled_mask` for batches with prefill.
            max_seq_len_np = self.req_states.max_seq_len[idx_mapping_np]

        prefill_runs_as_decode_np = None
        decode_graph_eligible = not has_prefill
        if has_prefill:
            # One new prompt token over existing context runs as a decode.
            num_new_tokens = num_scheduled_tokens
            one_token_ext = (num_new_tokens == 1) & (num_computed_prefill_tokens_np > 0)
            prefill_runs_as_decode_np = is_prefilling_np & one_token_ext
            decode_graph_eligible = bool(
                (prefill_runs_as_decode_np | ~is_prefilling_np).all()
            )

        batch_state = BatchReqState(
            req_ids=req_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_toks,
            idx_mapping_np=idx_mapping_np,
            prefill_len_np=prefill_len_np,
            num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
            is_prefilling_np=is_prefilling_np,
            has_prefill=has_prefill,
            prefill_runs_as_decode_np=prefill_runs_as_decode_np,
            decode_graph_eligible=decode_graph_eligible,
            max_seq_len_np=max_seq_len_np,
        )
        return batch_state, get_uniform_decode_token_count(
            num_reqs, num_toks, max_query_len, decode_graph_eligible
        )

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        batch_req_state: BatchReqState,
    ) -> InputBatch:
        num_tokens = batch_req_state.num_tokens
        assert num_tokens > 0
        req_ids = batch_req_state.req_ids
        num_reqs = len(req_ids)

        is_padding = self.input_buffers.is_padding[:num_tokens]
        if envs.VLLM_MOE_SKIP_PADDING:
            is_padding.fill_(False)

        num_scheduled_tokens = batch_req_state.num_scheduled_tokens
        idx_mapping_np = batch_req_state.idx_mapping_np
        idx_mapping = torch.from_numpy(idx_mapping_np)
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.zeros(
            num_reqs, dtype=torch.int32, device=self.device
        )

        cu_num_logits = torch.arange(
            num_reqs + 1, dtype=torch.int32, device=self.device
        )
        cu_num_logits_np = cu_num_logits.numpy()

        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        query_start_loc_np = query_start_loc.numpy()
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])

        if batch_req_state.has_prefill:
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids,
                self.req_states.prefill_len,
                self.req_states.num_computed_tokens,
            )

        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs]
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len,
            self.req_states.last_sampled_tokens[:, :0],
            cu_num_logits,
            num_reqs,
            self.model_state.num_new_sampled_tokens_per_step,
        )

        fast_prefill = None
        if self.fast_prefill is not None:
            fast_prefill = self.fast_prefill.prepare(
                logits_indices,
                cu_num_logits_np,
                batch_req_state.has_prefill,
            )

        num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]
        prompt_lens = None
        if self.model_config.rswa_window is not None:
            prompt_lens = self.req_states.prompt_len[idx_mapping]

        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens,
            num_computed_tokens_np=num_computed_tokens_np,
            prefill_len_np=batch_req_state.prefill_len_np,
            num_computed_prefill_tokens_np=(
                batch_req_state.num_computed_prefill_tokens_np
            ),
            is_prefilling_np=batch_req_state.is_prefilling_np,
            has_prefill=batch_req_state.has_prefill,
            prefill_runs_as_decode_np=batch_req_state.prefill_runs_as_decode_np,
            decode_graph_eligible=batch_req_state.decode_graph_eligible,
            max_seq_len_np=batch_req_state.max_seq_len_np,
            input_ids=self.input_buffers.input_ids[:num_tokens],
            positions=self.input_buffers.positions[:num_tokens],
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=(
                scheduler_output.has_structured_output_requests
            ),
            prompt_lens=prompt_lens,
            fast_prefill=fast_prefill,
        )

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        block_tables = self.block_tables.gather_block_tables(
            input_batch.idx_mapping,
            num_reqs_padded=input_batch.num_reqs_after_padding,
            num_computed_tokens=self.req_states.num_computed_tokens,
        )
        slot_mappings = self.block_tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            num_tokens_padded=input_batch.num_tokens_after_padding,
        )
        return block_tables, slot_mappings

    def prepare_dummy_attn(
        self, input_batch: InputBatch, valid_state_slots: bool = False
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        if valid_state_slots:
            state_slots = torch.arange(
                1, input_batch.num_reqs + 1, dtype=torch.int32, device=self.device
            )
            for block_table in block_tables:
                block_table[:, 0].copy_(state_slots)
        slot_mappings = self.block_tables.get_dummy_slot_mappings(
            input_batch.num_tokens
        )
        return block_tables, slot_mappings

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        if grammar_output is not None and logits.shape[0] > 0:
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        assert self.sampler is not None
        sampler_output = self.sampler(logits, input_batch)
        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected

    def postprocess_sampled(
        self,
        idx_mapping: torch.Tensor,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
    ) -> None:
        output_bin_counts = None
        if self.is_last_pp_rank:
            assert self.sampler is not None
            output_bin_counts = self.sampler.penalties_state.output_bin_counts

        post_update(
            idx_mapping,
            self.req_states.num_computed_tokens,
            self.req_states.last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            self.req_states.all_token_ids,
            self.req_states.total_len,
        )

    def postprocess_num_computed_tokens(self, input_batch: InputBatch) -> None:
        post_update_num_computed_tokens(
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens,
            input_batch.query_start_loc,
        )

    def _merge_ec_connector_no_forward(
        self, scheduler_output: SchedulerOutput, output: ModelRunnerOutput
    ) -> ModelRunnerOutput:
        return ModelRunnerOutput.with_ec_conn_output(
            output,
            self.ec_connector.no_forward(scheduler_output).ec_connector_output,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        valid_dummy_state_slots: bool = False,
        randomize_inputs: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if not dummy_run:
            self.update_pp_decode_requests()
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            if self.encoder_cache is not None:
                # add_requests initializes RoPE positions from CPU metadata first.
                for req_id, updates in scheduler_output.encoder_input_updates.items():
                    self.encoder_cache.update_encoder_inputs(req_id, updates)

            if scheduler_output.total_num_scheduled_tokens == 0:
                empty_output = self.kv_connector.no_forward(scheduler_output)
                return self._merge_ec_connector_no_forward(
                    scheduler_output, empty_output
                )

        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_tokens = scheduler_output.total_num_scheduled_tokens
        _, num_tokens_across_dp, _ = coordinate_batch_across_dp(
            num_tokens_unpadded=num_tokens,
            num_tokens_padded=num_tokens,
            allow_microbatching=False,
            parallel_config=self.parallel_config,
            cudagraph_mode=0,
        )
        num_active_loras = 0
        if dummy_run:
            input_batch = InputBatch.make_dummy(
                num_reqs,
                num_tokens,
                self.input_buffers,
                is_padding=not is_profile,
            )
            if randomize_inputs:
                input_batch.input_ids.random_(0, self.vocab_size)
            if skip_attn_for_dummy_run:
                block_tables = None
                slot_mappings = None
            else:
                block_tables, slot_mappings = self.prepare_dummy_attn(
                    input_batch, valid_dummy_state_slots
                )
            if self.lora_config is not None:
                num_active_loras = self.lora_config.max_loras
        else:
            batch_req_state, _ = self.gather_batch_req_state(
                scheduler_output, dummy_run=False
            )
            assert batch_req_state is not None
            input_batch = self.prepare_inputs(scheduler_output, batch_req_state)
            block_tables, slot_mappings = self.prepare_attn(input_batch)
            self.model_state.preprocess_state(
                input_batch,
                block_tables,
                self.kv_cache_config,
                self.req_states.num_computed_tokens,
            )

            if self.lora_config:
                lora_inputs = self.lora_state.make_lora_inputs(
                    input_batch.req_ids,
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                    np.diff(input_batch.cu_num_logits_np),
                )
                num_active_loras = len(lora_inputs[2])
                self._set_active_loras(*lora_inputs)

        attn_metadata = None
        slot_mappings_by_layer = None
        if not (dummy_run and skip_attn_for_dummy_run):
            assert block_tables is not None and slot_mappings is not None
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            attn_metadata = self.model_state.prepare_attn(
                input_batch,
                block_tables,
                slot_mappings,
                self.attn_groups,
                self.kv_cache_config,
            )

        input_ids = input_batch.input_ids
        inputs_embeds = None
        ec_connector_output = None
        if self.uses_inputs_embeds and self.is_first_pp_rank:
            if dummy_run:
                inputs_embeds = self.model_state.dummy_inputs_embeds(
                    input_batch.num_tokens_after_padding
                )
            else:
                scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
                with self.ec_connector.maybe_get_output(
                    scheduler_output
                ) as ec_connector_output:
                    inputs_embeds = self.model_state.prepare_inputs_embeds(
                        scheduled_encoder_inputs, input_batch, self.req_states
                    )
            if inputs_embeds is not None and not requires_raw_input_tokens(self.model):
                input_ids = None

        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": None,
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        if not self.is_first_pp_rank:
            if not requires_raw_input_tokens(self.model):
                model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None

            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            n = input_batch.num_tokens_after_padding
            new_tensors = {
                k: v[:n]
                if dummy_run
                else v[:n].copy_(intermediate_tensors.tensors[k][:n])
                for k, v in self.intermediate_tensors.tensors.items()
            }
            model_inputs["intermediate_tensors"] = IntermediateTensors(new_tensors)
            del intermediate_tensors

        skip_compiled = bool(
            self.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs
        )
        batch_descriptor = BatchDescriptor(
            num_tokens=input_batch.num_tokens_after_padding,
            has_lora=self.lora_config is not None,
            num_active_loras=num_active_loras,
        )
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=input_batch.num_tokens_after_padding,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=batch_descriptor,
            slot_mapping=slot_mappings_by_layer,
            skip_compiled=skip_compiled,
            is_padding=input_batch.is_padding,
        ):
            self.kv_connector.pre_forward(scheduler_output, input_batch)
            model_output = self.model(**model_inputs)
        self.kv_connector.finish_forward()

        if self.is_last_pp_rank:
            assert isinstance(model_output, torch.Tensor)
            hidden_states = model_output
            output_intermediate_tensors = None
        else:
            assert isinstance(model_output, IntermediateTensors)
            hidden_states = None
            output_intermediate_tensors = model_output

        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            hidden_states=hidden_states,
            finished_req_ids=scheduler_output.finished_req_ids,
            ec_connector_output=ec_connector_output,
        )

        return output_intermediate_tensors

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | None:
        if self.execute_model_state is None:
            return None

        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        ec_connector_output = self.execute_model_state.ec_connector_output
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            assert self.pp_handler is not None
            self.pp_handler.receive(input_batch)
            self.postprocess_num_computed_tokens(input_batch)

            kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
            output = ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
            return ModelRunnerOutput.with_ec_conn_output(output, ec_connector_output)

        assert hidden_states is not None
        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )

        if self.pp_handler is not None:
            self.pp_handler.broadcast(
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch,
            )

        assert self.prompt_logprobs_worker is not None
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            hidden_states,
            input_batch,
            self.req_states.all_token_ids,
            self.req_states.num_computed_tokens,
            self.req_states.prompt_len_np,
        )
        prompt_token_id_logprobs_dict = (
            self.prompt_logprobs_worker.compute_prompt_token_id_logprobs(
                self.model.compute_logits,
                hidden_states,
                input_batch,
                self.req_states.prompt_len_np,
            )
        )

        sampled_token_ids = sampler_output.sampled_token_ids.tolist()
        num_sampled_list = num_sampled.tolist()
        for token_ids, num_tokens in zip(
            sampled_token_ids, num_sampled_list, strict=True
        ):
            del token_ids[num_tokens:]

        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=sampled_token_ids,
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
            prompt_token_id_logprobs_dict=prompt_token_id_logprobs_dict,
        )
        if sampler_output.logprobs_tensors is not None:
            model_runner_output.logprobs = sampler_output.logprobs_tensors.tolists()
        if sampler_output.sampling_mask_tensors is not None:
            model_runner_output.sampling_masks = (
                sampler_output.sampling_mask_tensors.tolists(num_sampled.numpy())
            )
        if sampler_output.num_nans is not None:
            model_runner_output.num_nans_in_logits = dict(
                zip(
                    input_batch.req_ids,
                    sampler_output.num_nans.tolist(),
                    strict=True,
                )
            )

        model_runner_output.kv_connector_output = self.kv_connector.post_forward(
            finished_req_ids
        )

        self.postprocess_sampled(
            input_batch.idx_mapping,
            sampler_output.sampled_token_ids,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
        )

        model_runner_output.ec_connector_output = ec_connector_output
        if (
            envs.VLLM_RAISE_ON_LOGIT_NANS
            and model_runner_output.num_nans_in_logits is not None
        ):
            raise_if_nan_logits(model_runner_output.num_nans_in_logits)
        return model_runner_output

    @torch.inference_mode()
    def pool(self) -> ModelRunnerOutput | None:
        if self.execute_model_state is None:
            return None

        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        ec_connector_output = self.execute_model_state.ec_connector_output
        self.execute_model_state = None

        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
        if not self.is_last_pp_rank:
            self.postprocess_num_computed_tokens(input_batch)
            output = ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
            return ModelRunnerOutput.with_ec_conn_output(output, ec_connector_output)

        assert hidden_states is not None
        assert self.pooling_runner is not None
        pooler_output, finished_mask = self.pooling_runner.pool(
            hidden_states, input_batch, self.req_states
        )

        outputs = (
            pooler_output.unbind()
            if isinstance(pooler_output, torch.Tensor)
            else pooler_output
        )
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            pooler_output=[
                None if output is None or not is_finished else output
                for output, is_finished in zip(outputs, finished_mask, strict=True)
            ],
            kv_connector_output=kv_connector_output,
            ec_connector_output=ec_connector_output,
        )
        self.postprocess_num_computed_tokens(input_batch)
        return model_runner_output

    def shutdown(self) -> None:
        """Release CPU tensors (model weights, KV caches, workspace) so that
        memory is reclaimable when running in the same process."""
        set_offloader(None)
        self.fast_prefill = None
        if self.pooling_runner is not None:
            self.pooling_runner.clear()
        self.pooling_runner = None
        self.kv_block_zeroer = None
        if hasattr(self, "kv_caches"):
            self.kv_caches.clear()
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            del self.kv_cache_config
        if self.encoder_cache is not None:
            self.encoder_cache.reset_encoder_cache()
        free_before_shutdown(self.vllm_config)
        if hasattr(self, "model_state"):
            del self.model_state
        # Detach the layer-level KV/state cache tensors before dropping the
        # model; the model object can outlive this runner.
        if hasattr(self, "model"):
            clear_layer_kv_caches(self.model.modules())
            del self.model

        gc.collect()
        logger.debug("Cleaned up model weights, KV caches, and workspace")
