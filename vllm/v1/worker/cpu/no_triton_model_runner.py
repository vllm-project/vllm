# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Iterable
from contextlib import AbstractContextManager
from copy import deepcopy

import torch

from vllm.compilation.wrapper import compile_model_with_stock_torch
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.reload.inplace import reload_weights
from vllm.model_executor.offloader import (
    create_offloader,
    get_offloader,
    set_offloader,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.watermarking import create_watermarker
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.cpu.attn_utils import (
    FastPrefillHelper,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.cpu.block_table import CPUBlockTables
from vllm.v1.worker.cpu.ec_connector import get_ec_connector
from vllm.v1.worker.cpu.encoder_cache import EncoderCache
from vllm.v1.worker.cpu.input_batch import InputBuffers
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
from vllm.v1.worker.cpu.sampler import CPUSampler, CPUWatermarkSampler
from vllm.v1.worker.cpu.states import CPURequestState
from vllm.v1.worker.cpu.structured_outputs import CPUStructuredOutputsWorker
from vllm.v1.worker.cpu.utils import CPUKVBlockZeroer, free_before_shutdown
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.utils import clear_layer_kv_caches

logger = init_logger(__name__)


class NoTritonCPUModelRunner(LoRAModelRunnerMixin):
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
