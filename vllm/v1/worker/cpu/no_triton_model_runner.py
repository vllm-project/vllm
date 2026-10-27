# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import torch

from vllm.compilation.wrapper import compile_model_with_stock_torch
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
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
from vllm.v1.watermarking import create_watermarker
from vllm.v1.worker.cpu.attn_utils import FastPrefillHelper
from vllm.v1.worker.cpu.ec_connector import get_ec_connector
from vllm.v1.worker.cpu.encoder_cache import EncoderCache
from vllm.v1.worker.cpu.input_batch import InputBuffers
from vllm.v1.worker.cpu.kv_connector import NO_OP_KV_CONNECTOR, KVConnector
from vllm.v1.worker.cpu.logprob import CPUPromptLogprobsWorker
from vllm.v1.worker.cpu.lora_utils import LoraState
from vllm.v1.worker.cpu.model_states import init_model_state
from vllm.v1.worker.cpu.pooling_runner import PoolingRunner
from vllm.v1.worker.cpu.pp_utils import PPHandler
from vllm.v1.worker.cpu.sampler import CPUSampler, CPUWatermarkSampler
from vllm.v1.worker.cpu.states import CPURequestState
from vllm.v1.worker.cpu.structured_outputs import CPUStructuredOutputsWorker
from vllm.v1.worker.cpu.utils import CPUKVBlockZeroer
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

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
