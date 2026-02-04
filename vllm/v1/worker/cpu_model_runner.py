# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.tracing import instrument
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        assert device == torch.device("cpu")
        # Note: speculative decoding is now supported on CPU with PyTorch fallbacks

        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        self._postprocess_tensors()

    def _postprocess_tensors(self) -> None:
        # Note: replace device tensors with cpu tensors
        def replace_tensor(obj: Any, cpu_attr_name: str, device_attr_name) -> None:
            cpu_tensor = getattr(obj, cpu_attr_name, None)
            device_tensor = getattr(obj, device_attr_name, None)
            if cpu_tensor is not None and device_tensor is not None:
                assert isinstance(cpu_tensor, torch.Tensor)
                assert isinstance(device_tensor, torch.Tensor)
                setattr(obj, device_attr_name, cpu_tensor)

        for v in vars(self).values():
            if isinstance(v, CpuGpuBuffer):
                v.gpu = v.cpu

        for k, v in vars(self.input_batch).items():
            if k.endswith("_cpu_tensor") and isinstance(v, torch.Tensor):
                replace_tensor(self.input_batch, k, k[:-11])

        for block_table in self.input_batch.block_table.block_tables:
            for v in vars(block_table).values():
                if isinstance(v, CpuGpuBuffer):
                    v.gpu = v.cpu

    @instrument(span_name="Loading (CPU)")
    def load_model(self, eep_scale_up: bool = False) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

        if hasattr(self, "drafter"):
            logger.info_once("Loading drafter model...")
            self.drafter.load_model(self.model)

    def get_model(self) -> nn.Module:
        return self.model

    @instrument(span_name="Warmup (CPU)")
    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        with _set_global_compilation_settings(self.vllm_config):
            self._dummy_run(
                min(
                    max(16, self.max_num_reqs),
                    self.scheduler_config.max_num_batched_tokens,
                )
            )

        # Warm up drafter for speculative decoding
        if self.speculative_config and (self.speculative_config.uses_draft_model()):
            from vllm.v1.spec_decode.draft_model import DraftModelProposer

            if isinstance(self.drafter, (DraftModelProposer)):
                logger.info("Warming up drafter model...")
                self.drafter.dummy_run(max(16, self.max_num_reqs))

        logger.info("Warming up done.")

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache for CPU backend, including drafter cache if needed.

        Args:
            kv_cache_config: Configuration for the KV cache
        """
        # Call parent implementation
        super().initialize_kv_cache(kv_cache_config)

        # Log CPU-specific confirmation for speculative decoding
        if self.speculative_config:
            if self.speculative_config.use_eagle():
                logger.info("EAGLE drafter KV cache initialized for CPU backend")
            elif self.speculative_config.uses_draft_model():
                logger.info("Draft model KV cache initialized for CPU backend")

    def _init_device_properties(self) -> None:
        pass

    def _sync_device(self) -> None:
        pass

    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        # Note: For CPU backend, dp padding is not required for now.
        return 0, None

    # =========================================================================
    # CPU-safe overrides for speculative decoding methods
    # These methods override GPU-specific implementations that use CUDA streams
    # =========================================================================

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False
    ) -> None:
        """CPU-safe version: no async copy needed, tensors already on CPU."""
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        draft_token_ids: torch.Tensor = self._draft_token_ids
        if not torch.is_tensor(draft_token_ids):
            return

        num_reqs = draft_token_ids.shape[0]
        if self.draft_token_ids_cpu is not None:
            if not zeros_only:
                self.draft_token_ids_cpu[:num_reqs].copy_(draft_token_ids)
            else:
                self.draft_token_ids_cpu[:num_reqs] = 0

    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:
        """CPU-safe version: no event synchronization needed."""
        if isinstance(self._draft_token_ids, list):
            return self._draft_token_ids, self.input_batch.req_ids
        req_ids = self._draft_token_req_ids
        if req_ids is None:
            return [], []
        if self.draft_token_ids_cpu is not None:
            return self.draft_token_ids_cpu[: len(req_ids)].tolist(), req_ids
        return [], []

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        """CPU-safe version: direct copy without CUDA streams."""
        if self.valid_sampled_token_count_cpu is None:
            return

        counts = valid_sampled_tokens_count
        counts_cpu = self.valid_sampled_token_count_cpu
        counts_cpu[: counts.shape[0]].copy_(counts)
        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def _get_valid_sampled_token_count(self) -> list[int]:
        """CPU-safe version: no event synchronization needed."""
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        if prev_sampled_token_ids is None:
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        if counts_cpu is None:
            return []
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        """CPU-safe version: direct tolist() without CUDA events."""
        return sampled_token_ids.tolist()


@contextmanager
def _torch_cuda_wrapper():
    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            pass

    cuda_event = torch.Event
    cuda_stream = torch.cuda.Stream
    try:
        torch.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        yield
    finally:
        torch.Event = cuda_event
        torch.cuda.Stream = cuda_stream


@contextmanager
def _set_global_compilation_settings(config: VllmConfig):
    import torch._inductor.config as torch_inductor_config

    inductor_config = config.compilation_config.inductor_compile_config
    # Note: The MKLDNN and CPPGEMM backend requires freezing parameters.
    freezing_value = torch_inductor_config.freezing
    try:
        if inductor_config.get("max_autotune", False):
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value
