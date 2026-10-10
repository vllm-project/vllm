# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import AbstractContextManager, contextmanager
from typing import Any

import torch
import torch.nn as nn

from vllm.config import (
    CompilationMode,
    VllmConfig,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.tracing import instrument
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # avoid calling accelerator APIs for methods inherited from super class
        _set_torch_accelerator_to_noop()

        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        assert device == torch.device("cpu")
        # Note: speculative decoding is now supported on CPU with C++ native impls

        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        self._postprocess_tensors()

    def _postprocess_tensors(self) -> None:
        # Note: replace device tensors with cpu tensors
        def replace_tensor(obj: Any, cpu_attr_name: str, device_attr_name) -> None:
            cpu_tensor = getattr(obj, cpu_attr_name, None)
            device_tensor = getattr(obj, device_attr_name, None)
            if isinstance(cpu_tensor, torch.Tensor) and isinstance(
                device_tensor, torch.Tensor
            ):
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
    def load_model(self, load_dummy_weights: bool = False) -> None:
        if load_dummy_weights:
            raise ValueError(
                "Loading dummy weights (needed for elastic EP scale-up) "
                "Is not supported by the CPU Model Runner."
            )
        logger.info("Starting to load model %s...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

        if hasattr(self, "drafter"):
            logger.info_once("Loading drafter model...")
            self.drafter.load_model(self.model)

        self._setup_eagle3_aux_hidden_state_outputs()

    def get_model(self) -> nn.Module:
        return self.model

    @instrument(span_name="Warmup (CPU)")
    def warming_up_model(self) -> None:
        if self.vllm_config.compilation_config.mode == CompilationMode.NONE:
            return
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        with _set_global_compilation_settings(self.vllm_config):
            self.profile_run()
        logger.info("Warming up done.")

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        super().initialize_kv_cache(
            kv_cache_config,
            is_profiling,
            kv_cache_allocation_context=kv_cache_allocation_context,
        )

        if self.speculative_config:
            if self.speculative_config.use_eagle():
                logger.info("EAGLE drafter KV cache initialized for CPU backend")
            elif self.speculative_config.uses_draft_model():
                logger.info("Draft model KV cache initialized for CPU backend")

    def _init_device_properties(self) -> None:
        pass

    def _sync_device(self) -> None:
        pass

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        # Zero full-attention blocks to prevent stale data corruption on partial writes.
        # Encoder-only (runner-only) layers are not FullAttentionSpec, so the
        # spec filter below already excludes them; no runner-only skip needed.
        seen_ptrs: set[int] = set()
        for group in self.kv_cache_config.kv_cache_groups:
            if not isinstance(group.kv_cache_spec, FullAttentionSpec):
                continue
            for layer_name in group.layer_names:
                ctx = self.compilation_config.static_forward_context.get(layer_name)
                if ctx is None:
                    continue
                kv = ctx.kv_cache
                if not isinstance(kv, torch.Tensor):
                    continue
                if kv.data_ptr() in seen_ptrs:
                    continue
                seen_ptrs.add(kv.data_ptr())
                for block_id in block_ids:
                    kv[block_id].zero_()

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
            self.wait_stream = lambda *a, **kw: None
            self.device = torch.device("cpu")

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


def _set_torch_accelerator_to_noop() -> None:
    def noop(*args: Any, **kwargs: Any) -> None:
        pass

    # Distinct no-op so empty_cache does not alias synchronize: Dynamo's
    # handle_synchronize is keyed on that object and asserts on CPU-only hosts.
    def empty_cache_noop(*args: Any, **kwargs: Any) -> None:
        pass

    torch.accelerator.synchronize = noop
    torch.accelerator.empty_cache = empty_cache_noop
