# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MPS Model Runner for Apple Silicon GPUs."""
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


@contextmanager
def _torch_cuda_wrapper():
    """Wrapper to prevent CUDA-specific operations during init."""
    # Patch torch.cuda.Stream and torch.cuda.Event to prevent instantiation
    original_stream = torch.cuda.Stream
    original_event = torch.cuda.Event

    class DummyStream:
        def __init__(self, *args, **kwargs):
            pass
        def wait_stream(self, *args, **kwargs):
            pass
        def record_event(self, *args, **kwargs):
            pass
        def wait_event(self, *args, **kwargs):
            pass
        def synchronize(self):
            pass

    class DummyEvent:
        def __init__(self, *args, **kwargs):
            pass
        def record(self, *args, **kwargs):
            pass
        def wait(self, *args, **kwargs):
            pass
        def query(self):
            return True
        def synchronize(self):
            pass

    try:
        torch.cuda.Stream = DummyStream
        torch.cuda.Event = DummyEvent
        yield
    finally:
        torch.cuda.Stream = original_stream
        torch.cuda.Event = original_event


class MPSModelRunner(GPUModelRunner):
    """Model runner for MPS (Metal Performance Shaders) backend."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Use wrapper to prevent CUDA-specific operations
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        assert device.type == "mps", f"Expected MPS device, got {device}"
        assert self.speculative_config is None, "Speculative decoding is not supported on MPS."

        # Disable CUDA-specific features
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # Replace the CUDA stream with a dummy object
        self.comm_stream = None

    def _init_device_properties(self) -> None:
        """Initialize device properties for MPS."""
        # MPS doesn't have the same device properties as CUDA
        # Create a minimal device properties object
        class MPSDeviceProperties:
            name = "Apple Silicon GPU"
            major = 1
            minor = 0
            multi_processor_count = 1

        self.device_properties = MPSDeviceProperties()

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        if len(self.kv_cache_config.kv_cache_groups) > 1:
            raise ValueError(
                "Multiple KVCacheGroups is not currently supported with MPS model runner."
            )
        super()._may_reorder_batch(scheduler_output)

    def load_model(self, eep_scale_up: bool = False) -> None:
        logger.info("Starting to load model %s on MPS...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

    def get_model(self) -> nn.Module:
        return self.model

    def warming_up_model(self) -> None:
        logger.info("Warming up model on MPS for compilation...")
        # Only generate graph for the generic shape
        from vllm.v1.worker.cpu_model_runner import _set_global_compilation_settings
        with _set_global_compilation_settings(self.vllm_config):
            self._dummy_run(self.max_num_tokens, None, None, None, None)
