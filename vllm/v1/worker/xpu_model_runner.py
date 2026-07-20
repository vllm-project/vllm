# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from functools import partial

import torch

from vllm.config import VllmConfig
from vllm.utils.torch_utils import supports_xpu_graph
from vllm.v1.worker.gpu.model_runner import (
    GPUModelRunner as GPUModelRunnerV2,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class XPUModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        # FIXME: To be verified.
        self.cascade_attn_enabled = False


class XPUModelRunnerV2(GPUModelRunnerV2):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)


@contextmanager
def _torch_cuda_wrapper():
    # Replace cuda APIs with xpu APIs. Each callable gets its own functools.partial
    # so it is not the same object as torch.xpu.* (Torch Dynamo _get_handlers()
    # asserts on duplicate registration when cuda aliases xpu directly).
    torch.cuda.Stream = torch.xpu.Stream
    torch.cuda.default_stream = partial(torch.xpu.current_stream)
    torch.cuda.current_stream = partial(torch.xpu.current_stream)
    torch.cuda.stream = partial(torch.xpu.stream)
    torch.cuda.set_stream = partial(torch.xpu.set_stream)

    # torch.xpu.Event does not accept the ``blocking`` kwarg that
    # torch.cuda.Event supports, so drop it here.
    def _xpu_event(*args, blocking=None, **kwargs):
        return torch.xpu.Event(*args, **kwargs)

    torch.cuda.Event = _xpu_event
    if supports_xpu_graph():
        torch.cuda.graph = partial(torch.xpu.graph)
        torch.cuda.CUDAGraph = torch.xpu.XPUGraph
        torch.cuda.graph_pool_handle = partial(torch.xpu.graph_pool_handle)
    yield
