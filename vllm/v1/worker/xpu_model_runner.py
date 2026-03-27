# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import supports_xpu_graph
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

# When set, all model computation runs on CPU to avoid SYCL kernel dispatch
# on simulators where AubLoad cannot execute compute kernels (L3 bank = 0).
_SIM_CPU_FALLBACK = os.environ.get("VLLM_SIM_CPU_FALLBACK", "0") == "1"


class XPUModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self._sim_cpu_fallback = _SIM_CPU_FALLBACK
        if self._sim_cpu_fallback:
            # Redirect all tensor / model allocation to CPU so that
            # no SYCL compute-kernels are ever dispatched to the
            # simulator.  The XPU device is still reported for driver /
            # environment validation (41 setup tests), but actual
            # inference runs on the host CPU.
            logger.warning(
                "VLLM_SIM_CPU_FALLBACK=1 — redirecting model runner "
                "device from %s to cpu for simulator compatibility.",
                device,
            )
            self._original_xpu_device = device
            device = torch.device("cpu")
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        # FIXME: To be verified.
        self.cascade_attn_enabled = False

    def _sync_device(self) -> None:
        if self._sim_cpu_fallback:
            # Nothing to synchronize on CPU.
            return
        torch.xpu.synchronize()


@contextmanager
def _torch_cuda_wrapper():
    try:
        # replace cuda APIs with xpu APIs, this should work by default
        torch.cuda.Stream = torch.xpu.Stream
        torch.cuda.default_stream = torch.xpu.current_stream
        torch.cuda.current_stream = torch.xpu.current_stream
        torch.cuda.stream = torch.xpu.stream
        torch.cuda.mem_get_info = torch.xpu.mem_get_info
        torch.cuda.synchronize = torch.xpu.synchronize
        if supports_xpu_graph():
            try:
                torch.cuda.graph = torch.xpu.graph
                torch.cuda.CUDAGraph = torch.xpu.XPUGraph
            except AttributeError:
                logger.warning("torch.xpu.graph not available, "
                               "XPU graph support disabled")
        yield
    finally:
        pass
