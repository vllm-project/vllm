# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# isort: skip_file
# ruff: noqa: E402
# mypy: disable-error-code="misc, assignment"

from collections.abc import Sequence
from typing import Any

import numpy as np

# Patch torch APIs
import torch


def noop(*args: Any, **kwargs: Any) -> None:
    pass


def fake_pin_memory(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    return self


class _EventPlaceholder:
    def __init__(self, *args, **kwargs) -> None:
        self.record = noop
        self.synchronize = noop


class _StreamPlaceholder:
    def __init__(self, *args, **kwargs) -> None:
        self.wait_stream = noop
        self.device = torch.device("cpu")

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


from vllm.utils.cpu_resource_utils import get_memory_node_info


def get_memory_info(*args: Any, **kwargs: Any) -> tuple[int, int]:
    meminfo = get_memory_node_info()
    return meminfo.available_memory, meminfo.total_memory


torch.Event = _EventPlaceholder
torch.cuda.Event = _EventPlaceholder
torch.cuda.Stream = _StreamPlaceholder
torch.cuda.set_stream = noop
torch.cuda.current_stream = lambda *args, **kwargs: _StreamPlaceholder()
torch.accelerator.synchronize = noop
torch.accelerator.empty_cache = noop
torch.Tensor.pin_memory = fake_pin_memory
torch.accelerator.get_memory_info = get_memory_info

# Patch vLLM torch utils
import vllm.utils.torch_utils as torch_utils


def async_tensor_h2d(
    data: list | np.ndarray | torch.Tensor,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    return torch.tensor(data, dtype=dtype, device="cpu")


torch_utils.async_tensor_h2d = async_tensor_h2d

# Patch model runner APIs
import vllm.v1.worker.gpu.buffer_utils as gpu_buffer_utils
import vllm.v1.worker.cpu.buffer_utils as cpu_buffer_utils
from vllm.triton_utils import HAS_TRITON

gpu_buffer_utils.UvaBuffer = cpu_buffer_utils.UvaBuffer


class CPUStagedWriteTensor(gpu_buffer_utils.StagedWriteTensor):
    def apply_write(self) -> None:
        offset = 0
        for index, start, end in zip(
            self._staged_write_indices,
            self._staged_write_starts,
            self._staged_write_cu_lens,
        ):
            values = torch.tensor(
                self._staged_write_contents[offset:end],
                dtype=self.dtype,
                device=self.gpu.device,
            )
            row_offset = index * self.gpu.stride(0) + start
            self.gpu.reshape(-1)[row_offset : row_offset + len(values)] = values
            offset = end
        self.clear_staged_writes()


class CPUFusedStagedWriter(gpu_buffer_utils.FusedStagedWriter):
    def apply(
        self,
        tensors: Sequence[gpu_buffer_utils.StagedWriteTensor],
        output_ptrs: torch.Tensor,
        output_strides: torch.Tensor,
    ) -> None:
        for tensor in tensors:
            tensor.apply_write()


if not HAS_TRITON:
    gpu_buffer_utils.StagedWriteTensor = CPUStagedWriteTensor
    gpu_buffer_utils.FusedStagedWriter = CPUFusedStagedWriter
