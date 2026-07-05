# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# isort: skip_file
# ruff: noqa: E402
# mypy: disable-error-code="misc, assignment"

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


torch.Event = _EventPlaceholder
torch.cuda.Stream = _StreamPlaceholder
torch.cuda.set_stream = noop
torch.cuda.current_stream = lambda *args, **kwargs: _StreamPlaceholder()
torch.accelerator.synchronize = noop
torch.accelerator.empty_cache = noop
torch.Tensor.pin_memory = fake_pin_memory

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

gpu_buffer_utils.UvaBuffer = cpu_buffer_utils.UvaBuffer
