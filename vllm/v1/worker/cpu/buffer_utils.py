# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.buffer_utils import FusedStagedWriter, StagedWriteTensor


def _staged_write_tensor_apply_write(self: "StagedWriteTensor") -> None:
    if not self._staged_write_indices:
        return

    write_contents = torch.tensor(
        self._staged_write_contents,
        dtype=self.dtype,
        device=self.device,
    )
    flat_output = self.gpu.view(-1)
    output_stride = self.gpu.stride(0)
    cu_start = 0
    writes = zip(
        self._staged_write_indices,
        self._staged_write_starts,
        self._staged_write_cu_lens,
        strict=True,
    )
    for row_idx, start_idx, cu_end in writes:
        output_start = row_idx * output_stride + start_idx
        content_len = cu_end - cu_start
        flat_output[output_start : output_start + content_len].copy_(
            write_contents[cu_start:cu_end]
        )
        cu_start = cu_end

    self.clear_staged_writes()


def _fused_staged_writer_apply(
    self: "FusedStagedWriter",
    tensors: Sequence["StagedWriteTensor"],
    output_ptrs: torch.Tensor,
    output_strides: torch.Tensor,
) -> None:
    del self, output_ptrs, output_strides
    for tensor in tensors:
        _staged_write_tensor_apply_write(tensor)


class UvaBuffer:
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()

    def uva(self, n: int | None = None):
        return self.cpu[:n] if n is not None else self.cpu
