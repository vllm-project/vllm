# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import torch

from vllm.utils.platform_utils import is_uva_available
from vllm.v1.worker.gpu.buffer_utils import (
    FusedStagedWriter as GPUFusedStagedWriter,
)
from vllm.v1.worker.gpu.buffer_utils import (
    StagedWriteTensor as GPUStagedWriteTensor,
)


class UvaBuffer:
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        if not is_uva_available():
            raise RuntimeError("UVA is not available")
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()
        self.uva = self.cpu


class StagedWriteTensor(GPUStagedWriteTensor):
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


class FusedStagedWriter(GPUFusedStagedWriter):
    def apply(
        self,
        tensors: Sequence[GPUStagedWriteTensor],
        output_ptrs: torch.Tensor,
        output_strides: torch.Tensor,
    ) -> None:
        for tensor in tensors:
            tensor.apply_write()
