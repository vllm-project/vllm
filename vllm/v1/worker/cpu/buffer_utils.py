# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import torch

from vllm.utils.platform_utils import is_uva_available


class UvaBuffer:
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        if not is_uva_available():
            raise RuntimeError("UVA is not available")
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()
        self.uva = self.cpu


def apply_write(self) -> None:
    """Torch equivalent of ``StagedWriteTensor.apply_write``.

    Writes straight from the staged Python lists, skipping the UVA staging
    buffers and host-to-device copy the GPU path needs.
    """
    if not self._staged_write_indices:
        return

    # Mirror the kernel's flat offset arithmetic so this holds for any rank.
    flat = self.gpu.view(-1)
    row_stride = self.gpu.stride(0)
    cu_start = 0
    for i, row_idx in enumerate(self._staged_write_indices):
        cu_end = self._staged_write_cu_lens[i]
        content = self._staged_write_contents[cu_start:cu_end]
        cu_start = cu_end
        if not content:
            continue
        offset = row_idx * row_stride + self._staged_write_starts[i]
        flat[offset : offset + len(content)] = torch.tensor(
            content, dtype=self.gpu.dtype
        )

    self.clear_staged_writes()


def fused_apply(self, tensors: Sequence, output_ptrs, output_strides) -> None:
    """Torch equivalent of ``FusedStagedWriter.apply``.

    The fused kernel exists to avoid a per-group launch; on CPU each tensor
    can just apply its own staged writes.
    """
    for tensor in tensors:
        tensor.apply_write()
