# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence

import numpy as np
import torch


class UvaBuffer:
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()

    def uva(self, n: int | None = None):
        return self.cpu[:n] if n is not None else self.cpu


class UvaBufferPool:
    """CPU stand-in for the staging pool.

    The pool rotates through several pinned buffers so a host write for step
    N+1 cannot land in the buffer an in-flight DMA is still reading for step
    N. With one address space and no asynchronous copy there is no second
    reader, so the source is handed back directly and both the rotation and
    the copy disappear.
    """

    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int | None = None,
    ):
        self.size = size
        self.dtype = dtype

    def copy_to_uva(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return torch.tensor(x, dtype=self.dtype)

    def copy_to_gpu(
        self,
        x: torch.Tensor | np.ndarray,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        uva = self.copy_to_uva(x)
        return uva.clone() if out is None else out.copy_(uva)


class UvaBackedTensor:
    """CPU stand-in where the device view *is* the source tensor.

    Upstream keeps an unpinned source and republishes it into a pinned pool
    buffer every step. Here the two are the same memory, so republishing is
    just handing back the (possibly truncated) view.
    """

    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int | None = None,
    ):
        self.dtype = dtype
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()
        self.gpu = self.cpu

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        self.gpu = self.cpu if n is None else self.cpu[:n]
        return self.gpu


class StagedWriteTensor:
    """CPU stand-in that writes through instead of staging.

    Staging batches many small host-to-device writes into one transfer and
    one kernel launch, which is only worth its bookkeeping when the write
    has to cross to another device. Here it does not, so each write lands
    in place and `apply_write` has nothing left to do.
    """

    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        max_concurrency: int | None = None,
        uva_instead_of_gpu: bool = False,
    ):
        supported_dtypes = [torch.int32, torch.int64, torch.float32]
        if dtype not in supported_dtypes:
            raise ValueError(
                f"Unsupported dtype {dtype}: should be one of {supported_dtypes}"
            )
        self.num_rows = size if isinstance(size, int) else size[0]
        self.dtype = dtype
        self.device = device

        # There is one memory pool, so `uva_instead_of_gpu` has nothing to
        # choose between and the tensor is allocated directly either way.
        self.gpu = torch.zeros(size, dtype=dtype, device="cpu")
        # The upstream kernel addresses rows through gpu.stride(0), so a flat
        # view reproduces its indexing for both the 1-D and 2-D tensors.
        self._flat = self.gpu.view(-1)
        self._row_stride = self.gpu.stride(0)

        # FusedStagedWriter reads these directly; staying empty makes it a
        # no-op for the multi-group block-table path.
        self._staged_write_indices: list[int] = []
        self._staged_write_starts: list[int] = []
        self._staged_write_contents: list[int | float] = []
        self._staged_write_cu_lens: list[int] = []

    def stage_write(
        self, index: int, start: int, x: Iterable[int] | Iterable[float]
    ) -> None:
        assert index >= 0
        assert start >= 0
        if not x:
            return
        if not isinstance(x, list):
            x = list(x)
        offset = index * self._row_stride + start
        self._flat[offset : offset + len(x)] = torch.tensor(x, dtype=self.dtype)

    def stage_write_elem(self, index: int, x: int) -> None:
        assert index >= 0
        self._flat[index * self._row_stride] = x

    def apply_write(self) -> None:
        pass

    def clear_staged_writes(self) -> None:
        pass
