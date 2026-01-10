# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.platform_utils import is_pin_memory_available, is_uva_available
from vllm.utils.torch_utils import get_cuda_view_from_cpu_tensor


class UvaBuffer:
    def __init__(self, size: int | Sequence[int | torch.SymInt], dtype: torch.dtype):
        if not is_uva_available():
            raise RuntimeError("UVA is not available")
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        self.np = self.cpu.numpy()
        self.uva = get_cuda_view_from_cpu_tensor(self.cpu)


class UvaBackedTensor:
    def __init__(
        self,
        size: int | Sequence[int | torch.SymInt],
        dtype: torch.dtype,
        max_concurrency: int = 2,
    ):
        self.dtype = dtype
        self.max_concurrency = max_concurrency

        # Source of truth
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)
        self.np = self.cpu.numpy()

        # Buffers for concurrency
        self._uva_bufs = [UvaBuffer(size, dtype) for _ in range(max_concurrency)]

        # Current buffer index and its GPU view
        self._curr = 0
        self.gpu = self._uva_bufs[self._curr].uva

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._uva_bufs[self._curr]
        # CPU-to-CPU copy
        buf.np[:n] = self.np[:n]
        # Zero-copy
        self.gpu = buf.uva
        return self.gpu[:n]


class StagedWriteTensor:
    def __init__(
        self,
        size: int | Sequence[int | torch.SymInt],
        dtype: torch.dtype,
        device: torch.device,
        max_concurrency: int = 2,
        uva_instead_of_gpu: bool = False,
    ):
        if dtype not in [torch.int32, torch.int64]:
            raise ValueError(
                f"Unsupported dtype {dtype}: should be either int32 or int64"
            )
        self.num_rows = size if isinstance(size, int) else size[0]
        self.dtype = dtype
        self.max_concurrency = max_concurrency

        if not uva_instead_of_gpu:
            # Create a GPU tensor (default)
            self.gpu = torch.zeros(size, dtype=dtype, device=device)
        else:
            # For a large but not-frequently-accessed tensor, we can use UVA instead of
            # GPU to save GPU memory
            self._uva_buf = UvaBuffer(size, dtype)
            self.gpu = self._uva_buf.uva

        self._staged_write_indices: list[int] = []
        self._staged_write_starts: list[int] = []
        self._staged_write_contents: list[int] = []
        self._staged_write_cu_lens: list[int] = []

        self.write_indices = UvaBackedTensor(
            self.num_rows, dtype=torch.int32, max_concurrency=max_concurrency
        )
        self.write_starts = UvaBackedTensor(
            self.num_rows, dtype=torch.int32, max_concurrency=max_concurrency
        )
        init_size = next_power_of_2(self.num_rows)
        self.write_contents = UvaBackedTensor(
            init_size, dtype=dtype, max_concurrency=max_concurrency
        )
        self.write_cu_lens = UvaBackedTensor(
            self.num_rows, dtype=torch.int32, max_concurrency=max_concurrency
        )

    def stage_write(self, index: int, start: int, x: list[int]) -> None:
        assert index >= 0
        assert start >= 0
        if not x:
            return
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(start)
        self._staged_write_contents.extend(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def stage_write_elem(self, index: int, x: int) -> None:
        assert index >= 0
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(0)
        self._staged_write_contents.append(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def prepare(self) -> None:
        n = len(self._staged_write_indices)
        if n == 0:
            return

        self.write_indices.np[:n] = self._staged_write_indices
        self.write_indices.copy_to_gpu(n)

        self.write_starts.np[:n] = self._staged_write_starts
        self.write_starts.copy_to_gpu(n)

        self.write_cu_lens.np[:n] = self._staged_write_cu_lens
        self.write_cu_lens.copy_to_gpu(n)

        # Special handling for write_contents
        diff_len = len(self._staged_write_contents)
        if diff_len > self.write_contents.np.shape[0]:
            # Re-allocate a larger buffer for the write_contents
            new_size = next_power_of_2(diff_len)
            self.write_contents = UvaBackedTensor(
                new_size, dtype=self.dtype, max_concurrency=self.max_concurrency
            )
        self.write_contents.np[:diff_len] = self._staged_write_contents
        self.write_contents.copy_to_gpu(diff_len)

    def apply_write(self) -> None:
        n = len(self._staged_write_indices)
        if n == 0:
            return

        # Prepare the buffers for staging writes
        self.prepare()
        # Write diffs to the GPU buffer
        _apply_write_kernel[(n,)](
            self.gpu,
            self.gpu.stride(0),
            self.write_indices.gpu,
            self.write_starts.gpu,
            self.write_contents.gpu,
            self.write_cu_lens.gpu,
            BLOCK_SIZE=1024,
        )
        # Clear the staged writes
        self.clear_staged_writes()

    def clear_staged_writes(self) -> None:
        self._staged_write_indices.clear()
        self._staged_write_starts.clear()
        self._staged_write_contents.clear()
        self._staged_write_cu_lens.clear()


@triton.jit
def _apply_write_kernel(
    output_ptr,
    output_stride,
    write_indices_ptr,
    write_starts_ptr,
    write_contents_ptr,
    write_cu_lens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = tl.load(write_indices_ptr + pid)
    start_idx = tl.load(write_starts_ptr + pid)

    cu_start = tl.load(write_cu_lens_ptr + pid - 1) if pid > 0 else 0
    cu_end = tl.load(write_cu_lens_ptr + pid)
    content_len = cu_end - cu_start

    for i in range(0, content_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < content_len
        content = tl.load(write_contents_ptr + cu_start + block, mask=mask)
        tl.store(
            output_ptr + row_idx * output_stride + start_idx + block, content, mask=mask
        )


class DoubleBufferTensor:
    def __init__(
        self,
        size: int | Sequence[int | torch.SymInt],
        dtype: torch.dtype,
        device: torch.device,
        max_concurrency: int = 2,
    ):
        self.dtype = dtype
        self.device = device
        self.max_concurrency = max_concurrency

        # Source of truth
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)
        self.np = self.cpu.numpy()

        # CPU buffers for concurrency
        if not is_pin_memory_available():
            raise RuntimeError("Pin memory is not available")
        self._bufs = [
            torch.zeros_like(self.cpu, device="cpu", pin_memory=True)
            for _ in range(max_concurrency)
        ]
        # Current buffer index
        self._curr = 0

        # Destination GPU tensor
        self.gpu = torch.zeros_like(self.cpu, device=device)

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._bufs[self._curr]
        # CPU-to-CPU copy
        buf[:n] = self.cpu[:n]
        # CPU-to-GPU copy
        return self.gpu[:n].copy_(buf[:n], non_blocking=True)
