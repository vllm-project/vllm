# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Paged shared memory storage
The paged shared memory storage is utilized by three components.
- PagedShmServer: Manages the entire lifecycle—creates on start, unlinks on stop.
    No I/O operations.
- API Server: Opens existing memory, writes preprocessed Multi-modal Tensors, and
    closes (without unlinking) on exit.
- GPU Worker: Opens existing memory, reads data with PIN_MEMORY-accelerated H2D
    transfers, and closes (without unlinking) on exit.
"""

import weakref
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from multiprocessing import shared_memory
from unittest.mock import patch

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.utils.torch_utils import PIN_MEMORY, DeviceLikeType


class PagedShmStorage:
    def __init__(
        self, size: int, block_size: int, *, name: str | None = None, pin: bool = False
    ):
        self.name = name
        self.pin = pin
        self.block_size = block_size
        self.n_block = size // block_size
        self.size = block_size * self.n_block
        self.dtype = np.uint8
        self._created = name is None

        self._resources = ExitStack()
        self._finalizer = weakref.finalize(self, self._resources.close)

        if self._created:
            self._shm = shared_memory.SharedMemory(create=True, size=self.size)
        else:
            # Avoid resource tracker warnings when attaching to existing segment
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                try:
                    self._shm = shared_memory.SharedMemory(name=name)
                    if self._shm.size < self.size:
                        raise ValueError(
                            f"Existing shared memory segment '{name}' is too small "
                            f"({self._shm.size} < {self.size})"
                        )
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Shared memory '{name}' not found"
                    ) from None
        assert self._shm.buf is not None, "Buffer was not created"
        self._resources.callback(_close_shm, self._shm, self._created)

        self.name = self._shm.name
        self._shm_np = np.ndarray(self.size, dtype=self.dtype, buffer=self._shm.buf)
        self._shm_np.resize(self.n_block, self.block_size)
        self._shm_tensor = torch.from_numpy(self._shm_np)

        self.is_pinned = False
        if pin and PIN_MEMORY:
            from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor, unpin_tensor

            pin_tensor(self._shm_tensor)
            self.is_pinned = True
            self._resources.callback(unpin_tensor, self._shm_tensor)

    def _ensure_pinned(self) -> None:
        """Raise an error if the shared memory is not pinned."""
        if not self.is_pinned:
            raise RuntimeError(
                "Cannot perform GPU transfer: shared memory is not pinned. "
                "Initialize with pin=True and ensure PIN_MEMORY is enabled."
            )

    def _validate_blocks(self, size: int, blocks: list[int]) -> None:
        """
        Validate size and block indices for a read/write operation.
        """
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")
        if not blocks:
            raise ValueError("Blocks list cannot be empty")
        if size > len(blocks) * self.block_size:
            raise ValueError(
                f"Requested size {size} exceeds capacity "
                f"{len(blocks) * self.block_size}"
            )
        max_idx = max(blocks)
        if max_idx >= self.n_block:
            raise ValueError(
                f"Block index {max_idx} out of range (max {self.n_block - 1})"
            )

    def _iterate_blocks(self, size: int, blocks: list[int], as_numpy: bool = True):
        """
        Yield (buffer_view, chunk_size, offset_in_data) for each block.
        `buffer_view` is either a numpy array or a torch tensor.
        """
        data = self._shm_np if as_numpy else self._shm_tensor
        full_blocks = size // self.block_size
        remainder = size % self.block_size

        for i in range(full_blocks):
            yield data[blocks[i]], self.block_size, i * self.block_size

        if remainder > 0:
            yield data[blocks[full_blocks]], remainder, full_blocks * self.block_size

    def _build_address_lists(
        self, size: int, blocks: list[int], data_ptr: int, is_read: bool
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Build source/destination address lists for batched CUDA transfers.
        Returns (src_addrs, dst_addrs, sizes).
        """
        src_addrs_list: list[int] = []
        dst_addrs_list: list[int] = []
        sizes_list: list[int] = []

        for tensor, offset, start in self._iterate_blocks(size, blocks, as_numpy=False):
            if is_read:  # CPU (shared memory) -> GPU
                src_addrs_list.append(tensor.data_ptr())
                dst_addrs_list.append(data_ptr + start)
            else:  # GPU -> CPU (shared memory)
                src_addrs_list.append(data_ptr + start)
                dst_addrs_list.append(tensor.data_ptr())
            sizes_list.append(offset)

        return src_addrs_list, dst_addrs_list, sizes_list

    def _batched_transfer(
        self,
        src_addrs: list[int],
        dst_addrs: list[int],
        sizes: list[int],
    ) -> None:
        """
        Execute a batched memcpy using the custom CUDA operation.

        Args:
            src_addrs: List of source addresses.
            dst_addrs: List of destination addresses.
            sizes: List of transfer sizes in bytes.
        """
        src_t = torch.tensor(src_addrs, dtype=torch.int64, device="cpu")
        dst_t = torch.tensor(dst_addrs, dtype=torch.int64, device="cpu")
        sizes_t = torch.tensor(sizes, dtype=torch.int64, device="cpu")

        current_stream = torch.cuda.current_stream()
        default_stream = torch.cuda.default_stream()
        sync = current_stream == default_stream
        stream = torch.cuda.Stream() if sync else current_stream

        with torch.cuda.stream(stream):
            ops.swap_blocks_batch(src_t, dst_t, sizes_t)

        if sync:
            stream.synchronize()

    def get_iterator_numpy(
        self, size: int, blocks: list[int]
    ) -> Callable[[], Iterator[tuple[np.ndarray, int]]]:
        """
        Return a callable that yields (numpy_view, chunk_size) for each block.
        The view is a slice of the shared memory buffer.
        """
        self._validate_blocks(size, blocks)

        def iterator():
            for array, offset, _ in self._iterate_blocks(size, blocks, as_numpy=True):
                yield array, offset

        return iterator

    def get_iterator_tensor(
        self, size: int, blocks: list[int]
    ) -> Callable[[], Iterator[tuple[torch.Tensor, int]]]:
        """
        Return a callable that yields (torch_tensor_view, chunk_size) for each block.
        The view is a slice of the shared memory buffer.
        """
        self._validate_blocks(size, blocks)

        def iterator():
            for tensor, offset, _ in self._iterate_blocks(size, blocks, as_numpy=False):
                yield tensor, offset

        return iterator

    def _write_cpu(self, data_np: np.ndarray, blocks: list[int]) -> None:
        """Write CPU data (as contiguous uint8 numpy array) into blocks."""
        size = data_np.shape[0]
        self._validate_blocks(size, blocks)
        for array, offset, start in self._iterate_blocks(size, blocks, as_numpy=True):
            array[:offset] = data_np[start : start + offset]

    def _write_gpu(self, data: torch.Tensor, blocks: list[int]) -> None:
        """Write GPU tensor data into blocks via batched GPU->CPU transfer."""
        self._ensure_pinned()
        if data.device.type == "cpu":
            raise TypeError("_write_gpu() requires a GPU tensor")
        data = data.contiguous().view(torch.uint8)
        size = data.numel()
        self._validate_blocks(size, blocks)

        data_ptr = data.data_ptr()
        src_addrs, dst_addrs, sizes = self._build_address_lists(
            size, blocks, data_ptr, is_read=False
        )
        self._batched_transfer(src_addrs, dst_addrs, sizes)

    def write(self, data: bytes | np.ndarray | torch.Tensor, blocks: list[int]) -> None:
        """
        Write data into the given blocks.
        - If `data` is a GPU tensor, it is transferred via the GPU path.
        - Otherwise, CPU data (bytes, numpy array, or CPU tensor) is written directly.
        """
        if isinstance(data, torch.Tensor):
            if data.device.type != "cpu":
                self._write_gpu(data, blocks)
                return
            data_np = data.contiguous().view(torch.uint8).numpy()
        elif isinstance(data, bytes):
            data_np = np.frombuffer(data, dtype=np.uint8)
        elif isinstance(data, np.ndarray):
            data_np = np.ascontiguousarray(data).view(np.uint8)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        self._write_cpu(data_np.flatten(), blocks)

    def write_from_device(self, data: torch.Tensor, blocks: list[int]) -> None:
        """
        Write a GPU tensor directly into shared memory (GPU -> CPU).
        The shared memory must be pinned.
        """
        self._write_gpu(data, blocks)

    def read_to_numpy(self, size: int, blocks: list[int]) -> np.ndarray:
        """
        Read data from blocks and return as a contiguous numpy array (CPU).
        """
        self._validate_blocks(size, blocks)
        out = np.empty(size, dtype=np.uint8)
        for array, offset, start in self._iterate_blocks(size, blocks, as_numpy=True):
            out[start : start + offset] = array[:offset]
        return out

    def read_to_tensor(
        self,
        size: int,
        blocks: list[int],
        device: DeviceLikeType = "cpu",
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Read data into a torch tensor.
        If `device` is not CPU, a batched GPU transfer is used (requires pinned memory).
        If `out` is provided, it will be used (must have correct size and device);
        otherwise a new tensor is allocated.
        """
        if device != "cpu":
            return self.read_to_device(size, blocks, device, out=out)

        self._validate_blocks(size, blocks)
        if out is None:
            out = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
        else:
            if out.numel() != size:
                raise ValueError(
                    f"Output tensor size {out.numel()} does not match requested {size}"
                )
            if out.device.type != "cpu":
                raise ValueError("Output tensor must be on CPU for CPU read")
        for tensor, offset, start in self._iterate_blocks(size, blocks, as_numpy=False):
            out[start : start + offset] = tensor[:offset]
        return out

    def read_to_device(
        self,
        size: int,
        blocks: list[int],
        device: DeviceLikeType,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Read data from blocks into a GPU tensor via batched transfer.
        Requires pinned shared memory.
        If `out` is provided, it is used; otherwise a new tensor is allocated.
        """
        self._ensure_pinned()
        self._validate_blocks(size, blocks)

        if out is None:
            out = torch.empty(size, dtype=torch.uint8, device=device)
        else:
            if out.numel() != size:
                raise ValueError(
                    f"Output tensor size {out.numel()} does not match requested {size}"
                )
            if out.device.type != "cuda":
                raise ValueError("Output tensor must be on GPU for read_to_device")
        if out.device.type == "cpu":
            raise TypeError("read_to_device() requires a GPU tensor")

        data_ptr = out.data_ptr()
        src_addrs, dst_addrs, sizes = self._build_address_lists(
            size, blocks, data_ptr, is_read=True
        )
        self._batched_transfer(src_addrs, dst_addrs, sizes)
        return out

    def close(self) -> None:
        """Release the shared memory segment and any pinned resources."""
        self._resources.close()


def _close_shm(shm: shared_memory.SharedMemory, created: bool = False) -> None:
    """Close and (if created) unlink the shared memory segment."""
    if created:
        shm.unlink()
    shm.close()
