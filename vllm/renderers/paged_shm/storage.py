# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Paged shared memory storage:
Manages a shared memory segment divided into fixed-size blocks. Provides
iterators and read/write methods supporting both CPU and GPU direct (batch)
transfers.
"""

from multiprocessing import shared_memory
from unittest.mock import patch

import numpy as np
import torch
from torch._prims_common import DeviceLikeType

from vllm import _custom_ops as ops
from vllm.utils.torch_utils import PIN_MEMORY


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
                    assert self._shm.size >= self.size
                except FileNotFoundError:
                    raise FileNotFoundError(f"Shared memory '{name}' not found")
        assert self._shm.buf is not None, "Buffer was not created"

        self.name = self._shm.name
        self._shm_np = np.ndarray(self.size, dtype=self.dtype, buffer=self._shm.buf)
        self._shm_np.resize(self.n_block, self.block_size)

        self._shm_tensor = torch.from_numpy(self._shm_np)

        # Pin memory if requested and the global flag allows it
        self.is_pinned = False
        if pin and PIN_MEMORY:
            from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor

            pin_tensor(self._shm_tensor)
            self.is_pinned = True

    def get_iterator_numpy(self, size: int, blocks: list[int]):
        """Return a callable that yields (block_array, valid_length) tuples as numpy arrays."""

        def iterator():
            full_blocks = size // self.block_size
            remainder = size % self.block_size

            for i in range(full_blocks):
                blk = blocks[i]
                yield self._shm_np[blk], self.block_size

            if remainder > 0:
                blk = blocks[full_blocks]
                yield self._shm_np[blk], remainder

        return iterator

    def get_iterator_tensor(self, size: int, blocks: list[int]):
        """Return a callable that yields (block_tensor, valid_length) tuples as torch tensors."""

        def iterator():
            full_blocks = size // self.block_size
            remainder = size % self.block_size

            for i in range(full_blocks):
                blk = blocks[i]
                yield self._shm_tensor[blk], self.block_size

            if remainder > 0:
                blk = blocks[full_blocks]
                yield self._shm_tensor[blk], remainder

        return iterator

    def write(self, data: bytes | np.ndarray | torch.Tensor, blocks: list[int]) -> None:
        """Write data into the given blocks. Supports CPU bytes/numpy/tensor and GPU tensor."""
        if isinstance(data, torch.Tensor):
            if data.device.type != "cpu":
                self.write_from_device(data, blocks)
                return
            data_np = data.contiguous().view(torch.uint8).numpy()
        elif isinstance(data, bytes):
            data_np = np.frombuffer(data, dtype=np.uint8)
        elif isinstance(data, np.ndarray):
            data_np = np.ascontiguousarray(data).view(np.uint8)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        size = len(data_np)
        n_blocks = len(blocks)
        if size > n_blocks * self.block_size:
            raise ValueError("Data too large for provided blocks")

        it = self.get_iterator_numpy(size, blocks)()
        for i, (array, offset) in enumerate(it):
            start = i * self.block_size
            array[:offset] = data_np[start : start + offset]

    def write_from_device(self, data: torch.Tensor, blocks: list[int]) -> None:
        """GPU → CPU bulk copy using batched cuMemcpy (requires pinned memory)."""
        if not self.is_pinned:
            raise RuntimeError(
                "Cannot write from device: shared memory is not pinned. "
                "Initialize with pin=True and ensure PIN_MEMORY is enabled."
            )

        if data.device.type == "cpu":
            raise TypeError("write_from_device() requires a GPU tensor")

        data = data.contiguous().view(torch.uint8)
        size = data.numel()
        n_blocks = len(blocks)
        if size > n_blocks * self.block_size:
            raise ValueError("Data too large for provided blocks")

        data_ptr = data.data_ptr()

        src_addrs_list = []
        dst_addrs_list = []
        sizes_list = []

        it = self.get_iterator_tensor(size, blocks)()
        for i, (tensor, offset) in enumerate(it):
            start = i * self.block_size
            src_addrs_list.append(data_ptr + start)
            dst_addrs_list.append(tensor.data_ptr())
            sizes_list.append(offset)

        src_addrs = torch.tensor(src_addrs_list, dtype=torch.int64)
        dst_addrs = torch.tensor(dst_addrs_list, dtype=torch.int64)
        sizes = torch.tensor(sizes_list, dtype=torch.int64)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ops.swap_blocks_batch(src_addrs, dst_addrs, sizes)
        torch.cuda.current_stream().wait_stream(stream)

    def read_to_numpy(self, size: int, blocks: list[int]) -> np.ndarray:
        """Read data from the given blocks and return as a contiguous numpy array."""
        if size > len(blocks) * self.block_size:
            raise ValueError("Requested data too large for provided blocks")

        output_np = np.empty(size, dtype=np.uint8)

        it = self.get_iterator_numpy(size, blocks)()
        for i, (ndarray, offset) in enumerate(it):
            output_np[i * self.block_size : i * self.block_size + offset] = ndarray[
                :offset
            ]

        return output_np

    def read_to_tensor(
        self, size: int, blocks: list[int], device: DeviceLikeType = "cpu"
    ) -> torch.Tensor:
        """Read data into a torch tensor. If device != 'cpu', a GPU direct transfer is used."""
        if device != "cpu":
            return self.read_to_device(size, blocks, device)

        if size > len(blocks) * self.block_size:
            raise ValueError("Requested data too large for provided blocks")

        output_tensor = torch.empty(size, dtype=torch.uint8, device=device)

        it = self.get_iterator_tensor(size, blocks)()
        for i, (tensor, offset) in enumerate(it):
            output_tensor[i * self.block_size : i * self.block_size + offset] = tensor[
                :offset
            ]

        return output_tensor

    def read_to_device(self, size: int, blocks: list[int], device: DeviceLikeType):
        """CPU → GPU bulk copy using batched cuMemcpy (requires pinned memory)."""
        if not self.is_pinned:
            raise RuntimeError(
                "Cannot read to device: shared memory is not pinned. "
                "Initialize with pin=True and ensure PIN_MEMORY is enabled."
            )

        if size > len(blocks) * self.block_size:
            raise ValueError("Requested data too large for provided blocks")

        output_tensor = torch.empty(size, dtype=torch.uint8, device=device)

        if output_tensor.device.type == "cpu":
            raise TypeError("read_to_device() requires a GPU tensor")

        data_ptr = output_tensor.data_ptr()

        src_addrs_list = []
        dst_addrs_list = []
        sizes_list = []

        it = self.get_iterator_tensor(size, blocks)()
        for i, (tensor, offset) in enumerate(it):
            start = i * self.block_size
            src_addrs_list.append(tensor.data_ptr())
            dst_addrs_list.append(data_ptr + start)
            sizes_list.append(offset)

        src_addrs = torch.tensor(src_addrs_list, dtype=torch.int64)
        dst_addrs = torch.tensor(dst_addrs_list, dtype=torch.int64)
        sizes = torch.tensor(sizes_list, dtype=torch.int64)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ops.swap_blocks_batch(src_addrs, dst_addrs, sizes)
        torch.cuda.current_stream().wait_stream(stream)

        return output_tensor

    def close(self):
        if hasattr(self, "_shm"):
            self._shm.close()

        if self.pin:
            from vllm.v1.simple_kv_offload.cuda_mem_ops import unpin_tensor

            unpin_tensor(self._shm_tensor)

        del self._shm_tensor, self._shm_np

    def unlink(self):
        if self._created and hasattr(self, "_shm"):
            self._shm.unlink()
            del self._shm

    def __del__(self):
        self.close()
        self.unlink()
