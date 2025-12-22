# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import ctypes
import math
from collections import OrderedDict
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class MemoryBlock:
    size: int
    addr: int


"""A memory pool for managing pinned host memory allocations for tensors.

This class implements a buddy allocation system to efficiently manage pinned
host memory for tensor storage. It supports allocation, deallocation, and
tensor storage/retrieval operations.

Key Features:
- Uses power-of-two block sizes for efficient buddy allocation
- Supports splitting and merging of memory blocks
- Provides methods to store CUDA tensors in pinned host memory
- Allows loading tensors from pinned memory back to device
- Automatically cleans up memory on destruction

Attributes:
    max_block_size (int): Maximum block size (rounded to nearest power of two)
    min_block_size (int): Minimum block size (rounded to nearest power of two)
    free_lists (dict): Dictionary of free memory blocks by size
    allocated_blocks (dict): Dictionary of currently allocated blocks
    base_tensor (torch.Tensor): Base pinned memory tensor
    base_address (int): Base memory address of the pinned memory region

Example:
    >>> pool = TensorMemoryPool(max_block_size=1024*1024)
    >>> tensor = torch.randn(100, device='cuda')
    >>> addr = pool.store_tensor(tensor)
    >>> loaded_tensor = pool.load_tensor(addr, tensor.dtype,
    ...                                  tensor.shape, 'cuda')
    >>> pool.free(addr)
"""


class TensorMemoryPool:
    """Initializes the memory pool with given size constraints.

    Args:
        max_block_size (int): Maximum size of memory blocks to manage
        min_block_size (int, optional): Minimum size of memory blocks
            to manage. Defaults to 512.
        device_type (str, optional): Type of memory pool - 'cpu' for pinned
            host memory or 'cuda' for GPU memory. Defaults to 'cpu'.

    Raises:
        ValueError: If block sizes are invalid or max_block_size is less
            than min_block_size
    """

    def __init__(
        self,
        max_block_size: int,
        min_block_size: int = 512,
        device_type: str = "cpu",
        auto_evict: bool = False,
    ):
        if max_block_size <= 0 or min_block_size <= 0:
            raise ValueError("Block sizes must be positive")
        if max_block_size < min_block_size:
            raise ValueError("Max block size must be greater than min block size")
        if device_type not in ["cpu", "cuda"]:
            raise ValueError("device_type must be 'cpu' or 'cuda'")

        self.max_block_size = self._round_to_power_of_two(max_block_size)
        self.min_block_size = self._round_to_power_of_two(min_block_size)
        self.device_type = device_type
        self.auto_evict = auto_evict

        self.free_lists: dict[int, dict[int, MemoryBlock]] = {}
        self.allocated_blocks: OrderedDict[int, MemoryBlock] = {}

        self._initialize_free_lists()
        self._allocate_memory()

        atexit.register(self.cleanup)

    def _round_to_power_of_two(self, size: int) -> int:
        return 1 << (size - 1).bit_length()

    def _initialize_free_lists(self):
        size = self.max_block_size
        while size >= self.min_block_size:
            self.free_lists[size] = {}
            size //= 2

    def _allocate_memory(self):
        if self.device_type == "cpu":
            self.base_tensor = torch.empty(
                self.max_block_size // 4, dtype=torch.float32, pin_memory=True
            )
        else:  # cuda
            self.base_tensor = torch.empty(
                self.max_block_size // 4, dtype=torch.float32, device="cuda"
            ).contiguous()

        self.base_address = self.base_tensor.data_ptr()
        initial_block = MemoryBlock(size=self.max_block_size, addr=self.base_address)
        self.free_lists[self.max_block_size][initial_block.addr] = initial_block

        logger.debug(
            "TensorMemoryPool, device_type:%s, base_address:%d, max_block_size:%d",
            self.device_type,
            self.base_address,
            self.max_block_size,
        )

    def _allocate(self, required_size: int) -> int:
        current_size = required_size
        while current_size <= self.max_block_size:
            if self.free_lists[current_size]:
                _, block = self.free_lists[current_size].popitem()
                self._split_block(block, required_size)
                self.allocated_blocks[block.addr] = block
                return block.addr
            current_size *= 2

        raise ValueError("Insufficient memory")

    def allocate(self, size: int) -> int:
        """Allocates a memory block of at least the requested size.

        Args:
            size (int): Minimum size of memory to allocate

        Returns:
            int: Address of the allocated memory block

        Raises:
            ValueError: If size is invalid or insufficient memory is available
        """
        if size <= 0:
            raise ValueError("Allocation size must be positive")

        required_size = self._round_to_power_of_two(max(size, self.min_block_size))
        if required_size > self.max_block_size:
            raise ValueError("Requested size exceeds maximum block size")

        while True:
            try:
                return self._allocate(required_size)
            except ValueError:
                if self.auto_evict:
                    self.free()
                else:
                    raise

    def _split_block(self, block: MemoryBlock, required_size: int):
        while block.size > required_size and block.size // 2 >= self.min_block_size:
            buddy_size = block.size // 2
            buddy_addr = block.addr + buddy_size

            buddy = MemoryBlock(size=buddy_size, addr=buddy_addr)
            block.size = buddy_size

            self.free_lists[buddy_size][buddy.addr] = buddy

    def free(self, addr: int | None = None):
        """Frees an allocated memory block.

        Args:
            addr (int): Address of the block to free

        Raises:
            ValueError: If address is invalid or not allocated
        """
        if not addr:
            if self.allocated_blocks:
                # Retrieved the earliest inserted key
                addr = next(iter(self.allocated_blocks))
            else:
                raise ValueError("No available block to free")

        if addr not in self.allocated_blocks:
            raise ValueError("Invalid address to free")

        block = self.allocated_blocks.pop(addr)
        self._merge_buddies(block)

    def _merge_buddies(self, block: MemoryBlock):
        MAX_MERGE_DEPTH = 30
        depth = 0

        while depth < MAX_MERGE_DEPTH:
            buddy_offset = (
                block.size
                if (block.addr - self.base_address) % (2 * block.size) == 0
                else -block.size
            )
            buddy_addr = block.addr + buddy_offset
            buddy = self.free_lists[block.size].get(buddy_addr)
            if buddy:
                del self.free_lists[buddy.size][buddy.addr]
                merged_addr = min(block.addr, buddy.addr)
                merged_size = block.size * 2
                block = MemoryBlock(size=merged_size, addr=merged_addr)
                depth += 1
            else:
                break
        self.free_lists[block.size][block.addr] = block

    def store_tensor(self, tensor: torch.Tensor) -> int:
        """Stores a tensor in the memory pool.

        Args:
            tensor (torch.Tensor): Tensor to store (CUDA for both CPU and GPU pools)

        Returns:
            int: Address where the tensor is stored

        Raises:
            ValueError: If tensor device is incompatible or allocation fails
        """
        if not tensor.is_cuda:
            raise ValueError("Only CUDA tensors can be stored")

        size = tensor.element_size() * tensor.numel()
        addr = self.allocate(size)
        block = self.allocated_blocks[addr]

        if block.size < size:
            self.free(addr)
            raise ValueError(
                f"Allocated block size {block.size} is smaller than "
                f"required size {size}"
            )

        try:
            buffer = (ctypes.c_byte * block.size).from_address(block.addr)
            pool_tensor = torch.frombuffer(
                buffer, dtype=tensor.dtype, count=tensor.numel()
            ).reshape(tensor.shape)
        except ValueError as err:
            self.free(addr)
            raise ValueError(f"Failed to create tensor view: {err}") from err

        pool_tensor.copy_(tensor)

        return addr

    def load_tensor(
        self,
        addr: int,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        device: torch.device = None,
        copy: bool = True,
    ) -> torch.Tensor:
        """Loads a tensor from the memory pool.

        Args:
            addr (int): Address where tensor is stored
            dtype (torch.dtype): Data type of the tensor
            shape (tuple[int, ...]): Shape of the tensor
            device (torch.device, optional): Target device for the loaded tensor.
                Required if copy=True, ignored if copy=False.
            copy (bool, optional): If True, copies tensor to device. If False,
                returns a view at the stored address. Defaults to True.

        Returns:
            torch.Tensor: The loaded tensor (copied to device or view at address)

        Raises:
            ValueError: If address is invalid, sizes don't match, or device not
                specified when copy=True
        """
        if addr not in self.allocated_blocks:
            raise ValueError("Invalid address to load")

        block = self.allocated_blocks[addr]
        num_elements = math.prod(shape)
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        required_size = num_elements * dtype_size

        if required_size > block.size:
            raise ValueError("Requested tensor size exceeds block size")

        buffer = (ctypes.c_byte * block.size).from_address(block.addr)
        pool_tensor = torch.frombuffer(buffer, dtype=dtype, count=num_elements).reshape(
            shape
        )

        if not copy:
            return pool_tensor

        target_tensor = torch.empty(shape, dtype=dtype, device=device)
        target_tensor.copy_(pool_tensor)

        return target_tensor

    def cleanup(self):
        """Cleans up all memory resources and resets the pool state."""
        self.free_lists.clear()
        self.allocated_blocks.clear()
        if hasattr(self, "base_tensor"):
            del self.base_tensor

    def __del__(self):
        self.cleanup()
