# SPDX-License-Identifier: Apache-2.0

import atexit
import ctypes
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypeVar

import tensor_store_load_mem as cuda_kernels
import torch

# Define type variable
T = TypeVar('T')


@dataclass
class MemoryBlock:
    size: int
    addr: int
    is_free: bool = True
    tensor: Optional[torch.Tensor] = None
    buddy: Optional['MemoryBlock'] = None
    prev: Optional['MemoryBlock'] = None
    next: Optional['MemoryBlock'] = None


class PinnedMemoryPool:

    def __init__(self, max_block_size: int, min_block_size: int = 512):
        """
        Initialize pinned memory pool
        :param max_block_size: Maximum block size (bytes)
        :param min_block_size: Minimum block size (bytes), default is 512 bytes
        """
        if max_block_size <= 0 or min_block_size <= 0:
            raise ValueError("Block sizes must be positive")
        if max_block_size < min_block_size:
            raise ValueError(
                "Max block size must be greater than min block size")

        # Ensure block sizes are powers of two
        self.max_block_size = self._round_to_power_of_two(max_block_size)
        self.min_block_size = self._round_to_power_of_two(min_block_size)

        # Initialize buddy system free lists
        self.free_lists: Dict[int, List[MemoryBlock]] = {}
        self.allocated_blocks: Dict[int, MemoryBlock] = {
        }  # Address to block mapping

        # Create free lists for largest blocks
        self._initialize_free_lists()

        # Allocate actual pinned memory
        self._allocate_pinned_memory()

        # Register cleanup function
        atexit.register(self.cleanup)

    def _round_to_power_of_two(self, size: int) -> int:
        """Round size to nearest power of two"""
        return 1 << (size - 1).bit_length()

    def _initialize_free_lists(self):
        """Initialize free lists"""
        size = self.max_block_size
        while size >= self.min_block_size:
            self.free_lists[size] = []
            size = size // 2

    def _allocate_pinned_memory(self):
        """Allocate pinned memory"""
        # Use PyTorch to allocate pinned memory
        self.base_tensor = torch.empty(self.max_block_size // 4,
                                       dtype=torch.float32,
                                       pin_memory=True)

        # Get raw pointer address
        self.base_address = self.base_tensor.data_ptr()

        # Create largest memory block
        initial_block = MemoryBlock(size=self.max_block_size,
                                    addr=self.base_address)
        self.free_lists[self.max_block_size].append(initial_block)

    def allocate(self, size: int) -> int:
        """
        Allocate memory
        :param size: Required size (bytes)
        :return: Allocated memory address
        """
        if size <= 0:
            raise ValueError("Allocation size must be positive")

        # Calculate minimum required block size
        required_size = self._round_to_power_of_two(
            max(size, self.min_block_size))

        # Check if we have a large enough block
        if required_size > self.max_block_size:
            raise MemoryError("Requested size exceeds maximum block size")

        # Find suitable block in free lists
        current_size = required_size
        while current_size <= self.max_block_size:
            if self.free_lists[current_size]:
                # Found suitable block
                block = self.free_lists[current_size].pop()
                self._split_block(block, required_size)
                block.is_free = False
                self.allocated_blocks[block.addr] = block
                return block.addr
            current_size *= 2

        # No suitable block found
        raise MemoryError("Insufficient memory")

    def _split_block(self, block: MemoryBlock, required_size: int):
        """
        Split memory block until reaching required size
        """
        while (block.size > required_size
               and block.size // 2 >= self.min_block_size):
            # Create buddy block
            buddy_size = block.size // 2
            buddy_addr = block.addr + buddy_size

            buddy = MemoryBlock(size=buddy_size, addr=buddy_addr)
            block.size = buddy_size

            # Set buddy relationship
            block.buddy = buddy
            buddy.buddy = block

            # Add buddy to free list
            self.free_lists[buddy_size].append(buddy)

    def free(self, addr: int):
        """
        Free memory
        :param addr: Memory address to free
        """
        if addr not in self.allocated_blocks:
            raise ValueError("Invalid address to free")

        block = self.allocated_blocks.pop(addr)
        block.is_free = True

        # Try to merge buddy blocks
        self._merge_buddies(block)

    def _merge_buddies(self, block: MemoryBlock):
        """
        Attempt to merge buddy blocks
        """
        while block.buddy and block.buddy.is_free:
            # Get buddy
            buddy = block.buddy

            # Remove buddy from free list
            self.free_lists[buddy.size].remove(buddy)

            # Determine merged block address (take smaller address of the two)
            merged_addr = min(block.addr, buddy.addr)
            merged_size = block.size * 2

            # Create merged block
            merged_block = MemoryBlock(size=merged_size, addr=merged_addr)

            # Set new block's buddy relationship
            if merged_block.size < self.max_block_size:
                # Find new block's buddy
                buddy_offset = merged_size if merged_addr % (
                    2 * merged_size) == 0 else -merged_size
                buddy_addr = merged_addr + buddy_offset

                # Look for potential buddy in free lists
                for existing_block in self.free_lists[merged_size]:
                    if existing_block.addr == buddy_addr:
                        merged_block.buddy = existing_block
                        existing_block.buddy = merged_block
                        break

            # Add merged block to free list
            self.free_lists[merged_size].append(merged_block)

            # Update current block to merged block
            block = merged_block

    def store_tensor(self, tensor: torch.Tensor) -> int:
        """
        Store Tensor in memory pool
        :param tensor: CUDA Tensor to store
        :return: Stored memory address
        """
        if not tensor.is_cuda:
            raise ValueError("Only CUDA tensors can be stored")

        # Calculate required size (bytes)
        size = tensor.element_size() * tensor.numel()

        # Allocate memory
        addr = self.allocate(size)

        # Get block
        block = self.allocated_blocks[addr]

        # Create pinned CPU Tensor view
        dtype_size = tensor.element_size()
        num_elements = size // dtype_size
        cpu_tensor = torch.frombuffer(ctypes.cast(
            block.addr, ctypes.POINTER(ctypes.c_byte)),
                                      count=num_elements,
                                      dtype=tensor.dtype)

        # Asynchronously copy data to pinned memory
        with torch.cuda.stream(torch.cuda.Stream()):
            cpu_tensor.copy_(tensor, non_blocking=True)

        # Save Tensor metadata
        block.tensor = tensor

        return addr

    def load_tensor(self, addr: int, dtype: torch.dtype,
                    shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Load Tensor from memory pool
        :param addr: Stored memory address
        :param dtype: Tensor data type
        :param shape: Tensor shape
        :return: Recovered CUDA Tensor
        """
        if addr not in self.allocated_blocks:
            raise ValueError("Invalid address to load")

        block = self.allocated_blocks[addr]

        # Calculate element size and count
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = math.prod(shape)
        required_size = dtype_size * num_elements

        if required_size > block.size:
            raise ValueError("Requested tensor size exceeds block size")

        # Create CUDA Tensor
        cuda_tensor = torch.empty(shape, dtype=dtype, device='cuda')

        # Create pinned CPU Tensor view
        cpu_tensor = torch.frombuffer(ctypes.cast(
            block.addr, ctypes.POINTER(ctypes.c_byte)),
                                      count=num_elements,
                                      dtype=dtype)

        # Asynchronously copy data to CUDA
        with torch.cuda.stream(torch.cuda.Stream()):
            cuda_tensor.copy_(cpu_tensor[:num_elements], non_blocking=True)

        return cuda_tensor

    def cleanup(self):
        """Clean up all resources"""
        self.free_lists.clear()
        self.allocated_blocks.clear()
        if hasattr(self, 'base_tensor'):
            del self.base_tensor

    def __del__(self):
        self.cleanup()


class CudaPinnedMemoryPool(PinnedMemoryPool):

    def __init__(self, max_block_size: int, min_block_size: int = 512):
        super().__init__(max_block_size, min_block_size)

    def store_tensor(self, tensor: torch.Tensor) -> int:
        """Store Tensor using CUDA kernel"""
        if not tensor.is_cuda:
            raise ValueError("Only CUDA tensors can be stored")

        # Calculate required size (bytes)
        size = tensor.element_size() * tensor.numel()

        # Allocate memory (ensure enough space)
        addr = self.allocate(size)
        block = self.allocated_blocks[addr]

        # Verify allocated size is sufficient
        if block.size < size:
            self.free(addr)
            raise MemoryError(
                f"Allocated block size {block.size} is smaller than "
                f"required size {size}")

        # Create pinned CPU Tensor view
        try:
            # Use ctypes to create correctly sized buffer
            buffer = (ctypes.c_byte * block.size).from_address(block.addr)
            cpu_tensor = torch.frombuffer(buffer,
                                          dtype=tensor.dtype,
                                          count=tensor.numel())
        except ValueError as e:
            self.free(addr)
            raise MemoryError(f"Failed to create tensor view: {e}") from e

        # Use CUDA kernel to copy data
        cuda_kernels.store_tensor(tensor, cpu_tensor)

        # Synchronize to ensure copy completes
        torch.cuda.synchronize()

        block.tensor = tensor
        return addr

    def load_tensor(self, addr: int, dtype: torch.dtype,
                    shape: Tuple[int, ...]) -> torch.Tensor:
        """Load Tensor using CUDA kernel"""
        if addr not in self.allocated_blocks:
            raise ValueError("Invalid address to load")

        block = self.allocated_blocks[addr]
        num_elements = math.prod(shape)
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        required_size = num_elements * dtype_size

        if required_size > block.size:
            raise ValueError("Requested tensor size exceeds block size")

        # Create CUDA Tensor
        cuda_tensor = torch.empty(shape, dtype=dtype, device='cuda')

        # Create pinned CPU Tensor view
        buffer = (ctypes.c_byte * block.size).from_address(block.addr)
        cpu_tensor = torch.frombuffer(buffer, dtype=dtype, count=num_elements)

        # Use CUDA kernel to copy data
        cuda_kernels.load_tensor(cpu_tensor, cuda_tensor)

        # Synchronize to ensure copy completes
        torch.cuda.synchronize()

        return cuda_tensor
