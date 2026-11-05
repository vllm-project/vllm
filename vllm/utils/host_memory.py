# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host memory registration for CUDA/HIP DMA transfers."""

import math
import mmap

import torch


def pin_tensor(tensor: torch.Tensor) -> None:
    """Register one CPU range, bypassing the caching host allocator's rounding."""
    err = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)
    if err.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {err}")


def unpin_tensor(tensor: torch.Tensor) -> None:
    """Unregister exactly the range passed to pin_tensor."""
    cudart = torch.cuda.cudart()
    err = cudart.cudaHostUnregister(tensor.data_ptr())
    if err.value != 0:
        cudart.cudaGetLastError()
        raise RuntimeError(f"cudaHostUnregister failed (code={err.value}).")


def host_registration_ranges(
    tensor_sizes: list[int],
    num_blocks: int,
    total_size: int,
    max_chunk_bytes: int = 1 << 30,
) -> tuple[tuple[int, int], ...]:
    """Split a shared pool without bisecting any tensor's host block.

    CUDA batch copies reject descriptors spanning adjacent registrations.
    """
    if total_size % mmap.PAGESIZE:
        raise ValueError("Host pool must be OS-page aligned.")
    if max_chunk_bytes < mmap.PAGESIZE:
        raise ValueError("Host registration chunks must span at least one page.")

    tensors: list[tuple[int, int, int]] = []
    tensor_start = 0
    for tensor_size in tensor_sizes:
        if tensor_size % num_blocks:
            raise ValueError(
                f"Host tensor size {tensor_size} is not divisible by "
                f"{num_blocks} blocks."
            )
        tensor_end = tensor_start + tensor_size
        tensors.append((tensor_start, tensor_end, tensor_size // num_blocks))
        tensor_start = tensor_end
    if tensor_start > total_size:
        raise ValueError("Host tensors exceed the shared host pool.")

    def last_safe_boundary(lower: int, upper: int) -> int | None:
        best = None
        for start, end, block_bytes in tensors:
            first_block = max(0, (lower - start) // block_bytes + 1)
            last_block = min(num_blocks, (min(upper, end) - start) // block_bytes)
            if first_block > last_block:
                continue
            divisor = math.gcd(block_bytes, mmap.PAGESIZE)
            if start % divisor:
                continue
            period = mmap.PAGESIZE // divisor
            if period == 1:
                residue = 0
            else:
                # Find block boundaries that are also OS-page boundaries.
                residue = (
                    (-start // divisor) * pow(block_bytes // divisor, -1, period)
                ) % period
            block = last_block - (last_block - residue) % period
            if block >= first_block:
                candidate = start + block * block_bytes
                best = candidate if best is None else max(best, candidate)

        if upper >= tensor_start:
            candidate = upper // mmap.PAGESIZE * mmap.PAGESIZE
            if candidate >= tensor_start and candidate > lower:
                best = candidate if best is None else max(best, candidate)
        return best

    ranges: list[tuple[int, int]] = []
    start = 0
    while total_size - start > max_chunk_bytes:
        end = last_safe_boundary(start, start + max_chunk_bytes)
        if end is None:
            raise ValueError(
                "A host block is too large for the registration chunk limit."
            )
        ranges.append((start, end))
        start = end
    ranges.append((start, total_size))
    return tuple(ranges)
