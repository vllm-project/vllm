# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host memory registration for CUDA/HIP DMA transfers, and core dump exclusion."""

import math
import mmap

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

HOST_REGISTER_CHUNK_BYTES = 256 << 30


def exclude_from_core_dump(mmap_obj: mmap.mmap) -> None:
    """Leave the mapping out of core dumps (MADV_DONTDUMP, Linux 3.4+).

    A CPU KV cache can be hundreds of GiB, so a crashing worker would
    otherwise write all of it to disk. Failure only costs a larger dump.
    """
    madv_dontdump = getattr(mmap, "MADV_DONTDUMP", None)
    if madv_dontdump is None:
        return
    try:
        mmap_obj.madvise(madv_dontdump)
    except OSError as e:
        logger.warning(
            "MADV_DONTDUMP failed; core dumps will include the CPU KV cache: %s", e
        )


def registration_pins_pages() -> bool:
    """Whether registering host memory with the GPU runtime pins its pages.

    `cudaHostRegister` pins them. `hipHostRegister` does not: amdgpu tracks the
    range as a userptr through an MMU notifier, so the kernel may still migrate
    its pages (memory compaction, NUMA balancing, `move_pages(2)`), and each
    time it does KFD stops every queue of the process until the whole
    registered range is valid again. That takes about 0.25-0.5 s per GiB of
    4 KiB pages (MI355X), so a large long-lived registration such as a CPU KV
    cache can stall the GPU for tens of seconds.
    """
    from vllm.platforms import current_platform

    return not current_platform.is_rocm()


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
    max_chunk_bytes: int = HOST_REGISTER_CHUNK_BYTES,
    *,
    page_size: int = mmap.PAGESIZE,
) -> tuple[tuple[int, int], ...]:
    """Split a shared pool without bisecting any tensor's host block.

    Keep registration boundaries aligned to the backing pages. CUDA batch
    copies reject descriptors spanning adjacent registrations.
    """
    if total_size % page_size:
        raise ValueError("Host pool must be page aligned.")
    if max_chunk_bytes < page_size:
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
            divisor = math.gcd(block_bytes, page_size)
            if start % divisor:
                continue
            period = page_size // divisor
            if period == 1:
                residue = 0
            else:
                # Find block boundaries that are also backing-page boundaries.
                residue = (
                    (-start // divisor) * pow(block_bytes // divisor, -1, period)
                ) % period
            block = last_block - (last_block - residue) % period
            if block >= first_block:
                candidate = start + block * block_bytes
                best = candidate if best is None else max(best, candidate)

        if upper >= tensor_start:
            candidate = upper // page_size * page_size
            if candidate >= tensor_start and candidate > lower:
                best = candidate if best is None else max(best, candidate)
        return best

    ranges: list[tuple[int, int]] = []
    start = 0
    while total_size - start > max_chunk_bytes:
        end = last_safe_boundary(start, start + max_chunk_bytes)
        if end is None:
            raise ValueError(
                "No block and page boundary fits the registration chunk limit."
            )
        ranges.append((start, end))
        start = end
    ranges.append((start, total_size))
    return tuple(ranges)
