# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU kernels for efficient GPU<->CPU block transfers."""

from typing import NamedTuple

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Limits SM occupancy so compute kernels can run concurrently.
DEFAULT_COPY_NUM_SMS = 16


class LaunchParams(NamedTuple):
    """Pre-computed launch parameters for copy_blocks."""

    src_ptr_table: torch.Tensor
    dst_ptr_table: torch.Tensor
    wpb_table: torch.Tensor  # [num_layers] int64 words-per-block per layer
    num_layers: int
    max_words_per_block: int  # max across layers, for kernel config
    block_size: int
    num_warps: int
    num_sms: int


def _compute_launch_params(
    words_per_block: int,
) -> tuple[int, int]:
    """Compute Triton launch parameters for block copy kernels."""
    block_size = min(triton.next_power_of_2(words_per_block), 1024)
    num_warps = min(max(block_size // 32, 1), 32)
    return block_size, num_warps


@triton.jit
def _copy_blocks_kernel(
    src_ptrs,
    dst_ptrs,
    wpb_ptr,
    mapping_ptr,
    total_jobs,  # type: ignore[name-defined]
    num_pairs,  # type: ignore[name-defined]
    max_words_per_block: tl.constexpr,  # type: ignore[name-defined]
    BLOCK_SIZE: tl.constexpr,  # type: ignore[name-defined]
):
    """
    Kernel for copying blocks across multiple layers.

    Uses a grid-stride loop so the kernel can be launched with a small
    grid (e.g., 32 CTAs) to limit SM occupancy, allowing compute kernels
    to run concurrently on remaining SMs.

    Grid: (num_sms,)

    Args:
        src_ptrs: Pointer to uint64 tensor [num_layers] of source base addrs
        dst_ptrs: Pointer to uint64 tensor [num_layers] of dest base addrs
        wpb_ptr: Pointer to int64 tensor [num_layers] of words-per-block
        mapping_ptr: Pointer to int64 tensor [N * 2] of (src_id, dst_id) pairs
        total_jobs: num_pairs * num_layers
        num_pairs: Number of (src, dst) block pairs
        max_words_per_block: Max words-per-block across layers (loop bound)
        BLOCK_SIZE: Triton block size for vectorization
    """
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    # Grid-stride loop: each CTA processes multiple (pair, layer) jobs
    job_id = pid
    while job_id < total_jobs:
        pair_id = job_id % num_pairs
        layer_id = job_id // num_pairs

        # Load source and destination block IDs from mapping
        src_block = tl.load(mapping_ptr + pair_id * 2).to(tl.int64)
        dst_block = tl.load(mapping_ptr + pair_id * 2 + 1).to(tl.int64)

        # Load base addresses from pointer tables and cast to typed pointers
        src_base = tl.load(src_ptrs + layer_id).to(tl.pointer_type(tl.int64))
        dst_base = tl.load(dst_ptrs + layer_id).to(tl.pointer_type(tl.int64))

        # Per-layer words_per_block (supports varying page sizes)
        wpb = tl.load(wpb_ptr + layer_id)

        # Compute offsets using stride-based addressing
        src_off = src_block * wpb
        dst_off = dst_block * wpb

        # Copy in chunks of BLOCK_SIZE, masked by this layer's wpb
        offsets = tl.arange(0, BLOCK_SIZE)
        for start in range(0, max_words_per_block, BLOCK_SIZE):
            idx = start + offsets
            mask = idx < wpb
            data = tl.load(src_base + src_off + idx, mask=mask, other=0)
            tl.store(dst_base + dst_off + idx, data, mask=mask)

        job_id += num_ctas


def build_launch_params(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    num_sms: int = DEFAULT_COPY_NUM_SMS,
) -> LaunchParams:
    """
    Pre-compute launch parameters for copy_blocks.

    Call once at init time and pass the results to copy_blocks
    to avoid per-call overhead of pointer table construction.

    Args:
        src_caches: Dict mapping layer name to source KV cache tensor.
        dst_caches: Dict mapping layer name to destination KV cache tensor.
        num_sms: Number of SMs (CTAs) to use for the persistent kernel.
            Limits SM occupancy so compute kernels can run concurrently.

    Returns:
        LaunchParams with pointer tables, layer count, and
        kernel configuration.
    """
    assert list(src_caches.keys()) == list(dst_caches.keys()), (
        "src and dst cache dicts must have matching keys in the same order"
    )

    src_tensors = list(src_caches.values())
    dst_tensors = list(dst_caches.values())
    num_layers = len(src_tensors)

    # Build per-layer words_per_block table. Layers may have different
    # page sizes (e.g., UniformTypeKVCacheSpecs with varying head_size).
    wpb_list: list[int] = []
    for src_t, dst_t in zip(src_tensors, dst_tensors):
        src_wpb = src_t.stride(0) * src_t.element_size() // 8
        dst_wpb = dst_t.stride(0) * dst_t.element_size() // 8
        assert src_wpb == dst_wpb, (
            f"src/dst stride mismatch for layer: {src_wpb} vs {dst_wpb}"
        )
        wpb_list.append(src_wpb)

    max_wpb = max(wpb_list)
    wpb_table = torch.tensor(wpb_list, device="cuda", dtype=torch.int64)

    src_ptr_table = torch.tensor(
        [t.data_ptr() for t in src_tensors], device="cuda", dtype=torch.uint64
    )
    dst_ptr_table = torch.tensor(
        [t.data_ptr() for t in dst_tensors], device="cuda", dtype=torch.uint64
    )

    block_size, num_warps = _compute_launch_params(max_wpb)
    return LaunchParams(
        src_ptr_table=src_ptr_table,
        dst_ptr_table=dst_ptr_table,
        wpb_table=wpb_table,
        num_layers=num_layers,
        max_words_per_block=max_wpb,
        block_size=block_size,
        num_warps=num_warps,
        num_sms=num_sms,
    )


def copy_blocks(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    block_mapping: torch.Tensor,
    *,
    launch_params: LaunchParams | None = None,
) -> None:
    """
    Copy blocks across all layers in a single Triton kernel launch.

    Uses pointer tables and stride-based addressing to handle non-contiguous
    (permuted) tensors without requiring .view(-1).

    Args:
        src_caches: Dict mapping layer name to source KV cache tensor
        dst_caches: Dict mapping layer name to destination KV cache tensor
        block_mapping: [N, 2] tensor of (src_block_id, dst_block_id) pairs
        launch_params: Pre-computed launch parameters from
            build_launch_params(). If None, computed on the fly.
    """
    if block_mapping.numel() == 0:
        return

    if launch_params is None:
        launch_params = build_launch_params(src_caches, dst_caches)

    # Flatten mapping (already int64 contiguous from caller)
    mapping_flat = block_mapping.view(-1)

    num_pairs = block_mapping.shape[0]
    total_jobs = num_pairs * launch_params.num_layers
    grid_size = min(launch_params.num_sms, total_jobs)

    _copy_blocks_kernel[(grid_size,)](
        launch_params.src_ptr_table,
        launch_params.dst_ptr_table,
        launch_params.wpb_table,
        mapping_flat,
        total_jobs,
        num_pairs,
        launch_params.max_words_per_block,
        BLOCK_SIZE=launch_params.block_size,
        num_warps=launch_params.num_warps,
    )
