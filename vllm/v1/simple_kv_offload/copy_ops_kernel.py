# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU kernels for efficient GPU<->CPU block transfers."""

from typing import NamedTuple

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
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
    max_v2_groups: int  # max (wpb // 2) across layers, for kernel config
    block_size: int
    num_warps: int
    num_sms: int
    use_l2_256b: bool  # True for Blackwell (sm_100+), False for Hopper


def _compute_launch_params(
    max_v2_groups: int,
) -> tuple[int, int]:
    """Compute Triton launch parameters for block copy kernels.

    Args:
        max_v2_groups: Max number of v2 groups (= wpb // 2) across layers.
            Each v2 group is 2 int64 words = 16 bytes.
    """
    block_size = min(triton.next_power_of_2(max_v2_groups), 1024)
    num_warps = min(max(block_size // 32, 1), 32)
    return block_size, num_warps


# --- Non-caching v2.b64 load/store helpers (L2::128B for Hopper) ---


@triton.jit
def _load_v2_nc_128b(ptr):
    """NC v2 int64 load with L2::128B (Hopper)."""
    a, b = tl.inline_asm_elementwise(
        asm="ld.global.nc.L1::no_allocate.L2::128B.v2.b64 {$0,$1}, [$2];",
        constraints="=l,=l,l",
        args=[ptr],
        dtype=(tl.int64, tl.int64),
        is_pure=True,
        pack=1,
    )
    return a, b


@triton.jit
def _store_v2_nc(ptr, a, b):
    """NC v2 int64 store (bypasses L1, default L2 policy).

    Note: L2:: sector hints are load-only in PTX; stores use
    L1::no_allocate without an L2 modifier.
    """
    tl.inline_asm_elementwise(
        asm="st.global.L1::no_allocate.v2.b64 [$1], {$2,$3};",
        constraints="=l,l,l,l",
        args=[ptr, a, b],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


# --- Non-caching v2.b64 load helpers (L2::256B for Blackwell) ---


@triton.jit
def _load_v2_nc_256b(ptr):
    """NC v2 int64 load with L2::256B (Blackwell)."""
    a, b = tl.inline_asm_elementwise(
        asm="ld.global.nc.L1::no_allocate.L2::256B.v2.b64 {$0,$1}, [$2];",
        constraints="=l,=l,l",
        args=[ptr],
        dtype=(tl.int64, tl.int64),
        is_pure=True,
        pack=1,
    )
    return a, b


@triton.jit
def _copy_blocks_kernel(
    src_ptrs,
    dst_ptrs,
    wpb_ptr,
    mapping_ptr,
    total_jobs,  # type: ignore[name-defined]
    num_pairs,  # type: ignore[name-defined]
    MAX_V2_GROUPS: tl.constexpr,  # type: ignore[name-defined]
    BLOCK_SIZE: tl.constexpr,  # type: ignore[name-defined]
    USE_L2_256B: tl.constexpr,  # type: ignore[name-defined]
):
    """
    Kernel for copying blocks across multiple layers using non-caching
    v2.b64 loads/stores to avoid L1/L2 cache pollution.

    Each v2 group = 2 × int64 = 16 bytes. Loads use the texture/read-only
    path (.nc) to bypass L1 entirely; L2 sector size is selected by
    USE_L2_256B (256B for Blackwell sm_100+, 128B for Hopper sm_90).

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
        MAX_V2_GROUPS: Max v2 groups across layers (loop bound)
        BLOCK_SIZE: Number of v2 groups processed per iteration
        USE_L2_256B: True for 256B L2 sectors (Blackwell), False for 128B
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

        # Per-layer words_per_block in int64 words
        wpb = tl.load(wpb_ptr + layer_id)
        num_v2 = wpb // 2  # each v2 group = 2 int64 words

        # Compute offsets using stride-based addressing (in int64 words)
        src_off = src_block * wpb
        dst_off = dst_block * wpb

        # Copy in chunks of BLOCK_SIZE v2 groups using non-caching ops.
        # OOB threads (g >= num_v2) have pointers clamped to group 0
        # of the same block: they load src[0:2] and store to dst[0:2],
        # duplicating the work of thread 0 (idempotent, no data corruption).
        offsets = tl.arange(0, BLOCK_SIZE)
        for start in range(0, MAX_V2_GROUPS, BLOCK_SIZE):
            g = start + offsets
            mask = g < num_v2
            ptr_off = g * 2  # 2 int64 per v2 group

            src_ptr = src_base + src_off + ptr_off
            dst_ptr = dst_base + dst_off + ptr_off

            # Clamp OOB pointers to group 0 (same src/dst data → idempotent)
            safe_src = tl.where(mask, src_ptr, src_base + src_off)
            safe_dst = tl.where(mask, dst_ptr, dst_base + dst_off)

            # Dispatch to arch-matched L2 sector size for loads (constexpr
            # branch, dead code eliminated at compile time). Stores use a
            # single variant — L2:: hints are load-only in PTX.
            if USE_L2_256B:
                a, b = _load_v2_nc_256b(safe_src)
            else:
                a, b = _load_v2_nc_128b(safe_src)
            _store_v2_nc(safe_dst, a, b)

        job_id += num_ctas


def build_params(
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
    assert all(w % 2 == 0 for w in wpb_list), (
        f"All words_per_block must be even for v2 operations: {wpb_list}"
    )
    max_v2_groups = max_wpb // 2

    wpb_table = torch.tensor(wpb_list, device="cuda", dtype=torch.int64)

    src_ptr_table = torch.tensor(
        [t.data_ptr() for t in src_tensors], device="cuda", dtype=torch.uint64
    )
    dst_ptr_table = torch.tensor(
        [t.data_ptr() for t in dst_tensors], device="cuda", dtype=torch.uint64
    )

    # Blackwell (sm_100+) has 256B L2 cache lines; Hopper and earlier use 128B.
    capability = current_platform.get_device_capability()
    use_l2_256b = capability is not None and capability.major >= 10

    block_size, num_warps = _compute_launch_params(max_v2_groups)
    return LaunchParams(
        src_ptr_table=src_ptr_table,
        dst_ptr_table=dst_ptr_table,
        wpb_table=wpb_table,
        num_layers=num_layers,
        max_v2_groups=max_v2_groups,
        block_size=block_size,
        num_warps=num_warps,
        num_sms=num_sms,
        use_l2_256b=use_l2_256b,
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
        launch_params: Pre-computed launch parameters from build_params(). If
            None, computed on the fly.
    """
    if block_mapping.numel() == 0:
        return

    if launch_params is None:
        launch_params = build_params(src_caches, dst_caches)

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
        launch_params.max_v2_groups,
        BLOCK_SIZE=launch_params.block_size,
        USE_L2_256B=launch_params.use_l2_256b,
        num_warps=launch_params.num_warps,
    )
