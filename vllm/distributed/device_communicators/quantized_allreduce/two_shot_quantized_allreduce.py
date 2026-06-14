# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Code based on the Kraken project, Copyright (c) Meta Platforms, Inc. (BSD-3-Clause)
"""
Two-shot all-reduce with per-group quantization (int8 or fp8).

Performs all-reduce by quantizing activations before communication,
reducing NVLink bandwidth by 2x compared to bf16 all-reduce.

Uses a single packed symmetric memory buffer containing both per-group
scales and quantized data. The algorithm has three phases:

  Phase 1: Quantize input per group, write to symmetric memory buffer.
  Phase 2: Each rank reduces its assigned stripe by reading quantized
           data from all peers, dequantizing, summing, re-quantizing,
           and writing the result back. BF16 output for the rank's own
           stripe is written directly (fused dequant).
  Phase 3: Each rank dequantizes the remaining stripes from the local
           buffer to produce the full output.

When USE_P2P is enabled (determined by tuning config), Phase 2 writes
only to the reducer's own buffer, and Phase 3 uses P2P reads from each
reducer instead of reading from the local buffer.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from vllm.triton_utils import tl, triton

from . import symm_mem_barrier as ptx_utils
from .config_loading import load_config as _load_config_from_json

MAX_BLOCK_SIZE = 16384
DEFAULT_GROUP_SIZE = 256


@triton.jit
def _two_shot_quantized_allreduce_kernel(
    buf_ptrs_dev,
    signal_pad_ptrs,
    input_ptr,
    output_ptr,
    numel,
    num_groups_total: tl.constexpr,
    data_offset: tl.constexpr,
    stride_per_program: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    QMAX: tl.constexpr,
    USE_FP8: tl.constexpr,
    USE_P2P: tl.constexpr = False,
):
    pid = tl.program_id(0)
    input_ptr = tl.multiple_of(input_ptr, 16)
    output_ptr = tl.multiple_of(output_ptr, 16)
    ptrs = buf_ptrs_dev.to(tl.pointer_type(tl.uint64))
    my_buf = tl.load(ptrs + rank).to(tl.pointer_type(tl.uint8))

    QTYPE = tl.float8e4nv if USE_FP8 else tl.int8

    my_scales = my_buf.to(tl.pointer_type(tl.float32))
    my_data = (my_buf + data_offset).to(tl.pointer_type(QTYPE))
    my_data = tl.multiple_of(my_data, 16)

    group_ids = tl.arange(0, GROUPS_PER_BLOCK)
    elem_ids = tl.arange(0, GROUP_SIZE)

    # --- Phase 1: Per-group quantize, write to packed buffer ---
    num_compute_blocks = tl.cdiv(numel, BLOCK_SIZE)
    blk_id = pid
    while blk_id < num_compute_blocks:
        blk_off = blk_id * BLOCK_SIZE
        first_group = blk_id * GROUPS_PER_BLOCK

        offsets_2d = blk_off + group_ids[:, None] * GROUP_SIZE + elem_ids[None, :]
        mask_2d = offsets_2d < numel

        x = tl.load(input_ptr + offsets_2d, mask=mask_2d, other=0.0).to(tl.float32)
        grp_amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12)
        grp_scales = QMAX / grp_amax

        scale_mask = (first_group + group_ids) < num_groups_total
        tl.store(my_scales + first_group + group_ids, grp_scales, mask=scale_mask)

        x_scaled = tl.clamp(x * grp_scales[:, None], -QMAX, QMAX)

        offsets_1d = blk_off + tl.arange(0, BLOCK_SIZE)
        mask_1d = offsets_1d < numel
        tl.store(
            my_data + offsets_1d,
            tl.reshape(x_scaled, [BLOCK_SIZE]).to(QTYPE),
            mask=mask_1d,
        )

        blk_id += tl.num_programs(0)

    # Barrier 1
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # --- Phase 2: Reduce own stripe, re-quantize, write to peers + BF16 own output ---
    block_start = pid * stride_per_program
    while block_start < numel:
        stripe_off = block_start + rank * BLOCK_SIZE
        first_group = stripe_off // GROUP_SIZE

        offsets = stripe_off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel

        acc_2d = tl.zeros([GROUPS_PER_BLOCK, GROUP_SIZE], dtype=tl.float32)

        for i in tl.static_range(world_size):
            peer = tl.load(ptrs + i).to(tl.pointer_type(tl.uint8))
            peer_scales = tl.load(
                peer.to(tl.pointer_type(tl.float32)) + first_group + group_ids,
                mask=(first_group + group_ids) < num_groups_total,
                other=1.0,
            )
            peer_data = (peer + data_offset).to(tl.pointer_type(QTYPE))
            peer_data = tl.multiple_of(peer_data, 16)
            qvals = tl.load(peer_data + offsets, mask=mask, other=0.0)
            vals_2d = tl.reshape(qvals.to(tl.float32), [GROUPS_PER_BLOCK, GROUP_SIZE])
            acc_2d += vals_2d / peer_scales[:, None]

        # Re-quantize
        out_amax = tl.maximum(tl.max(tl.abs(acc_2d), axis=1), 1e-12)
        out_scales = QMAX / out_amax
        result_2d = tl.clamp(acc_2d * out_scales[:, None], -QMAX, QMAX)
        result_q = tl.reshape(result_2d, [BLOCK_SIZE]).to(QTYPE)

        if USE_P2P:
            tl.store(
                my_scales + num_groups_total + first_group + group_ids,
                out_scales,
                mask=(first_group + group_ids) < num_groups_total,
            )
            tl.store(my_data + offsets, result_q, mask=mask)
        else:
            for i in tl.static_range(world_size):
                peer = tl.load(ptrs + i).to(tl.pointer_type(tl.uint8))
                tl.store(
                    peer.to(tl.pointer_type(tl.float32))
                    + num_groups_total
                    + first_group
                    + group_ids,
                    out_scales,
                    mask=(first_group + group_ids) < num_groups_total,
                )
                peer_data = (peer + data_offset).to(tl.pointer_type(QTYPE))
                peer_data = tl.multiple_of(peer_data, 16)
                tl.store(peer_data + offsets, result_q, mask=mask)

        # Fused: write BF16 directly to own output for own stripe
        result_bf16 = tl.reshape(acc_2d, [BLOCK_SIZE]).to(tl.bfloat16)
        tl.store(output_ptr + offsets, result_bf16, mask=mask)

        block_start += tl.num_programs(0) * stride_per_program

    # Barrier 2
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # --- Phase 3: Dequant OTHER ranks' stripes only ---
    if USE_P2P:
        block_start = pid * stride_per_program
        while block_start < numel:
            for r in tl.static_range(world_size):
                if r != rank:
                    stripe_off = block_start + r * BLOCK_SIZE
                    first_group = stripe_off // GROUP_SIZE
                    offsets = stripe_off + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < numel

                    reducer = tl.load(ptrs + r).to(tl.pointer_type(tl.uint8))
                    reducer_data = (reducer + data_offset).to(tl.pointer_type(QTYPE))
                    reducer_data = tl.multiple_of(reducer_data, 16)
                    qvals = tl.load(reducer_data + offsets, mask=mask, other=0.0)
                    vals_2d = tl.reshape(
                        qvals.to(tl.float32), [GROUPS_PER_BLOCK, GROUP_SIZE]
                    )

                    r_out_scales = tl.load(
                        reducer.to(tl.pointer_type(tl.float32))
                        + num_groups_total
                        + first_group
                        + group_ids,
                        mask=(first_group + group_ids) < num_groups_total,
                        other=1.0,
                    )
                    result_2d = vals_2d / r_out_scales[:, None]
                    tl.store(
                        output_ptr + offsets,
                        tl.reshape(result_2d, [BLOCK_SIZE]).to(tl.bfloat16),
                        mask=mask,
                    )
            block_start += tl.num_programs(0) * stride_per_program
    else:
        block_start = pid * stride_per_program
        while block_start < numel:
            for r in tl.static_range(world_size):
                if r != rank:
                    stripe_off = block_start + r * BLOCK_SIZE
                    first_group = stripe_off // GROUP_SIZE
                    offsets = stripe_off + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < numel

                    qvals = tl.load(my_data + offsets, mask=mask, other=0.0)
                    vals_2d = tl.reshape(
                        qvals.to(tl.float32), [GROUPS_PER_BLOCK, GROUP_SIZE]
                    )

                    r_out_scales = tl.load(
                        my_scales + num_groups_total + first_group + group_ids,
                        mask=(first_group + group_ids) < num_groups_total,
                        other=1.0,
                    )
                    result_2d = vals_2d / r_out_scales[:, None]
                    tl.store(
                        output_ptr + offsets,
                        tl.reshape(result_2d, [BLOCK_SIZE]).to(tl.bfloat16),
                        mask=mask,
                    )
            block_start += tl.num_programs(0) * stride_per_program


def compute_layout(numel, group_size, max_block_size=MAX_BLOCK_SIZE):
    """Compute byte offsets and total size for the packed symmetric memory buffer."""
    num_groups_total = triton.cdiv(numel, group_size)
    num_out_groups = num_groups_total + max_block_size // group_size
    num_scale_slots = num_groups_total + num_out_groups
    scale_header = num_scale_slots * 4  # float32 scales
    data_offset = ((scale_header + 15) // 16) * 16
    packed_size = data_offset + numel  # 1 byte per quant element
    return num_groups_total, data_offset, packed_size


_cache = {}


class _State:
    __slots__ = [
        "buf",
        "hdl",
        "packed_size",
        "data_offset",
        "num_groups_total",
        "max_numel",
        "group_size",
        "world_size",
    ]

    def __init__(self, numel, group_size, device, group=None):
        if group is None:
            group = dist.group.WORLD
        self.world_size = dist.get_world_size(group)
        self.max_numel = numel
        self.group_size = group_size
        self.num_groups_total, self.data_offset, self.packed_size = compute_layout(
            numel, group_size
        )
        self.buf = symm_mem.empty((self.packed_size,), dtype=torch.uint8, device=device)
        self.hdl = symm_mem.rendezvous(self.buf, group=group)

    def get_metadata(self, numel):
        if numel == self.max_numel:
            return self.num_groups_total, self.data_offset
        return compute_layout(numel, self.group_size)[:2]


def _get_state(numel, group_size, device):
    key = (numel, group_size, device)
    if key not in _cache:
        _cache[key] = _State(numel, group_size, device)
    return _cache[key]


def _get_tuned_config(numel, ws, use_fp8=False, group_size=DEFAULT_GROUP_SIZE):
    """Return (block_size, num_warps) for the given tensor size and world size."""
    kernel = "fp8" if use_fp8 else "int8"
    return _load_config_from_json(numel, ws, kernel=kernel, group_size=group_size)


def two_shot_quantized_allreduce(
    input_tensor: torch.Tensor,
    output: torch.Tensor | None = None,
    max_num_blocks: int = 132,
    block_size: int | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    use_fp8: bool = False,
    num_warps: int | None = None,
    state: "_State | None" = None,
    use_p2p: bool | None = None,
) -> torch.Tensor:
    """
    Perform quantized two-shot all-reduce on the input tensor.

    Args:
        input_tensor: BF16 input tensor to all-reduce.
        output: Optional pre-allocated output tensor.
        max_num_blocks: Maximum number of persistent thread blocks.
        block_size: Elements per thread block. Loaded from config if None.
        group_size: Number of elements per quantization group.
        use_fp8: If True, use FP8 E4M3 quantization instead of int8.
        num_warps: Warps per thread block. Loaded from config if None.
        state: Pre-allocated buffer state. If None, one is created
            using dist.group.WORLD as the process group.
        use_p2p: If True, use peer-to-peer writes in the final phase
            instead of broadcasting through shared memory. Faster for
            large tensors (typically >=16M elements) where the reduced
            write traffic outweighs the P2P overhead. If None, the
            value is loaded from the tuned config.

    Returns:
        BF16 tensor with the all-reduced result.
    """
    assert input_tensor.dtype == torch.bfloat16
    assert input_tensor.is_contiguous()
    numel = input_tensor.numel()
    assert numel % 8 == 0

    if state is None:
        state = _get_state(numel, group_size, input_tensor.device)
    ws = state.world_size

    tuned_bs, tuned_nw, tuned_p2p = _get_tuned_config(
        numel, ws, use_fp8=use_fp8, group_size=group_size
    )
    if block_size is None:
        block_size = tuned_bs
    if num_warps is None:
        num_warps = tuned_nw
    if use_p2p is None:
        use_p2p = tuned_p2p

    assert block_size % group_size == 0
    assert block_size % ws == 0
    assert block_size * ws <= numel
    assert block_size <= MAX_BLOCK_SIZE, (
        f"block_size {block_size} exceeds MAX_BLOCK_SIZE {MAX_BLOCK_SIZE}"
    )

    groups_per_block = block_size // group_size
    QMAX = 448.0 if use_fp8 else 127.0

    ngt, doff = state.get_metadata(numel)
    stride = block_size * ws
    num_blocks = min(triton.cdiv(numel, stride), max_num_blocks)

    if output is None:
        output = torch.empty_like(input_tensor)
    else:
        assert output.is_contiguous()

    _two_shot_quantized_allreduce_kernel[(num_blocks,)](
        state.hdl.buffer_ptrs_dev,
        state.hdl.signal_pad_ptrs_dev,
        input_tensor,
        output,
        numel=numel,
        num_groups_total=ngt,
        data_offset=doff,
        stride_per_program=stride,
        rank=state.hdl.rank,
        world_size=ws,
        BLOCK_SIZE=block_size,
        GROUP_SIZE=group_size,
        GROUPS_PER_BLOCK=groups_per_block,
        QMAX=QMAX,
        USE_FP8=use_fp8,
        USE_P2P=use_p2p,
        num_warps=num_warps,
    )

    return output
