# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dequantize only the experts a batch routes to, without a host sync.

Every shape depends only on `topk_ids`' shape, so both helpers can run under
CUDA graph capture.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_ids"])
def _rank_routed_experts_kernel(
    ids_ptr,
    used_ptr,
    rank_ptr,
    num_ids,
    num_experts,
    num_slots,
    BLOCK_E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    experts = tl.arange(0, BLOCK_E)
    counts = tl.zeros((BLOCK_E,), dtype=tl.int32)
    for start in range(0, num_ids, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        ids = tl.load(ids_ptr + offs, mask=offs < num_ids, other=-1)
        valid = (ids >= 0) & (ids < num_experts)
        ids = tl.where(valid, ids, 0).to(tl.int32)
        counts += tl.histogram(ids, BLOCK_E, mask=valid)
    present = counts > 0
    rank = tl.cumsum(present.to(tl.int32), axis=0) - 1
    num_used = tl.sum(present.to(tl.int32), axis=0)
    tl.store(used_ptr + rank, experts, mask=present)
    tl.store(used_ptr + experts, -1, mask=(experts >= num_used) & (experts < num_slots))
    tl.store(
        rank_ptr + experts, tl.where(present, rank, -1), mask=experts < num_experts
    )


@triton.jit(do_not_specialize=["num_ids"])
def _renumber_routed_experts_kernel(
    ids_ptr, rank_ptr, local_ids_ptr, num_ids, num_experts, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_ids
    ids = tl.load(ids_ptr + offs, mask=mask, other=-1)
    valid = (ids >= 0) & (ids < num_experts)
    local = tl.load(rank_ptr + tl.where(valid, ids, 0), mask=mask & valid, other=-1)
    tl.store(local_ids_ptr + offs, local, mask=mask)


def dedup_routed_experts(
    topk_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distinct routed experts in min(topk_ids.numel(), num_experts) slots.

    Returns:
        `used[s]`, the expert in slot s in ascending order or -1 for an unused
        slot, and topk_ids renumbered onto the slots, with -1 for ids outside
        [0, num_experts) so that the expert alignment skips them.

    """
    flat = topk_ids.reshape(-1).contiguous()
    num_ids = flat.numel()
    num_slots = min(num_ids, num_experts)
    used = torch.empty(num_slots, dtype=torch.int32, device=flat.device)
    rank = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
    local_ids = torch.empty_like(flat)
    # Two launches: the renumbering reads `rank` after all of it is written.
    _rank_routed_experts_kernel[(1,)](
        flat,
        used,
        rank,
        num_ids,
        num_experts,
        num_slots,
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_N=1024,
    )
    BLOCK = 1024
    _renumber_routed_experts_kernel[(triton.cdiv(num_ids, BLOCK),)](
        flat, rank, local_ids, num_ids, num_experts, BLOCK=BLOCK
    )
    return used, local_ids.view(topk_ids.shape)


@triton.jit
def _e2m1_to_f32(code):
    exp = (code >> 1) & 0x3
    mantissa = code & 0x1
    # e2m1 -> fp32 bits: rebias the exponent (1 -> 127); 0b0001 is 0.5.
    nonzero = (exp > 0) | (mantissa > 0)
    f32_exp = tl.where(nonzero, exp + 126, 0)
    f32_mantissa = tl.where(exp > 0, mantissa, 0)
    bits = ((code >> 3) << 31) | (f32_exp << 23) | (f32_mantissa << 22)
    return bits.to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["num_slots"])
def _dequant_routed_mxfp4_kernel(
    w_ptr,
    scale_ptr,
    used_ptr,
    out_ptr,
    w_expert_stride,
    scale_expert_stride,
    blocks_per_expert,
    num_slots,
    ROWS: tl.constexpr,
):
    # Each row is one 32-value block: 16 packed bytes under one e8m0 scale.
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    row_mask = rows < blocks_per_expert
    w_offs = rows[:, None] * 16 + tl.arange(0, 16)[None, :]
    out_offs = rows[:, None] * 32 + tl.arange(0, 32)[None, :]
    # Used slots come first, so stop at the first unused one.
    slot = tl.program_id(1)
    expert = tl.load(used_ptr + slot, mask=slot < num_slots, other=-1)
    while expert >= 0:
        scale = tl.load(
            scale_ptr + expert.to(tl.int64) * scale_expert_stride + rows,
            mask=row_mask,
        ).to(tl.int32)
        # 2**(scale - 127) in fp32: 0 is 2**-127 (subnormal), 255 is inf.
        scale_f = tl.where(scale == 0, 0x00400000, scale << 23).to(
            tl.float32, bitcast=True
        )[:, None]
        packed = tl.load(
            w_ptr + expert.to(tl.int64) * w_expert_stride + w_offs,
            mask=row_mask[:, None],
        ).to(tl.int32)
        out_lo = (_e2m1_to_f32(packed & 0xF) * scale_f).to(tl.bfloat16)
        out_hi = (_e2m1_to_f32(packed >> 4) * scale_f).to(tl.bfloat16)
        # Low nibble first, as in quark's dq_mxfp4.
        out = tl.reshape(tl.join(out_lo, out_hi), (ROWS, 32))
        tl.store(
            out_ptr + slot.to(tl.int64) * (32 * blocks_per_expert) + out_offs,
            out,
            mask=row_mask[:, None],
        )
        slot += tl.num_programs(1)
        expert = tl.load(used_ptr + slot, mask=slot < num_slots, other=-1)


def can_dequant_routed_mxfp4(w: torch.Tensor, scale: torch.Tensor) -> bool:
    return (
        w.dtype == scale.dtype == torch.uint8
        and w[0].is_contiguous()
        and scale[0].is_contiguous()
        and w.size(-1) % 16 == 0
        and scale[0].numel() * 16 == w[0].numel()
    )


def dequant_routed_mxfp4(
    w: torch.Tensor, scale: torch.Tensor, used: torch.Tensor
) -> torch.Tensor:
    """Dequantize the MXFP4 experts `used` selects into a slot-indexed buffer.

    Matches quark's default (HIP) `dq_mxfp4` bit for bit apart from NaN
    payloads. Unused (-1) slots are left uninitialized; the caller routes no
    tokens to them.

    Args:
        w: Packed weights [E, rows, cols // 2], uint8.
        scale: e8m0 scales [E, rows, cols // 32], uint8.
        used: Expert per slot [S], int32, -1 for an unused slot.

    Returns:
        bf16 weights [S, rows, cols].

    """
    num_slots, blocks_per_expert = used.numel(), w[0].numel() // 16
    out = torch.empty(
        (num_slots, *w.shape[1:-1], 2 * w.shape[-1]),
        dtype=torch.bfloat16,
        device=w.device,
    )
    ROWS = 256
    grid = (triton.cdiv(blocks_per_expert, ROWS), min(num_slots, 32))
    _dequant_routed_mxfp4_kernel[grid](
        w,
        scale,
        used,
        out,
        w.stride(0),
        scale.stride(0),
        blocks_per_expert,
        num_slots,
        ROWS=ROWS,
        num_warps=4,
    )
    return out
