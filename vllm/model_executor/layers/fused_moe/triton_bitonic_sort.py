# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.triton_utils import tl, triton

"""
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=shfl#restricted-use-of-sub-word-sizes
8-bit or 16-bit values may be held directly in 32-bit or 64-bit registers 
when being loaded, stored, or converted to other types and sizes.
"""


@triton.jit
def bitonic_ce_descending(
    val, idx, stride: tl.constexpr, log2_length_pair: tl.constexpr
):
    new_val, new_idx = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 %partner_val;
            .reg .s32 %partner_idx;
            .reg .u32 %lane_id;
            .reg .u32 %is_left;
            .reg .pred %p_left, %p_swap;
            .reg .u32 %group_id;
            .reg .pred %group_id_mask;
            
            // save input args to partner regs
            shfl.sync.bfly.b32 %partner_val, $2, $4, 0x1f, 0xffffffff;
            shfl.sync.bfly.b32 %partner_idx, $3, $4, 0x1f, 0xffffffff;
            
            mov.u32 %lane_id, %laneid;
            
            shr.u32 %group_id, %lane_id, $5;
            and.b32 %group_id, %group_id, 1;
            setp.eq.u32 %group_id_mask, %group_id, 1;
            
            and.b32 %is_left, %lane_id, $4;
            setp.eq.u32 %p_left, %is_left, 0;
            
            // compare partner_val > val? if so, swap.
            setp.gt.f32 %p_swap, %partner_val, $2;

            // TODO(ijpq): 
            // this logic might be redundant.
            // require simplify further.           
            xor.pred %p_swap, %p_swap, %p_left;
            not.pred %p_swap, %p_swap;
            xor.pred %p_swap, %p_swap, %group_id_mask;
            
            selp.f32 $0, %partner_val, $2, %p_swap;
            selp.b32 $1, %partner_idx, $3, %p_swap;
        }
        """,
        constraints="=f,=r,f,r,n,n",
        args=[val, idx, stride, log2_length_pair],
        dtype=(tl.float32, tl.int32),
        is_pure=True,
        pack=1,
    )
    return new_val, new_idx


@triton.jit
def bitonic_compare_across_part_descending(val, idx, stride: tl.constexpr):
    new_val, new_idx = tl.inline_asm_elementwise(
        asm="""{
            .reg .f32 %partner_val;
            .reg .s32 %partner_idx;
            .reg .u32 %is_left;
            .reg .pred  %p_left;
            .reg .pred %p_swap;
            .reg .u32 %lane_id;
            // $2 val, $3 idx, $4 stride;
            
            shfl.sync.bfly.b32 %partner_val, $2, $4, 0x1f, 0xffffffff;
            shfl.sync.bfly.b32 %partner_idx, $3, $4, 0x1f, 0xffffffff;

            mov.u32 %lane_id, %laneid;
            and.b32 %is_left, %lane_id, $4;
            setp.eq.u32 %p_left, %is_left, 0;
            setp.gt.f32 %p_swap, %partner_val, $2;
            xor.pred %p_swap, %p_swap, %p_left;
            not.pred %p_swap, %p_swap;

            selp.f32 $0, %partner_val, $2, %p_swap;
            selp.b32 $1, %partner_idx, $3, %p_swap;
        }""",
        constraints="=f,=r,f,r,n",
        args=[val, idx, stride],
        dtype=(tl.float32, tl.int32),
        is_pure=True,
        pack=1,
    )
    return new_val, new_idx


@triton.jit
def bitonic_sort32_descending(val, idx):
    # length_pair = 2, log2(2) = 1
    val, idx = bitonic_ce_descending(val, idx, 1, 1)

    # length_pair = 4, log2(4) = 2
    val, idx = bitonic_ce_descending(val, idx, 2, 2)
    val, idx = bitonic_ce_descending(val, idx, 1, 2)

    # length_pair = 8, log2(8) = 3
    val, idx = bitonic_ce_descending(val, idx, 4, 3)
    val, idx = bitonic_ce_descending(val, idx, 2, 3)
    val, idx = bitonic_ce_descending(val, idx, 1, 3)

    # length_pair = 16, log2(16) = 4
    val, idx = bitonic_ce_descending(val, idx, 8, 4)
    val, idx = bitonic_ce_descending(val, idx, 4, 4)
    val, idx = bitonic_ce_descending(val, idx, 2, 4)
    val, idx = bitonic_ce_descending(val, idx, 1, 4)

    # length_pair = 32, log2(32) = 5
    val, idx = bitonic_compare_across_part_descending(val, idx, 16)
    val, idx = bitonic_compare_across_part_descending(val, idx, 8)
    val, idx = bitonic_compare_across_part_descending(val, idx, 4)
    val, idx = bitonic_compare_across_part_descending(val, idx, 2)
    val, idx = bitonic_compare_across_part_descending(val, idx, 1)

    return val, idx


@triton.jit
def bitonic_ce_descending_wrapper(
    val_ptr, idx_ptr, new_val_ptr, new_idx_ptr, stride: tl.constexpr
):
    offs = tl.arange(0, 32)
    val = tl.load(val_ptr + offs)
    idx = tl.load(idx_ptr + offs)
    new_val, new_idx = bitonic_ce_descending(val, idx, stride, 1)
    tl.store(new_val_ptr + offs, new_val)
    tl.store(new_idx_ptr + offs, new_idx)


@triton.jit
def bitonic_sort32_descending_wrapper(val_ptr, idx_ptr, new_val_ptr, new_idx_ptr):
    offs = tl.arange(0, 32)
    val = tl.load(val_ptr + offs)
    idx = tl.load(idx_ptr + offs)
    new_val, new_idx = bitonic_sort32_descending(val, idx)
    tl.store(new_val_ptr + offs, new_val)
    tl.store(new_idx_ptr + offs, new_idx)
