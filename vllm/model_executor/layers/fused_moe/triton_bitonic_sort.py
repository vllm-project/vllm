# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.triton_utils import tl, triton

"""
warp-level bitonic sort using PTX inline asm.

this module implements a 32-element bitonic sort that runs entirely within
a single warp using counterpart of shfl_sync function like in CUDA C,
avoiding shared memory overhead.

bitonic sort overview:
    bitonic sort is a parallel sorting algorithm that works in two phases:
    1. build phase: construct a "bitonic sequence" where the first half is
       ascending and the second half is descending (or vice versa).
    2. merge phase: recursively merge bitonic sequences into sorted order.

    for 32 elements, this requires:
    - build phase: 10 compare-exchange steps (length_pair = 2,4,8,16)
    - merge phase: 5 compare-exchange steps (stride = 16,8,4,2,1)
    - total: 15 warp shuffle operations

references:
    - https://en.wikipedia.org/wiki/Bitonic_sorter
    - https://www.geeksforgeeks.org/dsa/bitonic-sort/

PTX register types:
    we declare registers as .f32/.s32. sub-word types (8/16-bit)
    are held in 32-bit registers when loaded, stored, or converted.
    see: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=shfl#restricted-use-of-sub-word-sizes
"""


@triton.jit
def bitonic_compare_exchange_descending(
    val, idx, stride: tl.constexpr, log2_length_pair: tl.constexpr
):
    """
    bitonic compare_exchange for building bitonic sequences (descending order).

    this function performs one step of bitonic sequence construction. within
    each "length_pair" group, adjacent pairs alternate between ascending and
    descending order to form bitonic sequences.

    args:
        val: values to sort
        idx: corresponding indices to track original positions
        stride: distance between elements to compare (power of 2)
        log2_length_pair: log2 of the current subsequence length being processed.
            Used to determine sort direction (ascending/descending) for each group.

    example (length_pair=4, stride=2):
        before: [3, 1, 4, 2, 8, 6, 7, 5]
        groups: [3, 1, 4, 2] [8, 6, 7, 5]
                 ↓ desc      ↓ asc
        after:  [4, 2, 3, 1] [7, 5, 8, 6]

    PTX implementation:
        1. shfl.sync.bfly: save partner lane's value or index in own register
            by mode=XOR(bfly)
        2. determine group: (lane >> log2_length_pair) & 1 gives alternating
            (0 or 1, imply mean or odd) groups
        3. determine position: left ((lane & stride) == 0) or right
        4. compare and conditionally swap based on position and group.
        we only do swap if such cases:
            a, left < right in mean group;
            b, left > right in odd group.
    """

    new_val, new_idx = tl.inline_asm_elementwise(
        asm="""
        {
            // counterpart elems registers
            .reg .f32 %partner_val;
            .reg .s32 %partner_idx;
            .reg .u32 %lane_id;
            .reg .u32 %is_left;
            .reg .pred %p_left, %p_swap;
            .reg .u32 %group_id;
            .reg .pred %group_id_mask;
            
            // save partner's regs
            shfl.sync.bfly.b32 %partner_val, $2, $4, 0x1f, 0xffffffff;
            shfl.sync.bfly.b32 %partner_idx, $3, $4, 0x1f, 0xffffffff;
            
            // we can't directly use special register %laneid in many insts.
            mov.u32 %lane_id, %laneid;
            
            // assign elems in length_pair to a group id, and .pred reg 
            // indicates mean or odd.
            shr.u32 %group_id, %lane_id, $5;
            and.b32 %group_id, %group_id, 1;
            setp.eq.u32 %group_id_mask, %group_id, 1;
            
            // recognize if left or right in each lengthpair
            and.b32 %is_left, %lane_id, $4;
            setp.eq.u32 %p_left, %is_left, 0;
            
            // compare partner_val > val? if so, swap.
            setp.gt.f32 %p_swap, %partner_val, $2;

            // XNOR: left and swap -> swap, right and not swap -> swap.    
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
        # declaring pack=1 indicates each thread proceed 1 element at a time.
        # along this way, to ensure warp proceed 32 elements.
    )
    return new_val, new_idx


@triton.jit
def bitonic_compare_across_part_descending(val, idx, stride: tl.constexpr):
    """
    bitonic merge step: compare_exchange for final merging (descending order).

    args:
        val: values to sort
        idx: corresponding indices
        stride: distance between elements to compare

    behavior:
        compare values within pairs constructed by stride.
        this pushes larger values toward lower lane indices (descending order).
    """
    new_val, new_idx = tl.inline_asm_elementwise(
        asm="""{
            .reg .f32 %partner_val;
            .reg .s32 %partner_idx;
            .reg .u32 %is_left;
            .reg .pred  %p_left;
            .reg .pred %p_swap;
            .reg .u32 %lane_id;
            // $2 val, $3 idx, $4 stride;
            
            // save partner's register
            shfl.sync.bfly.b32 %partner_val, $2, $4, 0x1f, 0xffffffff;
            shfl.sync.bfly.b32 %partner_idx, $3, $4, 0x1f, 0xffffffff;

            // determine left or right in each pair
            mov.u32 %lane_id, %laneid;
            and.b32 %is_left, %lane_id, $4;
            setp.eq.u32 %p_left, %is_left, 0;

            // use XNOR here.
            // if partner's value > my value, predicate swap.
            // if (predicate swap && left) || (!(predicate swap) && right), swap.
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
def bitonic_sort_warp_size_descending(val, idx):
    """
    Refer to https://www.geeksforgeeks.org/dsa/bitonic-sort/ to understand bitonic sort,
    or a more formal reference: https://en.wikipedia.org/wiki/Bitonic_sorter.
    We build bitonic sequence at first, then proceed merge step.
    """
    # build phase
    # length_pair = 2, log2(2) = 1
    val, idx = bitonic_compare_exchange_descending(val, idx, 1, 1)

    # length_pair = 4, log2(4) = 2
    val, idx = bitonic_compare_exchange_descending(val, idx, 2, 2)
    val, idx = bitonic_compare_exchange_descending(val, idx, 1, 2)

    # length_pair = 8, log2(8) = 3
    val, idx = bitonic_compare_exchange_descending(val, idx, 4, 3)
    val, idx = bitonic_compare_exchange_descending(val, idx, 2, 3)
    val, idx = bitonic_compare_exchange_descending(val, idx, 1, 3)

    # length_pair = 16, log2(16) = 4
    val, idx = bitonic_compare_exchange_descending(val, idx, 8, 4)
    val, idx = bitonic_compare_exchange_descending(val, idx, 4, 4)
    val, idx = bitonic_compare_exchange_descending(val, idx, 2, 4)
    val, idx = bitonic_compare_exchange_descending(val, idx, 1, 4)

    # merge phase
    # length_pair = 32, log2(32) = 5
    val, idx = bitonic_compare_across_part_descending(val, idx, 16)
    val, idx = bitonic_compare_across_part_descending(val, idx, 8)
    val, idx = bitonic_compare_across_part_descending(val, idx, 4)
    val, idx = bitonic_compare_across_part_descending(val, idx, 2)
    val, idx = bitonic_compare_across_part_descending(val, idx, 1)

    return val, idx


"""
wrapper function used in unittest or other purpose.
"""


@triton.jit
def bitonic_compare_exchange_descending_wrapper(
    val_ptr,
    idx_ptr,
    new_val_ptr,
    new_idx_ptr,
    stride: tl.constexpr,
    log2_length_pair: tl.constexpr,
):
    offs = tl.arange(0, 32)
    val = tl.load(val_ptr + offs)
    idx = tl.load(idx_ptr + offs)
    new_val, new_idx = bitonic_compare_exchange_descending(
        val, idx, stride, log2_length_pair
    )
    tl.store(new_val_ptr + offs, new_val)
    tl.store(new_idx_ptr + offs, new_idx)


@triton.jit
def bitonic_sort_warp_size_descending_wrapper(
    val_ptr, idx_ptr, new_val_ptr, new_idx_ptr
):
    offs = tl.arange(0, 32)
    val = tl.load(val_ptr + offs)
    idx = tl.load(idx_ptr + offs)
    new_val, new_idx = bitonic_sort_warp_size_descending(val, idx)
    tl.store(new_val_ptr + offs, new_val)
    tl.store(new_idx_ptr + offs, new_idx)
