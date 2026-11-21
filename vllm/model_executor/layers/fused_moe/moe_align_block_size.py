# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, overload

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up


@overload
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: Literal[False] = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: bool,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
): ...


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    return_scatter_idx: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

    Args:
        topk_ids: A tensor of shape [total_tokens, top_k] representing the
            top-k expert indices for each token.
        block_size: The block size used in block matrix multiplication.
        num_experts: The total number of experts.
        expert_map: A tensor of shape [num_experts] that maps the expert index
            from the global space to the local index space of the current
            expert parallel shard. If the expert is not in the current expert
            parallel shard, the mapping is set to -1.
        pad_sorted_ids: A flag indicating whether the sorted_token_ids length
            should be padded to a multiple of block_size,
        ignore_invalid_experts: A flag indicating whether to ignore invalid
            experts. When False, all expert_ids in topk_ids will participate in
            counting and ranking, but invalid experts in expert_ids will be marked
            as -1. When True, all invalid expert_ids in topk_ids will be ignored
            and will not participate in counting or ranking, and there will be no
            -1 in expert_ids.
        return_scatter_idx: Whether to additionally return a masked identity
            mapping for the original routed rows.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.
    - scatter_idx: Only returned when return_scatter_idx=True. An int32 tensor
        shaped [topk_ids.numel(), 1], containing the original routed row index
        for valid routes and -1 otherwise. With expert_map, this requires
        ignore_invalid_experts=True.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.

    """
    if return_scatter_idx and expert_map is not None and not ignore_invalid_experts:
        raise ValueError("scatter_idx with expert_map requires ignore_invalid_experts")
    scatter_idx = None
    if return_scatter_idx:
        scatter_idx = torch.empty(
            (topk_ids.numel(), 1),
            dtype=torch.int32,
            device=topk_ids.device,
        )
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map if ignore_invalid_experts else None,
        scatter_idx,
    )

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    if scatter_idx is not None:
        return sorted_ids, expert_ids, num_tokens_post_pad, scatter_idx
    return sorted_ids, expert_ids, num_tokens_post_pad


def batched_moe_align_block_size(
    max_tokens_per_batch: int, block_size: int, expert_num_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given num_batches, max_tokens_per_batch, block_size and the number of
    valid-tokens in each batch, prepare sorted_token_ids, expert_ids and
    num_tokens_post_pad. sorted_token_ids, expert_ids and num_tokens_post_pad
    have the same semantics as in moe_align_block_size.

    This function is intended to be a drop in replacement for
    moe_align_batch_size for the batched case.

    Args:
        max_tokens_per_batch (int): Number of tokens in each batch (both
            valid and invalid).
        block_size (int): block_size to align the data to.
        expert_num_tokens (torch.Tensor): expert_num_tokens[i], indicates
            the number of valid tokens in batch i.

    Returns:
    - sorted_token_ids (torch.Tensor): Torch tensor of size
        (num_batches * max_tokens_per_batch) indicating the token indices for
        that block.
    - expert_ids (torch.Tensor): Torch tensor of size
        ceil((num_batches * max_tokens_per_batch) / block_size) indicating
        what expert to use for each block.
    - num_tokens_post_pad (torch.Tensor): Torch tensor of size 1
        indicating the number of valid blocks with actual data to
        process. This is represented in terms of num tokens.

    Example:
    Let num_batches=5, max_tokens_per_batch=8, block_size=4, and
    expert_num_tokens=[2, 3, 0, 6, 8]. This expert_num_tokens tensor
    indicates that,
     - The first 2 tokens in the 0th batch are valid and the rest 6 are
     invalid (i.e. in the 2D hidden_states tensor of shape,
     [num_batches * max_tokens_per_batch, K], indices 0, 1 are valid)
     - The first 3 tokens in the 1st batch are valid. i.e. indices 8, 9, 10
     - 0 tokens in the 2nd batch are valid
     - first 6 tokens in the  3rd batch are valid. i.e. indices,
     24, 25, 26, 27, 28, 29
     - so on ...

     In this case,
      sorted_token_ids will be [0, 1, 40, 40,
                                8, 9, 10, 40,
                                24, 25, 26, 27,
                                28, 29, 40, 40,
                                32, 33, 34, 35,
                                36, 37, 38, 39,
                                40, 40, 40, 40,
                                (rest all 40, 40, 40, 40)
                                ...]
      Here, 40 represents an invalid index. as there is no token index 40.
      The gemm kernel using this sorted_token_ids is expected to skip the
      gemm computation when it encounters this invalid index.

      expert_ids will be [0, 1, 3, 3, 4, 5, 5, -1, -1, (rest all -1) ...]
      Here, -1 represents an invalid expert. The gemm kernel using this
      expert_ids is expected to skip the gemm computation when it encounters
      an expert of id -1.

      num_tokens_post_pad will be 24 as sorted_token_ids has valid entries
      until 24.

    """
    B = expert_num_tokens.size(0)
    device = expert_num_tokens.device

    # Round up so each batch can be split to blocks evenly.
    max_num_tokens_padded = B * round_up(max_tokens_per_batch, block_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    assert max_num_tokens_padded % block_size == 0
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=device)

    ops.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad


@triton.jit
def _moe_align_block_size_pair_kernel(
    TopkIds,
    ExpertMap,
    SortedIds1,
    ExpertIds1,
    NumTokens1,
    SortedIds2,
    ExpertIds2,
    NumTokens2,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    STRIDE_TOKEN: tl.constexpr,
    STRIDE_TOPK: tl.constexpr,
    STRIDE_MAP: tl.constexpr,
    BLOCK_SIZE1: tl.constexpr,
    BLOCK_SIZE2: tl.constexpr,
    SORTED_SIZE1: tl.constexpr,
    SORTED_SIZE2: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
):
    route = tl.arange(0, BLOCK_ROUTES)
    expert = tl.load(
        TopkIds + route // TOP_K * STRIDE_TOKEN + route % TOP_K * STRIDE_TOPK,
        route < NUM_ROUTES,
        -1,
    ).to(tl.int32)
    valid = (route < NUM_ROUTES) & (expert >= 0) & (expert < NUM_EXPERTS)
    if HAS_EXPERT_MAP:
        expert = tl.load(ExpertMap + expert * STRIDE_MAP, valid, -1)
        valid &= expert >= 0

    # Count routed rows rather than scan the full expert space for decode.
    same_expert = expert[:, None] == expert[None, :]
    earlier = (route[None, :] < route[:, None]) & valid[None, :]
    rank = tl.sum((same_expert & earlier).to(tl.int32), 1)
    count = tl.sum((same_expert & valid[None, :]).to(tl.int32), 1)
    leader = valid & (rank == 0)
    padded1 = tl.where(leader, tl.cdiv(count, BLOCK_SIZE1) * BLOCK_SIZE1, 0)
    padded2 = tl.where(leader, tl.cdiv(count, BLOCK_SIZE2) * BLOCK_SIZE2, 0)
    lower_expert = expert[None, :] < expert[:, None]
    offset1 = tl.sum(tl.where(lower_expert, padded1[None, :], 0), 1)
    offset2 = tl.sum(tl.where(lower_expert, padded2[None, :], 0), 1)

    sorted1 = tl.arange(0, triton.next_power_of_2(SORTED_SIZE1))
    sorted2 = tl.arange(0, triton.next_power_of_2(SORTED_SIZE2))
    experts1 = tl.arange(
        0, triton.next_power_of_2(triton.cdiv(SORTED_SIZE1, BLOCK_SIZE1))
    )
    experts2 = tl.arange(
        0, triton.next_power_of_2(triton.cdiv(SORTED_SIZE2, BLOCK_SIZE2))
    )
    tl.store(SortedIds1 + sorted1, NUM_ROUTES, sorted1 < SORTED_SIZE1)
    tl.store(SortedIds2 + sorted2, NUM_ROUTES, sorted2 < SORTED_SIZE2)
    tl.store(ExpertIds1 + experts1, -1, experts1 < tl.cdiv(SORTED_SIZE1, BLOCK_SIZE1))
    tl.store(ExpertIds2 + experts2, -1, experts2 < tl.cdiv(SORTED_SIZE2, BLOCK_SIZE2))
    # Initializing padding and writing valid rows may involve different warps.
    tl.debug_barrier()
    tl.store(SortedIds1 + offset1 + rank, route, valid)
    tl.store(SortedIds2 + offset2 + rank, route, valid)
    block1 = tl.arange(0, triton.next_power_of_2(triton.cdiv(NUM_ROUTES, BLOCK_SIZE1)))
    block2 = tl.arange(0, triton.next_power_of_2(triton.cdiv(NUM_ROUTES, BLOCK_SIZE2)))
    tl.store(
        ExpertIds1 + offset1[:, None] // BLOCK_SIZE1 + block1[None, :],
        expert[:, None],
        leader[:, None] & (block1[None, :] < padded1[:, None] // BLOCK_SIZE1),
    )
    tl.store(
        ExpertIds2 + offset2[:, None] // BLOCK_SIZE2 + block2[None, :],
        expert[:, None],
        leader[:, None] & (block2[None, :] < padded2[:, None] // BLOCK_SIZE2),
    )
    tl.store(NumTokens1, tl.sum(padded1, 0))
    tl.store(NumTokens2, tl.sum(padded2, 0))


def moe_align_block_size_pair(
    topk_ids: torch.Tensor,
    block_size1: int,
    block_size2: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Align the same routes for two independently tuned MoE projections.

    Each result has the semantics of ``moe_align_block_size`` with
    ``ignore_invalid_experts=True``. Equal block sizes share one result.
    Small Hopper batches use one launch; other batches use the existing op.
    """
    if block_size1 == block_size2:
        alignment = moe_align_block_size(
            topk_ids.contiguous(),
            block_size1,
            num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )
        return alignment, alignment

    num_routes = topk_ids.numel()
    # Bound the quadratic route comparison and per-CTA padding initialization.
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(90)
        and 0 < num_routes <= 64
        and max(block_size1, block_size2) <= 256
    ):
        alignments = [
            moe_align_block_size(
                topk_ids.contiguous(),
                block_size,
                num_experts,
                expert_map,
                ignore_invalid_experts=True,
            )
            for block_size in (block_size1, block_size2)
        ]
        return alignments[0], alignments[1]

    outputs = []
    for block_size in (block_size1, block_size2):
        max_num_tokens_padded = min(
            num_routes * block_size, num_routes + num_experts * (block_size - 1)
        )
        outputs.append(
            (
                torch.empty(
                    max_num_tokens_padded, device=topk_ids.device, dtype=torch.int32
                ),
                torch.empty(
                    triton.cdiv(max_num_tokens_padded, block_size),
                    device=topk_ids.device,
                    dtype=torch.int32,
                ),
                torch.empty(1, device=topk_ids.device, dtype=torch.int32),
            )
        )
    _moe_align_block_size_pair_kernel[(1,)](
        topk_ids,
        expert_map,
        *outputs[0],
        *outputs[1],
        num_routes,
        num_experts,
        topk_ids.size(1),
        *topk_ids.stride(),
        expert_map.stride(0) if expert_map is not None else 1,
        block_size1,
        block_size2,
        outputs[0][0].numel(),
        outputs[1][0].numel(),
        expert_map is not None,
        triton.next_power_of_2(num_routes),
        num_warps=4,
    )
    return outputs[0], outputs[1]
