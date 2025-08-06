# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

HEAD_TILE_SIZE: int = 64
MAX_TREE_DEPTH: int = 16


def apply_draft_offsets(
    tree_draft_offsets: list[int],
    input_batch: "InputBatch",
    scheduler_output: "SchedulerOutput",
    query_start_loc_np: np.array,
    token_positions_np: np.array,
):
    """
    Updates the draft token positions with their offsets (levels) in the tree.

    Args:
        tree_draft_offsets: Offsets to apply to the draft token positions.
        input_batch: The input batch.
        scheduler_output: The scheduler output.
        query_start_loc_np: Start locations of the queries for each request.
        token_positions_np: Token positions. Will be updated in-place.
    """

    draft_token_offsets = np.array(tree_draft_offsets)
    for (
            req_id,
            draft_token_ids,
    ) in scheduler_output.scheduled_spec_decode_tokens.items():
        if len(draft_token_ids) == 0:
            continue
        req_idx = input_batch.req_id_to_index[req_id]
        start = query_start_loc_np[req_idx]
        end = query_start_loc_np[req_idx + 1]
        num_drafts = end - start - 1
        token_positions_np[start + 1:end] = (token_positions_np[start] +
                                             draft_token_offsets[:num_drafts])


def apply_accepted_draft_indices(
    sampled_token_indices: list[list[int]],
    query_start_loc_np: np.array,
    token_indices_np: np.array,
):
    """
    Updates token_indices_np with the sampled token indices.

    Args:
        sampled_token_indices: The indices of the accepted draft tokens.
        query_start_loc_np: Start locations of the queries for each request.
        token_indices_np: Token indices. Will be updated in-place.
    """

    for req_idx, seq in enumerate(sampled_token_indices):
        if len(seq) <= 1:
            continue
        start = query_start_loc_np[req_idx] + 1
        end = query_start_loc_np[req_idx + 1]
        token_indices_np[start:end] = token_indices_np[start] + np.array(
            seq[:-1])


def copy_kv_cache_slots(
    kv_caches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    from_token_indices: torch.Tensor,
    to_token_indices: torch.Tensor,
):
    """
    Copies K/Vs from from_token_indices to to_token_indices. Used for updating
    the KV cache after tree rejection sampling.

    Args:
        kv_caches: List of per-layer tensors, each with shape
                   (2, num_blocks, block_size, num_kv_heads, head_size)
        slot_mapping: Tensor mapping token indices to positions in the KV
                      cache.
        from_token_indices: Tensor containing token indices into slot_mapping
                            to copy from.
        to_token_indices: Tensor containing token indices into slot_mapping to
                            copy to.
    """
    if len(kv_caches) == 0:
        # Nothing to do.
        return

    # Get shape and stride from first kv_cache tensor.
    first_kv_cache = kv_caches[0]
    assert first_kv_cache.dtype == torch.bfloat16, (
        "Only bfloat16 is supported for now.")
    device = first_kv_cache.device
    KV, _, block_size, num_kv_heads, head_size = first_kv_cache.shape
    s0, s1, s2, s3, s4 = first_kv_cache.stride()
    assert KV == 2
    num_layers = len(kv_caches)

    # Prepare indices.
    from_indices = from_token_indices.contiguous()
    to_indices = to_token_indices.contiguous()
    slot_mapping = slot_mapping.contiguous()
    num_indices = from_indices.numel()
    assert to_indices.numel() == num_indices

    # Create array of pointers to kv_cache tensors.
    kv_cache_ptrs = torch.tensor(
        [kv_cache.data_ptr() for kv_cache in kv_caches],
        dtype=torch.int64,
        device=device,
    )

    # Compute grid dimensions.
    # Encode KV/layers/heads.
    grid0 = 2 * num_layers * num_kv_heads
    # Chunk size is set to the maximum tree depth to prevent a potential
    # race condition of writing to while reading from the same slot.
    chunk_size = MAX_TREE_DEPTH
    # Chunk across indices.
    grid1 = (num_indices + chunk_size - 1) // chunk_size
    # Tile across head_size.
    grid2 = (head_size + HEAD_TILE_SIZE - 1) // HEAD_TILE_SIZE

    # Launch single kernel for all layers.
    _kv_copy_chunked[grid0, grid1, grid2](
        kv_cache_ptrs.data_ptr(),
        slot_mapping.data_ptr(),
        from_indices.data_ptr(),
        to_indices.data_ptr(),
        num_indices,
        block_size,
        num_kv_heads,
        head_size,
        s0,
        s1,
        s2,
        s3,
        s4,
        head_tile_size=HEAD_TILE_SIZE,
        chunk_size=chunk_size,
    )


@triton.jit
def _kv_copy_chunked(
    kv_ptrs_ptr,
    slot_mapping_ptr,
    from_indices_ptr,
    to_indices_ptr,
    num_indices,
    block_size,
    num_kv_heads,
    head_size,
    s0,
    s1,
    s2,
    s3,
    s4,
    head_tile_size: tl.constexpr,
    chunk_size: tl.constexpr,
):
    pid0 = tl.program_id(0)
    chunk_id = tl.program_id(1)
    tile_id = tl.program_id(2)

    # Compute offsets in head dimension for this tile.
    tile_offsets = tile_id * head_tile_size + tl.arange(0, head_tile_size)
    mask_tile = tile_offsets < head_size

    # Decode pid0 -> KV, layer, head.
    tmp = pid0 // num_kv_heads
    KV = tmp % 2
    layer = tmp // 2
    head = pid0 % num_kv_heads

    # Load the pointer for this layer's kv_cache from the array.
    # 8 bytes per pointer (64-bit).
    kv_ptr_addr = kv_ptrs_ptr + layer * 8
    kv_ptr = tl.load(kv_ptr_addr.to(tl.pointer_type(tl.uint64)))
    src_ptr = kv_ptr.to(tl.pointer_type(tl.bfloat16))
    # Copying to the same tensor, so dst_ptr == src_ptr.
    dst_ptr = src_ptr

    # Fixed base depending on kv/head (no layer stride since we select the
    # tensor).
    base_fixed = KV * s0 + head * s3

    # Compute chunk bounds.
    chunk_start = chunk_id * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, num_indices)

    # Cast pointers to appropriate types.
    from_idx_base = from_indices_ptr.to(tl.pointer_type(tl.uint64))
    to_idx_base = to_indices_ptr.to(tl.pointer_type(tl.uint64))
    slot_mapping_base = slot_mapping_ptr.to(tl.pointer_type(tl.uint64))
    # Cast block_size to uint64 to match slot type.
    block_size_u64 = block_size.to(tl.uint64)

    for idx in range(chunk_start, chunk_end):
        # Load indices into slot_mapping.
        from_idx = tl.load(from_idx_base + idx)
        to_idx = tl.load(to_idx_base + idx)

        # Load actual KV cache slots from slot_mapping.
        from_slot = tl.load(slot_mapping_base + from_idx)
        to_slot = tl.load(slot_mapping_base + to_idx)

        # Skip copying if from_slot == to_slot.
        should_copy = from_slot != to_slot

        # Convert slots to blocks and offsets.
        from_block = from_slot // block_size_u64
        from_offset = from_slot % block_size_u64
        to_block = to_slot // block_size_u64
        to_offset = to_slot % block_size_u64

        src_base = base_fixed + from_block * s1 + from_offset * s2
        dst_base = base_fixed + to_block * s1 + to_offset * s2

        src_addr = (src_ptr + src_base + tile_offsets * s4).to(
            tl.pointer_type(tl.bfloat16))
        dst_addr = (dst_ptr + dst_base + tile_offsets * s4).to(
            tl.pointer_type(tl.bfloat16))

        copy_mask = mask_tile & should_copy
        vals = tl.load(src_addr, mask=copy_mask, other=0)
        tl.store(dst_addr, vals, mask=copy_mask)
