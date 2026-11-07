# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton helper for packed global encoder FlexAttention masks."""

import torch
from torch.nn.attention.flex_attention import BlockMask, _mask_mod_signature

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import is_torch_equal_or_newer


@triton.jit(do_not_specialize=["num_tokens"])
def _build_encoder_span_rows(
    query_start_loc,
    doc_ids,
    partial_num_blocks,
    partial_block_indices,
    full_num_blocks,
    full_block_indices,
    num_tokens,
    q_block_size: tl.constexpr,
    kv_block_size: tl.constexpr,
    partial_stride: tl.constexpr,
    full_stride: tl.constexpr,
):
    q_block = tl.program_id(0)
    q_start = q_block * q_block_size
    q_end = tl.minimum(q_start + q_block_size, num_tokens)

    if q_start < num_tokens:
        first_req = tl.load(doc_ids + q_start).to(tl.int32)
        last_req = tl.load(doc_ids + q_end - 1).to(tl.int32)
        partial_count = 0
        full_count = 0

        if first_req == last_req:
            req_start = tl.load(query_start_loc + first_req).to(tl.int32)
            req_end = tl.load(query_start_loc + first_req + 1).to(tl.int32)
            first_full = tl.cdiv(req_start, kv_block_size)
            end_full = req_end // kv_block_size
            full_count = tl.maximum(end_full - first_full, 0)
            full_offsets = tl.arange(0, full_stride)
            tl.store(
                full_block_indices + q_block * full_stride + full_offsets,
                first_full + full_offsets,
                full_offsets < full_count,
            )

            if req_start % kv_block_size != 0:
                tl.store(
                    partial_block_indices + q_block * partial_stride,
                    req_start // kv_block_size,
                )
                partial_count = 1
            if req_end % kv_block_size != 0:
                boundary_block = req_end // kv_block_size
                if partial_count == 0:
                    tl.store(
                        partial_block_indices + q_block * partial_stride,
                        boundary_block,
                    )
                    partial_count = 1
                else:
                    previous = tl.load(
                        partial_block_indices
                        + q_block * partial_stride
                        + partial_count
                        - 1
                    )
                    if previous != boundary_block:
                        tl.store(
                            partial_block_indices
                            + q_block * partial_stride
                            + partial_count,
                            boundary_block,
                        )
                        partial_count += 1
        else:
            previous_req = -1
            for q_pos in range(q_start, q_end):
                req = tl.load(doc_ids + q_pos).to(tl.int32)
                if req != previous_req:
                    req_start = tl.load(query_start_loc + req).to(tl.int32)
                    req_end = tl.load(query_start_loc + req + 1).to(tl.int32)
                    first_kv = req_start // kv_block_size
                    end_kv = tl.cdiv(req_end, kv_block_size)
                    for kv_block in range(first_kv, end_kv):
                        if partial_count == 0:
                            tl.store(
                                partial_block_indices
                                + q_block * partial_stride
                                + partial_count,
                                kv_block,
                            )
                            partial_count += 1
                        else:
                            previous = tl.load(
                                partial_block_indices
                                + q_block * partial_stride
                                + partial_count
                                - 1
                            )
                            if kv_block > previous:
                                tl.store(
                                    partial_block_indices
                                    + q_block * partial_stride
                                    + partial_count,
                                    kv_block,
                                )
                                partial_count += 1
                    previous_req = req

        # FlexAttention may prefetch the first entry after each live row.
        tl.store(
            partial_block_indices + q_block * partial_stride + partial_count,
            -1,
            partial_count < partial_stride,
        )
        tl.store(
            full_block_indices + q_block * full_stride + full_count,
            -1,
            full_count < full_stride,
        )
        tl.store(partial_num_blocks + q_block, partial_count)
        tl.store(full_num_blocks + q_block, full_count)


def build_encoder_span_block_mask(
    *,
    query_start_loc: torch.Tensor,
    doc_ids: torch.Tensor,
    num_tokens: int,
    max_model_len: int,
    q_block_size: int,
    kv_block_size: int,
    mask_mod: _mask_mod_signature,
) -> BlockMask:
    """Build block rows for packed requests with global bidirectional attention."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if query_start_loc.device.type != "cuda" or doc_ids.device.type != "cuda":
        raise ValueError("encoder span construction requires CUDA tensors")

    num_q_blocks = cdiv(num_tokens, q_block_size)
    max_kv_blocks = cdiv(max_model_len, kv_block_size)
    full_stride = triton.next_power_of_2(max(2, max_kv_blocks + 1))
    # A mixed query row can touch two long boundary requests and spans contained
    # by the query row itself; leave extra entries for block-boundary rounding.
    partial_capacity = 2 * max_kv_blocks + cdiv(q_block_size, kv_block_size) + 2
    partial_stride = triton.next_power_of_2(max(1, partial_capacity))
    device = doc_ids.device

    partial_num_blocks = torch.empty((num_q_blocks,), dtype=torch.int32, device=device)
    partial_block_indices = torch.empty(
        (num_q_blocks, partial_stride), dtype=torch.int32, device=device
    )
    full_num_blocks = torch.empty((num_q_blocks,), dtype=torch.int32, device=device)
    full_block_indices = torch.empty(
        (num_q_blocks, full_stride), dtype=torch.int32, device=device
    )

    _build_encoder_span_rows[(num_q_blocks,)](
        query_start_loc,
        doc_ids,
        partial_num_blocks,
        partial_block_indices,
        full_num_blocks,
        full_block_indices,
        num_tokens,
        q_block_size,
        kv_block_size,
        partial_stride,
        full_stride,
        num_warps=4,
    )

    block_mask_kwargs = {
        "seq_lengths": (num_tokens, num_tokens),
        "kv_num_blocks": partial_num_blocks[None, None, :],
        "kv_indices": partial_block_indices[None, None, :, :],
        "full_kv_num_blocks": full_num_blocks[None, None, :],
        "full_kv_indices": full_block_indices[None, None, :, :],
        "BLOCK_SIZE": (q_block_size, kv_block_size),
        "mask_mod": mask_mod,
    }
    if is_torch_equal_or_newer("2.9.0.dev0"):
        block_mask_kwargs["compute_q_blocks"] = False
    return BlockMask.from_kv_blocks(**block_mask_kwargs)
