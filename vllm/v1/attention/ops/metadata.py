# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side request mapping and sparse indexer metadata."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _build_dcp_valid_token_mask_kernel(
    seq_lens,
    query_start_loc,
    out,
    NUM_REQS: tl.constexpr,
    CAPACITY: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = tl.full((BLOCK,), False, tl.int1)
    if NUM_REQS > 0:
        # Upper-bound search skips requests with zero query tokens, including
        # repeated trailing boundaries used for CUDA graph padding.
        low = tl.full((BLOCK,), 0, tl.int32)
        high = tl.full((BLOCK,), NUM_REQS, tl.int32)
        for _ in range(SEARCH_STEPS):
            mid = (low + high) // 2
            end = tl.load(query_start_loc + mid + 1, mid < NUM_REQS, other=0)
            advance = (mid < NUM_REQS) & (end <= token)
            low = tl.where(advance, mid + 1, low)
            high = tl.where(advance, high, mid)
        local_len = tl.load(seq_lens + low, low < NUM_REQS, other=0)
        valid = (low < NUM_REQS) & (local_len > 0)
    tl.store(out + token, valid, token < CAPACITY)


def build_dcp_valid_token_mask(
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Refresh a persistent token mask once per metadata build, including padding.

    The persistent buffer is shared by layers using the same attention metadata.
    Updating it in place before replay preserves captured device pointers.
    """
    from vllm.v1.attention.ops.dcp import _validate_dcp_empty_shard_args

    _validate_dcp_empty_shard_args(seq_lens, query_start_loc)
    if out.ndim != 1 or out.dtype != torch.bool or not out.is_contiguous():
        raise ValueError("FI DCP token mask must be a contiguous bool vector")
    if out.device.type != "cuda" or any(
        value.device != out.device
        or value.dtype not in (torch.int32, torch.int64)
        or not value.is_contiguous()
        for value in (seq_lens, query_start_loc)
    ):
        raise ValueError("FI DCP mask metadata must be contiguous CUDA integer vectors")
    if out.numel():
        _build_dcp_valid_token_mask_kernel[(triton.cdiv(out.numel(), 256),)](
            seq_lens,
            query_start_loc,
            out,
            NUM_REQS=seq_lens.numel(),
            CAPACITY=out.numel(),
            SEARCH_STEPS=(seq_lens.numel() + 1).bit_length(),
            BLOCK=256,
        )


@triton.jit
def _token_request(query_start_loc, token, num_reqs):
    lo = tl.full(token.shape, 0, tl.int32)
    hi = tl.full(token.shape, num_reqs, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(query_start_loc + mid + 1, mid < num_reqs, other=0x7FFFFFFF)
        right = (lo < hi) & (end <= token)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where((lo < hi) & ~right, mid, hi)
    return tl.minimum(lo, num_reqs - 1)


@triton.jit(do_not_specialize=["num_reqs", "num_mapped", "num_tokens"])
def _token_request_mapping_kernel(
    query_start_loc, output, num_reqs, num_mapped, num_tokens
):
    token = tl.program_id(0) * 256 + tl.arange(0, 256)
    req = _token_request(query_start_loc, token, num_reqs)
    tl.store(output + token, tl.where(token < num_mapped, req, 0), token < num_tokens)


def compute_token_to_req_indices(
    query_start_loc: torch.Tensor,
    out: torch.Tensor,
    num_mapped_tokens: int,
    num_tokens: int,
) -> torch.Tensor:
    """Map each of the first ``num_tokens`` tokens to its request index.

    Reads only the device ``query_start_loc``, so it is safe to record in a
    CUDA graph. Tokens at or past ``num_mapped_tokens`` are mapped to 0.
    """
    _token_request_mapping_kernel[((num_tokens + 255) // 256,)](
        query_start_loc,
        out,
        query_start_loc.shape[0] - 1,
        num_mapped_tokens,
        num_tokens,
        num_warps=4,
    )
    return out[:num_tokens]


@triton.jit(
    do_not_specialize=[
        "num_reqs",
        "num_tokens",
        "capacity",
        "input_stride",
        "output_stride",
    ]
)
def _indexer_decode_metadata_kernel(
    query_start_loc,
    seq_lens,
    block_table,
    output_seq_lens,
    output_block_table,
    decode_lens,
    indices,
    per_req_lens,
    num_reqs,
    num_tokens,
    capacity,
    input_stride,
    output_stride,
    BLOCK_COLS: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, triton.next_power_of_2(BLOCK_COLS))
    actual_tokens = tl.load(query_start_loc + num_reqs)
    if token < num_tokens:
        req = _token_request(query_start_loc, tl.full((1,), token, tl.int32), num_reqs)
        req = tl.sum(req, 0)
        valid = token < actual_tokens
        start = tl.load(query_start_loc + req)
        end = tl.load(query_start_loc + req + 1)
        seq = tl.load(seq_lens + req)
        length = tl.where(valid, seq - (end - start) + token - start + 1, 0)
        tl.store(output_seq_lens + token, length)
        tl.store(decode_lens + token, 1)
        tl.store(
            indices + token, tl.where(valid, req, num_reqs + token - actual_tokens)
        )
        blocks = tl.load(
            block_table + req.to(tl.int64) * input_stride + cols,
            valid & (cols < BLOCK_COLS),
            other=0,
        )
        tl.store(
            output_block_table + token.to(tl.int64) * output_stride + cols,
            blocks,
            cols < BLOCK_COLS,
        )
    else:
        tail = num_tokens + (token - num_tokens) * 256 + tl.arange(0, 256)
        tl.store(output_seq_lens + tail, 0, tail < capacity)
    if token < num_reqs:
        start = tl.load(query_start_loc + token)
        end = tl.load(query_start_loc + token + 1)
        tl.store(per_req_lens + token, end - start)
