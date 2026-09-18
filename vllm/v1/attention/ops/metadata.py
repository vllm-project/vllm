# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side request mapping and sparse indexer metadata."""

from vllm.triton_utils import tl, triton


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
