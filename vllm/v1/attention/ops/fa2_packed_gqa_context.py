# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA2 short-query attention over a long cached context.

FA2 packs GQA heads and splits the KV sequence only when ``max_seqlen_q == 1``;
longer queries stream the whole context once per query head on few SMs. Here
the cached context is attended non-causally with query heads folded into rows
and the context cut into chunks FA2 sees as separate sequences, the new tokens
are attended causally, and the partials are merged by log-sum-exp.
"""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.utils import CpuGpuBuffer

_FA2_BLOCK_M = 64  # FA2 forward tile height for head_dim <= 128 on SM8x.
# A100: the kernel breaks even near 8K, end-to-end prefill only past 16K.
MIN_CONTEXT_LEN = 32768
MAX_PACKED_QUERY_ROWS = 512
_MIN_CHUNK_BLOCKS = 16
_MAX_CHUNKS_PER_REQ = 64


@dataclass
class PackedGQAContextPlan:
    """Per-batch layout: one virtual sequence per (request, context chunk)."""

    cu_seqlens_q: torch.Tensor  # [num_chunks + 1] int32, packed rows
    chunk_block_table: torch.Tensor  # [num_chunks, chunk_blocks] int32
    chunk_kv_lens: torch.Tensor  # [num_chunks] int32
    max_chunk_kv_len: int
    max_packed_q: int
    row_token: torch.Tensor  # [num_rows] int64, packed row -> query token
    row_member: torch.Tensor  # [num_rows] int64, packed row -> GQA member
    token_req: torch.Tensor  # [num_tokens] int32
    req_first_chunk: torch.Tensor  # [num_reqs] int32
    req_num_chunks: torch.Tensor  # [num_reqs] int32


class PackedGQAContextBuffers:
    """Pinned host staging and device buffers for the plan, sized once."""

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        group_size: int,
        num_kv_heads: int,
        num_sms: int,
        device: torch.device,
    ):
        max_chunks = max_num_reqs * _MAX_CHUNKS_PER_REQ
        # Chunking stops once the chunk grid reaches 2 * num_sms CTAs.
        max_rows = max(
            max_num_tokens * group_size,
            2 * num_sms * _FA2_BLOCK_M // num_kv_heads,
        )

        def buf(n: int, dtype: torch.dtype) -> CpuGpuBuffer:
            return CpuGpuBuffer(n, dtype=dtype, device=device)

        self.cu_seqlens_q = buf(max_chunks + 1, torch.int32)
        self.chunk_req = buf(max_chunks, torch.int64)
        self.chunk_start_block = buf(max_chunks, torch.int64)
        self.row_token = buf(max_rows, torch.int64)
        self.row_member = buf(max_rows, torch.int64)
        self.token_req = buf(max_num_tokens, torch.int32)
        self.req_first_chunk = buf(max_num_reqs, torch.int32)
        self.req_num_chunks = buf(max_num_reqs, torch.int32)
        self.max_chunks = max_chunks
        self.max_rows = max_rows


def build_packed_gqa_context_plan(
    *,
    buffers: PackedGQAContextBuffers,
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu_upper_bound: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_reqs: int,
    num_kv_heads: int,
    group_size: int,
    block_size: int,
    num_sms: int,
) -> PackedGQAContextPlan | None:
    """Return the plan, or None when the batch is better served by one call."""
    qsl = query_start_loc_cpu[: num_reqs + 1].numpy().astype(np.int64)
    q_lens = qsl[1:] - qsl[:-1]
    ctx_ub = seq_lens_cpu_upper_bound[:num_reqs].numpy().astype(np.int64) - q_lens
    if q_lens.max() * group_size > MAX_PACKED_QUERY_ROWS:
        return None
    if ctx_ub.max() < MIN_CONTEXT_LEN:
        return None

    ctx_blocks = -(-np.maximum(ctx_ub, 0) // block_size)
    m_blocks = -(-(q_lens * group_size) // _FA2_BLOCK_M)
    ctas_per_chunk = int((num_kv_heads * m_blocks[ctx_blocks > 0]).sum())
    target_chunks = max(1, min(_MAX_CHUNKS_PER_REQ, 2 * num_sms // ctas_per_chunk))
    chunk_blocks = max(_MIN_CHUNK_BLOCKS, -(-int(ctx_blocks.max()) // target_chunks))

    req_num_chunks = -(-ctx_blocks // chunk_blocks)
    num_chunks = int(req_num_chunks.sum())
    req_first_chunk = np.zeros(num_reqs, dtype=np.int64)
    np.cumsum(req_num_chunks[:-1], out=req_first_chunk[1:])

    chunk_req = np.repeat(np.arange(num_reqs), req_num_chunks)
    chunk_rows = q_lens[chunk_req] * group_size
    num_rows = int(chunk_rows.sum())
    num_tokens = int(qsl[-1])
    if num_chunks > buffers.max_chunks or num_rows > buffers.max_rows:
        return None

    b = buffers
    b.chunk_req.np[:num_chunks] = chunk_req
    b.chunk_start_block.np[:num_chunks] = (
        np.arange(num_chunks) - req_first_chunk[chunk_req]
    ) * chunk_blocks
    b.cu_seqlens_q.np[0] = 0
    np.cumsum(chunk_rows, out=b.cu_seqlens_q.np[1 : num_chunks + 1])
    # Row r of chunk c is packed query row qsl[req] * group + r.
    starts = np.repeat(b.cu_seqlens_q.np[:num_chunks], chunk_rows)
    packed = np.repeat(qsl[chunk_req] * group_size, chunk_rows)
    packed += np.arange(num_rows) - starts
    np.floor_divide(packed, group_size, out=b.row_token.np[:num_rows])
    np.remainder(packed, group_size, out=b.row_member.np[:num_rows])
    b.token_req.np[:num_tokens] = np.repeat(np.arange(num_reqs), q_lens)
    b.req_first_chunk.np[:num_reqs] = req_first_chunk
    b.req_num_chunks.np[:num_reqs] = req_num_chunks

    chunk_req_t = b.chunk_req.copy_to_gpu(num_chunks)
    start_block_t = b.chunk_start_block.copy_to_gpu(num_chunks)
    device = block_table.device
    cols = torch.arange(chunk_blocks, device=device, dtype=torch.int64)
    cols = (start_block_t[:, None] + cols[None, :]).clamp_(max=block_table.shape[1] - 1)
    chunk_block_table = block_table[chunk_req_t[:, None], cols]

    # CPU lengths are only an upper bound; per-chunk KV lengths use device ones.
    q_lens_t = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
    ctx_t = (seq_lens[:num_reqs] - q_lens_t).to(torch.int64)
    chunk_kv_lens = (ctx_t[chunk_req_t] - start_block_t * block_size).clamp_(
        0, chunk_blocks * block_size
    )

    return PackedGQAContextPlan(
        cu_seqlens_q=b.cu_seqlens_q.copy_to_gpu(num_chunks + 1),
        chunk_block_table=chunk_block_table,
        chunk_kv_lens=chunk_kv_lens.to(torch.int32),
        max_chunk_kv_len=chunk_blocks * block_size,
        max_packed_q=int(q_lens.max()) * group_size,
        row_token=b.row_token.copy_to_gpu(num_rows),
        row_member=b.row_member.copy_to_gpu(num_rows),
        token_req=b.token_req.copy_to_gpu(num_tokens),
        req_first_chunk=b.req_first_chunk.copy_to_gpu(num_reqs),
        req_num_chunks=b.req_num_chunks.copy_to_gpu(num_reqs),
    )


def packed_gqa_context_attention(
    flash_attn_varlen_func,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    plan: PackedGQAContextPlan,
    softmax_scale: float,
    fa_version: int,
) -> torch.Tensor:
    """Causal attention of the new tokens over their cached context and themselves."""
    num_tokens, num_heads, head_size = query.shape
    num_kv_heads = key.shape[1]
    group_size = num_heads // num_kv_heads

    # [T, Hk * G, D] -> [rows, Hk, D] with one row per (token, GQA member).
    q_grouped = query.view(num_tokens, num_kv_heads, group_size, head_size)
    q_rows = q_grouped[plan.row_token, :, plan.row_member]
    ctx_out, ctx_lse = flash_attn_varlen_func(
        q=q_rows,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=plan.cu_seqlens_q,
        max_seqlen_q=plan.max_packed_q,
        seqused_k=plan.chunk_kv_lens,
        max_seqlen_k=plan.max_chunk_kv_len,
        softmax_scale=softmax_scale,
        causal=False,
        block_table=plan.chunk_block_table,
        return_softmax_lse=True,
        fa_version=fa_version,
    )

    _, new_lse = flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        out=output,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=max_query_len,
        cu_seqlens_k=query_start_loc,
        max_seqlen_k=max_query_len,
        softmax_scale=softmax_scale,
        causal=True,
        return_softmax_lse=True,
        fa_version=fa_version,
    )

    _merge_chunks_and_new_kernel[(num_tokens, num_heads)](
        output,
        output.stride(0),
        output.stride(1),
        new_lse,
        new_lse.stride(0),
        ctx_out,
        ctx_out.stride(0),
        ctx_out.stride(1),
        ctx_lse,
        ctx_lse.stride(0),
        plan.cu_seqlens_q,
        plan.chunk_kv_lens,
        plan.token_req,
        plan.req_first_chunk,
        plan.req_num_chunks,
        query_start_loc,
        GROUP_SIZE=group_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
    )
    return output


@triton.jit
def _merge_chunks_and_new_kernel(
    out_ptr,  # [T, H, D], holds the new-token result on entry
    out_stride_t,
    out_stride_h,
    new_lse_ptr,  # [H, T] fp32
    new_lse_stride_h,
    ctx_out_ptr,  # [R, Hk, D]
    ctx_out_stride_r,
    ctx_out_stride_h,
    ctx_lse_ptr,  # [Hk, R] fp32
    ctx_lse_stride_h,
    cu_rows_ptr,
    chunk_kv_lens_ptr,
    token_req_ptr,
    req_first_chunk_ptr,
    req_num_chunks_ptr,
    query_start_loc_ptr,
    GROUP_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // GROUP_SIZE
    member = head % GROUP_SIZE

    offs = tl.arange(0, HEAD_SIZE_PADDED)
    mask = offs < HEAD_SIZE

    out_offs = token * out_stride_t + head * out_stride_h + offs
    acc = tl.load(out_ptr + out_offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(new_lse_ptr + head * new_lse_stride_h + token)
    denom = 1.0

    req = tl.load(token_req_ptr + token)
    local_row = (token - tl.load(query_start_loc_ptr + req)) * GROUP_SIZE + member
    first = tl.load(req_first_chunk_ptr + req)
    count = tl.load(req_num_chunks_ptr + req)
    for c in range(0, count):
        chunk = first + c
        # FA2 returns out=0, lse=0 for an empty chunk; it must not enter the merge.
        if tl.load(chunk_kv_lens_ptr + chunk) > 0:
            row = tl.load(cu_rows_ptr + chunk) + local_row
            lse = tl.load(ctx_lse_ptr + kv_head * ctx_lse_stride_h + row)
            part = tl.load(
                ctx_out_ptr
                + row * ctx_out_stride_r
                + kv_head * ctx_out_stride_h
                + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            new_m = tl.maximum(m, lse)
            old_scale = tl.exp(m - new_m)
            part_scale = tl.exp(lse - new_m)
            acc = acc * old_scale + part * part_scale
            denom = denom * old_scale + part_scale
            m = new_m

    tl.store(out_ptr + out_offs, (acc / denom).to(out_ptr.dtype.element_ty), mask=mask)
