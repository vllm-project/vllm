# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA attention of prefill steps with FlashInfer's
QToken-KvBlock-Sparse kernel (sm_100/sm_103, BF16 K/V).

The Triton kernel attends each row on its own, so a K/V load serves only the
query heads of one row. FlashInfer runs GROUP_SIZE consecutive rows of a
request as one tile over the union of their selected blocks, reading each block
once per group.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2, round_up

from ...common.qsa_indexer import _logical_positions

GROUP_SIZE = 8
# FlashInfer 0.7.4 QToken-KvBlock-Sparse limits. A tile holds GROUP_SIZE rows
# times the query heads of one KV head.
_BLOCK_SIZES = (4, 8, 16, 32, 64, 128)
_HEAD_DIMS = (64, 128, 192, 256)
_MAX_BLOCK_TOPK = 512
_MAX_TILE_ROWS = 128


def flashinfer_supported(
    block_size: int, block_topk: int, head_dim: int, num_heads: int, num_kv_heads: int
) -> bool:
    return (
        block_size in _BLOCK_SIZES
        and head_dim in _HEAD_DIMS
        and block_topk <= _MAX_BLOCK_TOPK
        and GROUP_SIZE * (num_heads // num_kv_heads) <= _MAX_TILE_ROWS
    )


@dataclass
class FlashInferBuffers:
    """Storage that FlashInfer plans retain across steps. The metadata builder
    owns it, so prepared plans, which reference the KV cache, go with the cache.
    All plans share one workspace, sized for the largest bucket seen."""

    max_num_reqs: int
    block_size: int  # tokens per selected block (the indexer's compress ratio)
    block_table: torch.Tensor | None = None
    qo_indptr: dict[int, torch.Tensor] = field(default_factory=dict)
    workspace: torch.Tensor | None = None
    plans: dict[tuple[str, int], Any] = field(default_factory=dict)


@dataclass
class FlashInferRoutes:
    """A step's rows padded to `num_rows`, so each layer reuses one plan per
    bucket. Padding rows sit at position 0 of request 0 and select nothing."""

    num_tokens: int
    num_rows: int
    qo_indptr: torch.Tensor  # int32 [routes + 1], groups of one request's rows
    token_to_req: torch.Tensor  # int32 [num_rows]
    positions: torch.Tensor  # int32 [num_rows]
    block_table: torch.Tensor  # int32 [max_num_reqs, blocks]
    buffers: FlashInferBuffers


def build_flashinfer_routes(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    num_tokens: int,
    buffers: FlashInferBuffers,
) -> FlashInferRoutes:
    num_rows = (
        max(64, next_power_of_2(num_tokens))
        if num_tokens <= 1024
        else round_up(num_tokens, 1024)
    )
    device = token_to_req.device
    rows = torch.zeros((2, num_rows), dtype=torch.int32, device=device)
    rows[0, :num_tokens] = token_to_req
    rows[1, :num_tokens] = _logical_positions(
        query_start_loc, seq_lens, token_to_req, num_tokens
    )
    # A route starts every GROUP_SIZE rows of a request (padding is one more).
    row = torch.arange(num_rows, dtype=torch.int32, device=device)
    request_start = torch.full_like(row, num_tokens)
    request_start[:num_tokens] = query_start_loc[token_to_req.long()]
    starts = (row - request_start) % GROUP_SIZE == 0
    route = starts.cumsum(0, dtype=torch.int32) - 1
    qo_indptr = buffers.qo_indptr.get(num_rows)
    if qo_indptr is None:
        # Every route holds a row, and FlashInfer rejects more routes than rows.
        num_routes = min(num_rows, num_rows // GROUP_SIZE + buffers.max_num_reqs + 1)
        qo_indptr = torch.empty(num_routes + 1, dtype=torch.int32, device=device)
        buffers.qo_indptr[num_rows] = qo_indptr
    # Unused routes are empty; the other rows write num_rows to the end.
    qo_indptr.fill_(num_rows)
    qo_indptr.scatter_(
        0,
        torch.where(starts, route, qo_indptr.numel() - 1).long(),
        torch.where(starts, row, num_rows),
    )
    table = buffers.block_table
    if table is None or table.shape[1] != block_table.shape[1]:
        table = block_table.new_zeros((buffers.max_num_reqs, block_table.shape[1]))
        buffers.block_table = table
    table[: block_table.shape[0]] = block_table
    return FlashInferRoutes(
        num_tokens, num_rows, qo_indptr, rows[0], rows[1], table, buffers
    )


@triton.jit
def _sigmoid_gate_kernel(
    attn_ptr,
    gate_ptr,
    out_ptr,
    stride_gate,
    stride_out,
    ROW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    for start in tl.static_range(0, ROW_SIZE, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        valid = cols < ROW_SIZE
        attn = tl.load(attn_ptr + row * ROW_SIZE + cols, mask=valid)
        gate = tl.load(gate_ptr + row * stride_gate + cols, mask=valid)
        gated = attn.to(tl.float32) * tl.sigmoid(gate.to(tl.float32))
        tl.store(
            out_ptr + row * stride_out + cols,
            gated.to(out_ptr.dtype.element_ty),
            mask=valid,
        )


def flashinfer_attention(
    layer_name: str,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected: torch.Tensor,
    routes: FlashInferRoutes,
    scale: float,
    output_gate: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Write sigmoid(output_gate) * attention over each row's selection.

    kv_cache is QSA's [blocks, kv_heads, page_size, 2 * head_dim] view;
    selected holds the indexer's expanded token selections.
    """
    from flashinfer.attention.prims_ts import (
        QTokenKvBlockSparsePagedTSWrapper,
        get_q_token_kv_block_sparse_workspace_size,
    )

    num_tokens, num_rows = query.shape[0], routes.num_rows
    _, num_kv_heads, page_size, row_size = kv_cache.shape
    head_dim = row_size // 2
    block_size = routes.buffers.block_size
    paged_kv = (kv_cache[..., :head_dim], kv_cache[..., head_dim:])
    # Bound the plan by the pages the table can address.
    max_seq_len_kv = routes.block_table.shape[1] * page_size
    block_topk = (selected.shape[1] - 1) // block_size
    q = query.new_empty((num_rows, *query.shape[1:]))
    q[:num_tokens] = query
    # Each complete group's first token names its block; tails are implied.
    block_ids = selected.new_empty((num_rows, block_topk))
    torch.floor_divide(
        selected[:, : block_topk * block_size : block_size],
        block_size,
        out=block_ids[:num_tokens],
    )
    attn = torch.empty_like(q)
    plans = routes.buffers.plans
    plan = plans.get((layer_name, num_rows))
    if plan is None:
        workspace_size = get_q_token_kv_block_sparse_workspace_size(
            q,
            paged_kv[0],
            routes.block_table,
            block_topk=block_topk,
            max_seq_len_kv=max_seq_len_kv,
            qo_indptr=routes.qo_indptr,
            group_size=GROUP_SIZE,
            kv_block_size=block_size,
            split_kv=False,
        )
        workspace = routes.buffers.workspace
        if workspace is None or workspace.numel() < workspace_size:
            # Plans bind views of the workspace, so they are rebuilt on the new one.
            plans.clear()
            workspace = torch.empty(workspace_size, dtype=torch.uint8, device=q.device)
            routes.buffers.workspace = workspace
        plan = QTokenKvBlockSparsePagedTSWrapper()
        plan.plan(
            routes.qo_indptr.numel() - 1,
            q.shape[1],
            num_kv_heads,
            head_dim,
            block_size,
            page_size,
            block_topk,
            max_seq_len_kv,
            device=q.device,
            workspace_buffer=workspace,
            group_size=GROUP_SIZE,
            packed_query=True,
            split_kv=False,
            q_dtype=q.dtype,
        )
        plans[layer_name, num_rows] = plan
    plan.run(
        q,
        paged_kv,
        routes.block_table,
        block_ids,
        routes.token_to_req,
        routes.positions,
        qo_indptr=routes.qo_indptr,
        sm_scale=scale,
        out=attn,
    )
    gate = output_gate.reshape(num_tokens, -1)
    _sigmoid_gate_kernel[(num_tokens,)](
        attn,
        gate,
        out,
        gate.stride(0),
        out.stride(0),
        ROW_SIZE=gate.shape[1],
        BLOCK=2048,
        num_warps=8,
    )
