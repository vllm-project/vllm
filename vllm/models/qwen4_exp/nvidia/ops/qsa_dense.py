# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged causal FA3 for fully visible QSA prefill rows on Hopper."""

from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

if TYPE_CHECKING:
    from ..qsa import QSADensePrefillMetadata


@triton.jit
def _apply_output_gate(
    out,
    gate,
    count,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    OUT_ROW: tl.constexpr,
    OUT_HEAD: tl.constexpr,
    GATE_ROW: tl.constexpr,
    GATE_HEAD: tl.constexpr,
    GATE_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, head, dim = idx // (HEADS * DIM), idx // DIM % HEADS, idx % DIM
    dst = out + row * OUT_ROW + head * OUT_HEAD + dim
    src = gate + row * GATE_ROW + head * GATE_HEAD + dim * GATE_DIM
    x = tl.load(dst, idx < count, other=0).to(tl.float32)
    g = tl.load(src, idx < count, other=0).to(tl.float32)
    tl.store(dst, x * tl.sigmoid(g), idx < count)


def dense_paged_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
    metadata: "QSADensePrefillMetadata",
) -> None:
    first = metadata.first_request
    table = metadata.block_table[first : first + metadata.num_dense_requests]
    flash_attn_varlen_func(
        q,
        k,
        v,
        metadata.max_dense_query_len,
        metadata.cu_query,
        metadata.max_dense_query_len,
        seqused_k=metadata.used_keys,
        block_table=table,
        causal=True,
        out=out,
        fa_version=3,
        num_splits=1,
    )
    gate = gate.view_as(out)
    _apply_output_gate[(triton.cdiv(out.numel(), 512),)](
        out,
        gate,
        out.numel(),
        out.shape[1],
        out.shape[2],
        out.stride(0),
        out.stride(1),
        *gate.stride(),
        512,
    )
