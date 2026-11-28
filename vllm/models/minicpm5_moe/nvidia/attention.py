# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention output gating for MiniCPM5's interleaved Q/gate projection."""

import torch

from vllm import ir
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _attention_gate_kernel(
    OUT,
    GATE,
    RESULT,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    OUT_STRIDE_0: tl.constexpr,
    OUT_STRIDE_1: tl.constexpr,
    GATE_STRIDE_0: tl.constexpr,
    GATE_STRIDE_1: tl.constexpr,
    GATE_STRIDE_2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, col = index // WIDTH, index % WIDTH
    gate = tl.load(
        GATE
        + token * GATE_STRIDE_0
        + (col // HEAD_DIM) * GATE_STRIDE_1
        + (col % HEAD_DIM) * GATE_STRIDE_2,
        index < N,
        other=0,
    )
    out = tl.load(
        OUT + token * OUT_STRIDE_0 + col * OUT_STRIDE_1,
        index < N,
        other=0,
    )
    # Preserve the reference's low-precision sigmoid rounding before multiply.
    sigmoid = tl.div_rn(1.0, 1.0 + tldevice.exp(-gate.to(tl.float32)))
    sigmoid = sigmoid.to(gate.dtype).to(tl.float32)
    tl.store(RESULT + index, out.to(tl.float32) * sigmoid, index < N)


def apply_attention_gate(out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Multiply [tokens, heads * dim] output by a strided [tokens, heads, dim] gate."""
    if not (
        current_platform.is_cuda()
        and out.is_cuda
        and out.dtype in (torch.bfloat16, torch.float16)
        and gate.dtype == out.dtype
    ):
        return out * gate.reshape(out.shape).sigmoid()
    result = torch.empty_like(out, memory_format=torch.contiguous_format)
    if out.numel():
        _attention_gate_kernel[(triton.cdiv(out.numel(), 1024),)](
            out,
            gate,
            result,
            out.numel(),
            out.shape[1],
            gate.shape[2],
            *out.stride(),
            *gate.stride(),
            BLOCK=1024,
        )
    return result


def apply_interleaved_rope(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_emb: RotaryEmbedding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate strided Q directly into contiguous output, avoiding a Q copy."""
    if (
        current_platform.is_cuda()
        and type(rotary_emb) is RotaryEmbedding
        and not rotary_emb.use_flashinfer
    ):
        cache = rotary_emb._match_cos_sin_cache_dtype(query)
        query, key = ir.ops.rotary_embedding(
            positions,
            query,
            key,
            rotary_emb.head_size,
            cache,
            rotary_emb.is_neox_style,
        )
        return query.flatten(1), key
    return rotary_emb(positions, query.flatten(1), key)
