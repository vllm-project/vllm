# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from ..op import register_op


@register_op
def mm_encoder_attn(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
    cu_seqlens: Tensor | None = None,
    max_seqlen: int | None = None,
) -> Tensor:
    """Multi-head encoder attention without KV cache (for multimodal encoders).

    Inputs are 4D: (batch_size, seq_len, num_heads, head_size).
    """
    enable_gqa = query.size(2) != key.size(2)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    if cu_seqlens is not None:
        outputs = []
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        for q_i, k_i, v_i in zip(
            torch.split(q, lens, dim=2),
            torch.split(k, lens, dim=2),
            torch.split(v, lens, dim=2),
        ):
            out_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
            )
            outputs.append(out_i)
        output = torch.cat(outputs, dim=2)
    else:
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
        )

    return output.transpose(1, 2)


@mm_encoder_attn.register_fake
def _mm_encoder_attn_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
    cu_seqlens: Tensor | None = None,
    max_seqlen: int | None = None,
) -> Tensor:
    return torch.empty_like(query)


@mm_encoder_attn.register_input_generator
def _mm_encoder_attn_input_generator(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    scale: float | None = None,
) -> tuple:
    if scale is None:
        scale = 1.0 / (head_size**0.5)
    query = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)
    key = torch.randn(batch_size, seq_len, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(batch_size, seq_len, num_kv_heads, head_size, dtype=dtype)
    return query, key, value, scale


mm_encoder_attn.override_tolerance(torch.float16, atol=1e-2, rtol=1e-2)
mm_encoder_attn.override_tolerance(torch.bfloat16, atol=1e-2, rtol=1e-2)
