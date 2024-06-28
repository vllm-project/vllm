###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import torch
from typing import Optional

import vllm.hpu.utils


def prompt_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        p: float = 0.0,
        scale: Optional[float] = None,
        qk_matmul_op=torch.matmul,
        softmax_op=torch.softmax,
        kv_matmul_op=torch.matmul,
) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query_heads = query.size(1)
    kv_heads = key.size(1)
    if query_heads != kv_heads:
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        attn_bias = attn_bias.unsqueeze(2)
    attn_weights = qk_matmul_op(query * scale, key.transpose(-1, -2))
    if attn_bias is not None:
        attn_weights.add_(attn_bias)
    attn_weights = softmax_op(attn_weights, dim=-1)
    attn_weights = kv_matmul_op(attn_weights, value)
    if query_heads != kv_heads:
        attn_weights = attn_weights.flatten(1, 2)
    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights
