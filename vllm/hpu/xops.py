###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import habana_frameworks.torch as htorch
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from .attn_bias import AttentionBias, BlockDiagonalCausalMask

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None

def memory_efficient_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    assert attn_bias is not None, "Attention mask is required for prompt processing"
    dim = query.dim()
    is_causal = isinstance(attn_bias, BlockDiagonalCausalMask)
    if FusedSDPA and (is_causal or attn_bias is None):
        bs = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_kv = key.shape[1]
        heads = query.shape[-2] if dim != 5 else query.shape[-3]
        attn_groups = 1 if dim != 5 else query.shape[-2]
        head_dim = query.shape[-1]
        if dim == 4:
            # [bs, seq_len, 1, heads, head_dim] -> [bs, heads, seq_len, head_dim]
            query = query.reshape(bs, seq_len_q, heads, head_dim).permute(0, 2, 1, 3)
            key = key.reshape(bs, seq_len_kv, heads, head_dim).permute(0, 2, 1, 3)
            value = value.reshape(bs, seq_len_kv, heads, head_dim).permute(0, 2, 1, 3)
        elif dim == 5:
            # [bs, seq_len, heads, attn_groups, head_dim] -> [bs, heads, attn_groups, seq_len, head_dim]
            query = query.reshape(bs, seq_len_q, heads, attn_groups, head_dim).permute(0, 2, 3, 1, 4) 
            key = key.reshape(bs, seq_len_kv, heads, attn_groups, head_dim).permute(0, 2, 3, 1, 4) 
            value = value.reshape(bs, seq_len_kv, heads, attn_groups, head_dim).permute(0, 2, 3, 1, 4) 
        else:
            raise ValueError(f"Unsupported attention dimension: {dim}")

        import habana_frameworks.torch.hpu as ht
        with ht.sdp_kernel(enable_recompute=False):  # (flash_attention_recompute and q_len == 1)):
            out = FusedSDPA.apply(
                query, key, value, None, p, is_causal, scale
            )
        htorch.core.mark_step()
        if dim == 4:
            # [bs, heads, seq_len, head_dim] -> [bs, seq_len, heads, head_dim]
            out = out.permute(0, 2, 1, 3).reshape(bs, seq_len_q, heads, head_dim)
        elif dim == 5:
            # [bs, heads, attn_groups, seq_len, head_dim] -> [bs, seq_len, heads, attn_groups, head_dim] 
            out = out.permute(0, 3, 1, 2, 4).reshape(bs, seq_len_q, heads, attn_groups, head_dim)
    else:
       raise NotImplementedError(f'Only FusedSDPA causal or non-masked attention is supported.\nFusedSDPA support: {FusedSDPA is not None}\nis_causal: {is_causal}\nmask_present: {attn_bias is not None}')

    return out
