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


def block_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    query = query * scale
    attn = query.transpose(0,1) @ key.transpose(0, 1).transpose(1, 2)
    if attn_mask is not None:
        attn = attn + attn_mask.to(attn.dtype)
    attn = attn.softmax(-1)
    out = attn @ value.transpose(0, 1)
    out = out.transpose(0, 1)
    return out


def memory_efficient_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_lens: List[int],
    attn_bias: Optional[torch.Tensor] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    assert attn_bias is not None, "Attention mask is required for prompt processing"
    dim = query.dim()
    if FusedSDPA and isinstance(attn_bias, BlockDiagonalCausalMask):
        _, _, heads, head_dim = query.shape
        bs = len(cu_seq_lens) - 1
        seq_len_q = query.shape[1] // bs
        seq_len_kv = key.shape[1] // bs
        query = query.reshape(bs, seq_len_q, heads, head_dim).permute(0, 2, 1, 3)
        key = key.reshape(bs, seq_len_kv, heads, head_dim).permute(0, 2, 1, 3)
        value = value.reshape(bs, seq_len_kv, heads, head_dim).permute(0, 2, 1, 3)
        attn_mask = torch.ones((bs, heads, seq_len_q, seq_len_kv), device=query.device, dtype=torch.bool).tril()
        import habana_frameworks.torch.hpu as ht
        with ht.sdp_kernel(enable_recompute=False):  # (flash_attention_recompute and q_len == 1)):
            out = FusedSDPA.apply(
                query, key, value, attn_mask, 0.0, False, None
            )
        htorch.core.mark_step()
        out = out.permute(0, 2, 1, 3).reshape(bs * seq_len_q, heads, head_dim)
    else:
        if dim == 4:
            query, key, value = query.squeeze(0), key.squeeze(0), value.squeeze(0)
        num_seqs = len(cu_seq_lens) - 1
        outputs = []
        for i in range(num_seqs):
            start_idx = cu_seq_lens[i]
            end_idx = cu_seq_lens[i + 1]
            seq_len = end_idx - start_idx
            mask_start_idx = i * seq_len
            mask_end_idx = (i + 1) * seq_len

            # Create attention mask.
            attn_mask = attn_bias.materialize(device=query.device)
            output = block_masked_attention(
                query[start_idx:end_idx],
                key[start_idx:end_idx],
                value[start_idx:end_idx],
                scale,
                attn_mask=attn_mask[mask_start_idx:mask_end_idx,
                                    mask_start_idx:mask_end_idx],
            )
            outputs.append(output)
        out = torch.cat(outputs, dim=0)
    if dim == 4:
        out = out.unsqueeze(0)
    return out
