# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused GDN prefill causal conv1d + post-conv prep (csrc/mamba/gdn_prefill_conv.cu).

Covers bf16, head dim 128, width-4 conv without bias, and SiLU; callers fall
back to `causal_conv1d_fn` + `fused_post_conv_prep` otherwise.
"""

import torch

from vllm._custom_ops import gdn_prefill_conv as _gdn_conv_cpp
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

_STATE_LEN = 3  # width-4 taps


def gdn_prefill_conv_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    head_k_dim: int,
    head_v_dim: int,
) -> bool:
    """Whether the kernel covers these tensors. Bias and activation are checked
    by the caller."""
    return (
        current_platform.is_cuda()
        # older precompiled extensions may lack the op
        and hasattr(torch.ops._C, "gdn_prefill_conv")
        and x.is_cuda
        and x.dim() == 2
        and x.stride(1) == 1
        and x.dtype == torch.bfloat16
        and conv_states.dtype == x.dtype
        and weight.dtype == x.dtype
        and weight.dim() == 2
        and weight.size(0) == x.size(1)
        and weight.size(1) == _STATE_LEN + 1
        and conv_states.dim() == 3
        and conv_states.size(1) == x.size(1)
        and conv_states.size(2) >= _STATE_LEN
        and head_k_dim == head_v_dim == 128
        and x.size(1) % head_k_dim == 0
        and weight.stride(1) == 1
        and weight.stride(0) == 4
        and (x.stride(0) * x.element_size()) % 8 == 0
        and x.data_ptr() % 8 == 0  # vectorized 8B loads
    )


def gdn_prefill_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_dim: int,
    output_g_exp: bool = False,
    beta_scale: float = 1.0,
    null_block_id: int | None = NULL_BLOCK_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """SiLU causal conv of `x` (tokens, [Q | K | V]), then the post-conv prep.

    Updates `conv_states` in place. Returns q, k (tokens, H, D), v
    (tokens, HV, D), and fp32 g, beta (tokens, HV).
    """
    num_tokens = x.size(0)
    H, HV, D = num_k_heads, A_log.size(0), head_dim
    assert x.size(1) == (2 * H + HV) * D
    assert A_log.dtype == torch.float32
    assert dt_bias.dtype in (torch.bfloat16, torch.float32)
    q = torch.empty(num_tokens, H, D, dtype=x.dtype, device=x.device)
    k = torch.empty(num_tokens, H, D, dtype=x.dtype, device=x.device)
    v = torch.empty(num_tokens, HV, D, dtype=x.dtype, device=x.device)
    g = torch.empty(num_tokens, HV, dtype=torch.float32, device=x.device)
    beta = torch.empty(num_tokens, HV, dtype=torch.float32, device=x.device)
    num_seqs = query_start_loc.size(0) - 1
    if num_tokens == 0 or num_seqs == 0:
        return q, k, v, g, beta
    _gdn_conv_cpp(
        x,
        weight,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        a,
        b,
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        num_k_heads,
        head_dim,
        output_g_exp,
        beta_scale,
        -1 if null_block_id is None else null_block_id,
    )
    return q, k, v, g, beta
