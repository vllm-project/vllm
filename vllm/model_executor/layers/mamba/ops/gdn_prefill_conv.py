# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN prefill: the causal conv1d and the post-conv prep in one kernel.

Replaces `causal_conv1d_fn` followed by `fused_post_conv_prep` on pure-prefill
batches. A warp streams a chunk of one sequence's tokens over one head's
channels and writes that head's final tensor: Q and K L2-normalized, V as is,
and g and beta for V heads. The conv output never goes to HBM. The CUDA kernel
lives in `csrc/mamba/gdn_prefill_conv.cu`; see its header comment for the
numerics contract (g, beta and the conv state follow the stock formulas with
fast __expf/__logf lowerings; q, k, v deviate by ~2 bf16 ulp via FP32 conv
products and a tanh.approx SiLU).

Supported: bf16, head dim 128, width-4 conv, no bias, SiLU; everything else
falls back to the stock two-kernel path in the caller.
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
    """Whether `gdn_prefill_conv` covers this conv: width 4, no bias
    (checked by the caller), bf16 state and weights, head dim 128, and the
    8-byte row alignment the kernel's vectorized loads need."""
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
        and weight.stride(0) == 4  # kernel reads weight rows with stride 4
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
    null_block_id: int | None = NULL_BLOCK_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """SiLU causal conv of the packed (tokens, [Q | K | V]) projection `x`, then
    the post-conv prep; `conv_states` (cache lines, dim, >= 3) is updated in
    place.

    Returns q, k (tokens, H, D), v (tokens, HV, D), and g, beta (tokens, HV)
    in FP32.
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
        -1 if null_block_id is None else null_block_id,
    )
    return q, k, v, g, beta
