# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains ops for ViT attention to be compatible with torch.compile
as there are operations here not supported by torch.compile (for instance,
`to_list` in xformers attn, or `.item()` in flash attention)

Using these ops and wrapping vision blocks with `torch.compile` can speed up
throughput in vision models by ~5% relative on H100, and improve token
latencies by ~7% (see qwen2_5_vl for example usage)

To use these ops, you must have a recent version of PyTorch installed (>= 2.4.0)
"""

import einops
import torch
import torch.nn.functional as F

from vllm.utils.torch_utils import direct_register_custom_op


def xformers_attn_seqlens_wrapper(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqlens: torch.Tensor
) -> torch.Tensor:
    from xformers import ops as xops
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask

    attn_bias = BlockDiagonalMask.from_seqlens(
        q_seqlen=seqlens.tolist(), kv_seqlen=None, device=q.device
    )
    context_layer = xops.memory_efficient_attention_forward(
        q, k, v, attn_bias=attn_bias, p=0, scale=None
    )
    context_layer = einops.rearrange(context_layer, "b s h d -> s b (h d)").contiguous()
    return context_layer


def xformers_attn_seqlens_wrapper_fake(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqlens: torch.Tensor
) -> torch.Tensor:
    b, s, h, d = q.shape
    return torch.empty((s, b, h * d), dtype=q.dtype, device=q.device)


direct_register_custom_op(
    op_name="xformers_attn_seqlens_wrapper",
    op_func=xformers_attn_seqlens_wrapper,
    fake_impl=xformers_attn_seqlens_wrapper_fake,
)


def vit_xformers_attn_wrapper(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqlens: torch.Tensor
) -> torch.Tensor:
    return torch.ops.vllm.xformers_attn_seqlens_wrapper(q, k, v, seqlens)


def flash_attn_maxseqlen_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    use_upstream_fa: bool,
) -> torch.Tensor:
    if is_rocm_aiter:
        from aiter import flash_attn_varlen_func
    else:
        if use_upstream_fa:
            from flash_attn import flash_attn_varlen_func
        else:
            from vllm.attention.utils.fa_utils import flash_attn_varlen_func
    q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen.item(),
        max_seqlen_k=max_seqlen.item(),
        dropout_p=0.0,
        causal=False,
    )
    context_layer = einops.rearrange(
        output, "(b s) h d -> s b (h d)", b=batch_size
    ).contiguous()
    return context_layer


def flash_attn_maxseqlen_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    use_upstream_fa: bool,
) -> torch.Tensor:
    b, s, h, d = q.shape
    return torch.empty((s, b, h * d), dtype=q.dtype, device=q.device)


direct_register_custom_op(
    op_name="flash_attn_maxseqlen_wrapper",
    op_func=flash_attn_maxseqlen_wrapper,
    fake_impl=flash_attn_maxseqlen_wrapper_fake,
)


def vit_flash_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    use_upstream_fa: bool,
) -> torch.Tensor:
    return torch.ops.vllm.flash_attn_maxseqlen_wrapper(
        q, k, v, cu_seqlens, max_seqlen, batch_size, is_rocm_aiter, use_upstream_fa
    )


# TODO: Once we have a torch 2.10, we can use tensor slices
# so we won't need to wrap this in custom ops
def torch_sdpa_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    for i in range(1, len(cu_seqlens)):
        start_idx = cu_seqlens[i - 1]
        end_idx = cu_seqlens[i]
        q_i = q[:, start_idx:end_idx]
        k_i = k[:, start_idx:end_idx]
        v_i = v[:, start_idx:end_idx]
        q_i, k_i, v_i = (
            einops.rearrange(x, "b s h d -> b h s d") for x in [q_i, k_i, v_i]
        )
        output_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0)
        output_i = einops.rearrange(output_i, "b h s d -> b s h d ")
        outputs.append(output_i)
    context_layer = torch.cat(outputs, dim=1)
    context_layer = einops.rearrange(context_layer, "b s h d -> s b (h d)").contiguous()
    return context_layer


def torch_sdpa_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    b, s, h, d = q.shape
    return torch.empty((s, b, h * d), dtype=q.dtype, device=q.device)


direct_register_custom_op(
    op_name="torch_sdpa_wrapper",
    op_func=torch_sdpa_wrapper,
    fake_impl=torch_sdpa_wrapper_fake,
)


def vit_torch_sdpa_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.vllm.torch_sdpa_wrapper(q, k, v, cu_seqlens)
