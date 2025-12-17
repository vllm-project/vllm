# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains ops for ViT attention to be compatible with torch.compile
as there are operations here not supported by torch.compile (for instance,
`.item()` in flash attention)

Using these ops and wrapping vision blocks with `torch.compile` can speed up
throughput in vision models by ~5% relative on H100, and improve token
latencies by ~7% (see qwen2_5_vl for example usage)

To use these ops, you must have a recent version of PyTorch installed (>= 2.4.0)
"""

import einops
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def flash_attn_maxseqlen_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int,
) -> torch.Tensor:
    kwargs = {}
    if is_rocm_aiter:
        from aiter import flash_attn_varlen_func
    else:
        from vllm.attention.utils.fa_utils import flash_attn_varlen_func

        kwargs["fa_version"] = fa_version
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
        **kwargs,
    )
    context_layer = einops.rearrange(output, "(b s) h d -> b s h d", b=batch_size)
    return context_layer


def flash_attn_maxseqlen_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int,
) -> torch.Tensor:
    return torch.empty_like(q)


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
    fa_version: int,
) -> torch.Tensor:
    return torch.ops.vllm.flash_attn_maxseqlen_wrapper(
        q, k, v, cu_seqlens, max_seqlen, batch_size, is_rocm_aiter, fa_version
    )


# TODO: Once we have a torch 2.10, we can use tensor slices
# so we won't need to wrap this in custom ops
def torch_sdpa_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    # Never remove the contiguous logic for ROCm
    # Without it, hallucinations occur with the backend
    if current_platform.is_rocm():
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

    outputs = []

    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    q_chunks = torch.split(q, lens, dim=1)
    k_chunks = torch.split(k, lens, dim=1)
    v_chunks = torch.split(v, lens, dim=1)
    for q_i, k_i, v_i in zip(q_chunks, k_chunks, v_chunks):
        q_i, k_i, v_i = (
            einops.rearrange(x, "b s h d -> b h s d") for x in [q_i, k_i, v_i]
        )
        output_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0)
        output_i = einops.rearrange(output_i, "b h s d -> b s h d ")
        outputs.append(output_i)
    context_layer = torch.cat(outputs, dim=1)
    return context_layer


def torch_sdpa_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q)


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
