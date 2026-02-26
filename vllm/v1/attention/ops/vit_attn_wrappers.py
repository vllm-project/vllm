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
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs = {}
    if is_rocm_aiter:
        from aiter import flash_attn_varlen_func
    else:
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        if not current_platform.is_rocm() and fa_version is not None:
            kwargs["fa_version"] = fa_version

    q_len = q.size(1)
    if cu_seqlens is None:
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=q.device
        )
    max_seqlen = q_len if max_seqlen is None else max_seqlen.item()

    q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        causal=False,
        softmax_scale=scale,
        **kwargs,
    )
    context_layer = einops.rearrange(output, "(b s) h d -> b s h d", b=batch_size)
    return context_layer


def flash_attn_maxseqlen_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
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
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.flash_attn_maxseqlen_wrapper(
        q,
        k,
        v,
        batch_size,
        is_rocm_aiter,
        fa_version,
        scale,
        cu_seqlens,
        max_seqlen,
    )


def triton_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd

    q_len = q.size(1)
    if cu_seqlens is None:
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=q.device
        )
    max_seqlen = q_len if max_seqlen is None else max_seqlen.item()

    q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
    output = torch.empty_like(q)
    context_attention_fwd(
        q,
        k,
        v,
        output,
        b_start_loc=cu_seqlens[:-1],
        b_seq_len=cu_seqlens[1:] - cu_seqlens[:-1],
        max_input_len=max_seqlen,
        is_causal=False,
        sliding_window_q=None,
        sliding_window_k=None,
        softmax_scale=scale,
    )

    context_layer = einops.rearrange(output, "(b s) h d -> b s h d", b=batch_size)
    return context_layer


def triton_attn_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


direct_register_custom_op(
    op_name="triton_attn_wrapper",
    op_func=triton_attn_wrapper,
    fake_impl=triton_attn_wrapper_fake,
)


def vit_triton_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.triton_attn_wrapper(
        q,
        k,
        v,
        batch_size,
        scale,
        cu_seqlens,
        max_seqlen,
    )


def apply_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """
    Input shape:
    (batch_size x seq_len x num_heads x head_size)
    """
    q, k, v = (einops.rearrange(x, "b s h d -> b h s d") for x in [q, k, v])
    output = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
    )
    output = einops.rearrange(output, "b h s d -> b s h d ")
    return output


# TODO: Once we have a torch 2.10, we can use tensor slices
# so we won't need to wrap this in custom ops
def torch_sdpa_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    # Never remove the contiguous logic for ROCm
    # Without it, hallucinations occur with the backend
    if current_platform.is_rocm():
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

    if cu_seqlens is None:
        return apply_sdpa(q, k, v, scale=scale, enable_gqa=enable_gqa)

    outputs = []

    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    q_chunks = torch.split(q, lens, dim=1)
    k_chunks = torch.split(k, lens, dim=1)
    v_chunks = torch.split(v, lens, dim=1)
    for q_i, k_i, v_i in zip(q_chunks, k_chunks, v_chunks):
        output_i = apply_sdpa(q_i, k_i, v_i, scale=scale, enable_gqa=enable_gqa)
        outputs.append(output_i)
    context_layer = torch.cat(outputs, dim=1)
    return context_layer


def torch_sdpa_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None,
    cu_seqlens: torch.Tensor | None,
    enable_gqa: bool = False,
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
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    return torch.ops.vllm.torch_sdpa_wrapper(
        q, k, v, scale, cu_seqlens, enable_gqa=enable_gqa
    )


def flashinfer_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache

    is_reshaped = q.dim() == 4

    if is_reshaped:
        reshape_batch_size = q.shape[0]
        q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

    # cu_seqlens: token-level start offsets (batch_size elements)
    # sequence_lengths: actual token counts per sequence (batch_size elements)
    batch_size = len(cu_seqlens)
    max_seq = max_seqlen.item()
    total_tokens = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]

    # cuDNN SDPA expects dense (batch * max_seq, heads, head_dim) layout
    # with actual_seq_lens indicating real lengths (no batch_offsets needed)
    is_dense = total_tokens == batch_size * max_seq

    if not is_dense:
        # Pad ragged tensors to dense layout
        q_dense = torch.zeros(
            batch_size * max_seq, num_heads, head_dim, dtype=q.dtype, device=q.device
        )
        k_dense = torch.zeros_like(q_dense)
        v_dense = torch.zeros_like(q_dense)
        for i in range(batch_size):
            src_start = cu_seqlens[i].item()
            length = sequence_lengths[i].item()
            dst_start = i * max_seq
            q_dense[dst_start : dst_start + length] = q[src_start : src_start + length]
            k_dense[dst_start : dst_start + length] = k[src_start : src_start + length]
            v_dense[dst_start : dst_start + length] = v[src_start : src_start + length]
        q, k, v = q_dense, k_dense, v_dense

    actual_seq_lens = sequence_lengths.view(-1, 1, 1, 1)

    output, _ = cudnn_batch_prefill_with_kv_cache(
        q,
        k,
        v,
        scale,
        workspace_buffer,
        max_token_per_sequence=max_seq,
        max_sequence_kv=max_seq,
        actual_seq_lens_q=actual_seq_lens,
        actual_seq_lens_kv=actual_seq_lens,
        causal=False,
        return_lse=False,
    )

    if not is_dense:
        # Extract back to ragged layout
        output_ragged = torch.zeros(
            total_tokens, num_heads, head_dim, dtype=output.dtype, device=output.device
        )
        for i in range(batch_size):
            dst_start = cu_seqlens[i].item()
            length = sequence_lengths[i].item()
            src_start = i * max_seq
            output_ragged[dst_start : dst_start + length] = output[
                src_start : src_start + length
            ]
        output = output_ragged

    if is_reshaped:
        output = einops.rearrange(output, "(b s) h d -> b s h d", b=reshape_batch_size)

    return output


def vit_flashinfer_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


direct_register_custom_op(
    op_name="flashinfer_wrapper",
    op_func=flashinfer_wrapper,
    fake_impl=vit_flashinfer_wrapper_fake,
)


def vit_flashinfer_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.flashinfer_wrapper(
        q, k, v, scale, workspace_buffer, cu_seqlens, max_seqlen, sequence_lengths
    )
