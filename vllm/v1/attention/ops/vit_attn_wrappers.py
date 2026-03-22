# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ViT 注意力操作封装模块。

本文件包含 ViT（Vision Transformer）注意力操作，与 torch.compile 兼容。
由于某些操作不被 torch.compile 支持（例如 flash attention 中的 `.item()`），
使用这些操作并将 vision 块用 `torch.compile` 包装可以在 H100 上提升
约 5% 的吞吐量，并改善约 7% 的 token 延迟。

要使用这些操作，需要较新版本的 PyTorch（>= 2.4.0）。

主要函数：
- flash_attn_maxseqlen_wrapper: Flash Attention 包装函数
- vit_flash_attn_wrapper: ViT Flash Attention 封装
- triton_attn_wrapper: Triton 注意力包装函数
- vit_triton_attn_wrapper: ViT Triton 注意力封装
- torch_sdpa_wrapper: PyTorch SDPA 包装函数
- vit_torch_sdpa_wrapper: ViT PyTorch SDPA 封装
- flashinfer_wrapper: FlashInfer 包装函数
- vit_flashinfer_wrapper: ViT FlashInfer 封装
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
    """Flash Attention 可变长度包装函数。

    支持 ROCm Aiter 和 CUDA Flash Attention 后端。

    Args:
        q: Query 张量
        k: Key 张量
        v: Value 张量
        batch_size: 批次大小
        is_rocm_aiter: 是否使用 ROCm Aiter
        fa_version: Flash Attention 版本
        scale: 缩放因子（可选）
        cu_seqlens: 累积序列长度（可选）
        max_seqlen: 最大序列长度（可选）

    Returns:
        注意力输出张量
    """
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
    """ViT Flash Attention 封装函数。

    调用已注册的自定义操作执行 Flash Attention。

    Args:
        q: Query 张量
        k: Key 张量
        v: Value 张量
        batch_size: 批次大小
        is_rocm_aiter: 是否使用 ROCm Aiter
        fa_version: Flash Attention 版本
        scale: 缩放因子（可选）
        cu_seqlens: 累积序列长度（可选）
        max_seqlen: 最大序列长度（可选）

    Returns:
        注意力输出张量
    """
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
    """Triton 注意力包装函数。

    使用 Triton kernel 执行 ViT 注意力计算。

    Args:
        q: Query 张量
        k: Key 张量
        v: Value 张量
        batch_size: 批次大小
        scale: 缩放因子（可选）
        cu_seqlens: 累积序列长度（可选）
        max_seqlen: 最大序列长度（可选）

    Returns:
        注意力输出张量
    """
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
    """应用 PyTorch 缩放点积注意力。

    Args:
        q: Query 张量 (batch_size x seq_len x num_heads x head_size)
        k: Key 张量
        v: Value 张量
        scale: 缩放因子（可选）
        enable_gqa: 是否启用分组查询注意力

    Returns:
        注意力输出张量
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
    """PyTorch SDPA 包装函数。

    支持可变长度序列和分组查询注意力。

    Args:
        q: Query 张量
        k: Key 张量
        v: Value 张量
        scale: 缩放因子（可选）
        cu_seqlens: 累积序列长度（可选）
        enable_gqa: 是否启用分组查询注意力

    Returns:
        注意力输出张量
    """
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
    """ViT PyTorch SDPA 封装函数。

    调用已注册的自定义操作执行 PyTorch SDPA。

    Args:
        q: Query 张量
        k: Key 张量
        v: Value 张量
        scale: 缩放因子（可选）
        cu_seqlens: 累积序列长度（可选）
        enable_gqa: 是否启用分组查询注意力

    Returns:
        注意力输出张量
    """
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
    """FlashInfer 包装函数。

    使用 FlashInfer 的 cudnn_batch_prefill_with_kv_cache 执行注意力计算。

    Args:
        q: Query 张量
        k: Key 张量
        v: Value 张量
        scale: 缩放因子
        workspace_buffer: 工作区缓冲区
        cu_seqlens: 累积序列长度（可选）
        max_seqlen: 最大序列长度（可选）
        sequence_lengths: 序列长度（可选）

    Returns:
        注意力输出张量
    """
    from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache

    is_reshaped = q.dim() == 4

    if is_reshaped:
        reshape_batch_size = q.shape[0]
        q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
    # cuDNN <= 9.10.2.21 requires q, k to be contiguous
    # this comes with no cost for ViTs with RoPE because
    # RoPE has already made q and k contiguous.
    q, k = q.contiguous(), k.contiguous()

    assert len(cu_seqlens) % 2 == 0, "cu_seqlens must be divisible by 2"
    cu_seqlength = len(cu_seqlens) // 2
    batch_offsets_qko = cu_seqlens[:cu_seqlength].view(-1, 1, 1, 1)
    batch_offsets_v = cu_seqlens[cu_seqlength:].view(-1, 1, 1, 1)
    sequence_lengths = sequence_lengths.view(-1, 1, 1, 1)
    max_seqlen = max_seqlen.item()

    output, _ = cudnn_batch_prefill_with_kv_cache(
        q,
        k,
        v,
        scale,
        workspace_buffer,
        max_token_per_sequence=max_seqlen,
        max_sequence_kv=max_seqlen,
        actual_seq_lens_q=sequence_lengths,
        actual_seq_lens_kv=sequence_lengths,
        causal=False,
        return_lse=False,
        batch_offsets_q=batch_offsets_qko,
        batch_offsets_k=batch_offsets_qko,
        batch_offsets_v=batch_offsets_v,
        batch_offsets_o=batch_offsets_qko,
    )

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
    """ViT FlashInfer 封装函数。

    调用已注册的自定义操作执行 FlashInfer 注意力计算。

    Args:
        q: Query 张量
        k: Key 张量
        v: Value 张量
        scale: 缩放因子
        workspace_buffer: 工作区缓冲区
        cu_seqlens: 累积序列长度（可选）
        max_seqlen: 最大序列长度（可选）
        sequence_lengths: 序列长度（可选）

    Returns:
        注意力输出张量
    """
    return torch.ops.vllm.flashinfer_wrapper(
        q, k, v, scale, workspace_buffer, cu_seqlens, max_seqlen, sequence_lengths
    )
