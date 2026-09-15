# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused rotary embeddings for vision attention."""

# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/vision_rope.py

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_pairs"])
def _fused_qk_complex_rope_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr,
    q_out_ptr,
    k_out_ptr,
    num_pairs,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    q_stride_token: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_dim: tl.constexpr,
    freq_stride_token: tl.constexpr,
    freq_stride_pair: tl.constexpr,
    freq_stride_complex: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pair_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pair_offsets < num_pairs
    pairs_per_head = head_dim // 2
    row = pair_offsets // pairs_per_head
    pair = pair_offsets - row * pairs_per_head
    token = row // num_heads
    head = row - token * num_heads

    q_base = token * q_stride_token + head * q_stride_head
    q_base += pair * 2 * q_stride_dim
    k_base = token * k_stride_token + head * k_stride_head
    k_base += pair * 2 * k_stride_dim
    freq_base = token * freq_stride_token + pair * freq_stride_pair

    cos = tl.load(freqs_ptr + freq_base, mask=mask).to(tl.float32)
    sin = tl.load(freqs_ptr + freq_base + freq_stride_complex, mask=mask).to(tl.float32)
    q_real = tl.load(q_ptr + q_base, mask=mask).to(tl.float32)
    q_imag = tl.load(q_ptr + q_base + q_stride_dim, mask=mask).to(tl.float32)
    k_real = tl.load(k_ptr + k_base, mask=mask).to(tl.float32)
    k_imag = tl.load(k_ptr + k_base + k_stride_dim, mask=mask).to(tl.float32)

    out_base = row * head_dim + pair * 2
    q_real_cos = q_real * cos
    q_imag_sin = q_imag * sin
    k_real_cos = k_real * cos
    k_imag_sin = k_imag * sin
    tl.store(
        q_out_ptr + out_base,
        q_real_cos - q_imag_sin,
        mask=mask,
    )
    tl.store(
        q_out_ptr + out_base + 1,
        q_imag * cos + q_real * sin,
        mask=mask,
    )
    tl.store(
        k_out_ptr + out_base,
        k_real_cos - k_imag_sin,
        mask=mask,
    )
    tl.store(
        k_out_ptr + out_base + 1,
        k_imag * cos + k_real * sin,
        mask=mask,
    )


def can_use_fused_qk_complex_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> bool:
    """Return whether the fused vision RoPE kernel supports the inputs."""
    if not current_platform.is_cuda():
        return False
    if not (
        query.is_cuda
        and key.is_cuda
        and freqs_cis.is_cuda
        and query.device == key.device == freqs_cis.device
    ):
        return False
    if query.dtype != key.dtype or query.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        return False
    if query.shape != key.shape or query.ndim != 3:
        return False
    if freqs_cis.dtype != torch.complex64 or query.numel() == 0 or query.shape[-1] % 2:
        return False
    if freqs_cis.shape != (query.shape[0], query.shape[-1] // 2):
        return False
    device_id = query.device.index or 0
    return current_platform.has_device_capability(90, device_id=device_id)


def apply_fused_qk_complex_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate strided Q/K views together and return contiguous outputs."""
    if not can_use_fused_qk_complex_rope(query, key, freqs_cis):
        raise ValueError(
            "Unsupported fused vision RoPE inputs: "
            f"query={query.shape}/{query.dtype}/{query.device}, "
            f"key={key.shape}/{key.dtype}/{key.device}, "
            f"freqs={freqs_cis.shape}/{freqs_cis.dtype}/{freqs_cis.device}"
        )

    query_out = torch.empty_like(query, memory_format=torch.contiguous_format)
    key_out = torch.empty_like(key, memory_format=torch.contiguous_format)
    freqs = torch.view_as_real(freqs_cis)

    block_size = 128
    num_pairs = query.numel() // 2
    _fused_qk_complex_rope_kernel[(triton.cdiv(num_pairs, block_size),)](
        query,
        key,
        freqs,
        query_out,
        key_out,
        num_pairs,
        query.shape[1],
        query.shape[2],
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        freqs.stride(0),
        freqs.stride(1),
        freqs.stride(2),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return query_out, key_out
