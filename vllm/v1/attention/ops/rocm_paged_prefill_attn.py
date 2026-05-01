# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HIP/WMMA paged prefill attention for AMD RDNA3 (gfx1100).

Drop-in replacement for the Triton ``context_attention_fwd`` path
(in :mod:`vllm.v1.attention.ops.prefix_prefill`) that the ROCM_ATTN
backend uses when ``max_query_len > 1``. Built on top of the
``paged_prefill_attn_rdna3`` C++ op registered by
``csrc/attention/paged_prefill_attn_rdna3.cu``.

Falls through gracefully when the op is unavailable (e.g. partial
rebuild, non-RDNA3 ROCm device, non-ROCm build); callers must check
:func:`is_available` and provide their own fallback.
"""

import torch

from vllm.platforms import current_platform

# Hard requirement of the v1 kernel. Models with other head sizes (96,
# 192, ...) fall back to the Triton path.
SUPPORTED_HEAD_SIZE = 128

# Kernel internal K-tile; block_size of the KV cache must be a multiple
# of this to guarantee a K-tile never straddles a physical block (the
# load path assumes single-block addressing per K-tile).
KERNEL_K_TILE = 16


def is_available() -> bool:
    """Whether the RDNA3 prefill op is registered AND we're on gfx11.

    The op may be absent when:
      * the user runs a partial-rebuild without recompiling the C++ ext,
      * the user is on a non-RDNA3 ROCm device (CDNA / MI300),
      * the user is on a non-ROCm build (CUDA / CPU).
    """
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import _GCN_ARCH
    except ImportError:
        return False
    # gfx11xx (RDNA3 / RDNA3.5).  Substring match mirrors the pattern used
    # by ``get_split_k_arch_config`` and avoids depending on an
    # ``on_gfx11()`` helper that is not present on every vLLM revision.
    if "gfx11" not in _GCN_ARCH:
        return False
    return hasattr(torch.ops._C, "paged_prefill_attn_rdna3")


def supports_shape(
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
) -> bool:
    """Whether the v1 kernel handles a given configuration."""
    if head_size != SUPPORTED_HEAD_SIZE:
        return False
    if block_size % KERNEL_K_TILE != 0:
        return False
    return dtype in (torch.float16, torch.bfloat16)


def paged_prefill_attn_rdna3(
    out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    sm_scale: float,
    causal: bool = True,
) -> None:
    """Run the RDNA3 paged prefill attention kernel in place into ``out``.

    Args:
        out: ``[num_actual_tokens, num_query_heads, head_size]``,
            same dtype as ``query``. Written in-place.
        query: ``[num_actual_tokens, num_query_heads, head_size]``.
        key: current-chunk K
            ``[num_actual_tokens, num_kv_heads, head_size]``.
        value: current-chunk V, same shape as ``key``.
        key_cache: 5-D
            ``[num_blocks, num_kv_heads, head_size // x, block_size, x]``
            from ``PagedAttention.split_kv_cache``.
        value_cache: 4-D
            ``[num_blocks, num_kv_heads, head_size, block_size]``.
        block_table: ``[num_seqs, max_blocks_per_seq]``, int32.
        cu_seqlens_q: ``[num_seqs + 1]`` cumulative query lengths, int32.
        seq_lens: ``[num_seqs]`` total sequence lengths (ctx + chunk),
            int32.
        max_query_len: largest per-sequence query length in the batch.
            Passed in from the metadata builder as a Python int so the
            C++ entry point doesn't have to sync ``cu_seqlens_q`` to CPU
            once per call.
        sm_scale: softmax scale (typically ``1 / sqrt(head_size)``).
        causal: only ``True`` is supported in v1.
    """
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if cu_seqlens_q.dtype != torch.int32:
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)

    torch.ops._C.paged_prefill_attn_rdna3(
        out,
        query,
        key,
        value,
        key_cache,
        value_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        int(max_query_len),
        float(sm_scale),
        bool(causal),
    )
