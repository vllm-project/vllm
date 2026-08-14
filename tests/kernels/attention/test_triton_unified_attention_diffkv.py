# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the Triton DiffKV unified-attention kernel.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    set_random_seed,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)

DEVICE_TYPE = current_platform.device_type

# (num_query_heads, num_kv_heads): MHA, GQA, and the num_kv_heads==1
# (degenerate-stride) case.
NUM_HEADS = [(4, 4), (8, 2), (5, 1)]
# (head_size_qk, head_size_v).  (192, 128) is the canonical asymmetric
# DiffKV shape; FA4 on Blackwell only supports head_size>128 when it is
# 192, and FA3 on Hopper supports it too -- so this pair is runnable on
# both.  (128, 128) keeps the equal-dim path covered through the DiffKV
# kernel.
HEAD_SIZES = [(128, 128), (192, 128)]
BLOCK_SIZES = [16]
DTYPES = [torch.bfloat16]

NUM_BLOCKS = 2048

# 0: 2D decode kernel; 8: 3D (split-KV) decode kernel.
SEQ_THRESHOLD_3D_VALUES = [0, 8]

NUM_PAR_SOFTMAX_SEGMENTS = 16


def _alloc_segm_buffers(seq_threshold_3D: int, num_query_heads: int, head_size_v: int):
    """Allocate the split-KV softmax scratch (last dim == head_size_v)."""
    head_size_v_padded = next_power_of_2(head_size_v)
    segm_output = torch.empty(
        (
            seq_threshold_3D,
            num_query_heads,
            NUM_PAR_SOFTMAX_SEGMENTS,
            head_size_v_padded,
        ),
        dtype=torch.float32,
    )
    segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, NUM_PAR_SOFTMAX_SEGMENTS),
        dtype=torch.float32,
    )
    segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, NUM_PAR_SOFTMAX_SEGMENTS),
        dtype=torch.float32,
    )
    return segm_output, segm_max, segm_expsum


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328), (5, 18), (129, 463)],  # mixed prefill + decode
        [(1, 523), (1, 37), (1, 2011)],  # decode-only (exercises 3D path)
    ],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_sizes", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 128])
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_diffkv_vs_fa(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_sizes: tuple[int, int],
    sliding_window: int | None,
    soft_cap: float | None,
    dtype: torch.dtype,
    block_size: int,
    seq_threshold_3D: int,
) -> None:
    head_size_qk, head_size_v = head_sizes

    # DiffKV requires FA3 (Hopper) / FA4 (Blackwell) as the reference.
    fa_version = get_flash_attn_version(head_size=head_size_qk, head_size_v=head_size_v)
    if not is_flash_attn_varlen_func_available() or fa_version not in (3, 4):
        pytest.skip(f"FA DiffKV needs FA3/FA4 (got version {fa_version}).")

    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size_qk**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size_qk, dtype=dtype)
    # Packed KV cache: [num_blocks, block_size, num_kv_heads, hqk + hv].
    kv_cache = torch.randn(
        NUM_BLOCKS,
        block_size,
        num_kv_heads,
        head_size_qk + head_size_v,
        dtype=dtype,
    )
    key_cache = kv_cache[..., :head_size_qk]
    value_cache = kv_cache[..., head_size_qk:]

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    # ---- FlashAttention DiffKV (ground truth) ---------------------------
    # Mirror the backend: fix degenerate strides on size-1 dims so FA's
    # TMA path sees ≥16-byte-aligned strides (matters for num_kv_heads==1).
    fa_k = canonicalize_singleton_dim_strides(key_cache)
    fa_v = canonicalize_singleton_dim_strides(value_cache)
    fa_out = torch.empty(sum(query_lens), num_query_heads, head_size_v, dtype=dtype)
    flash_attn_varlen_func(
        q=query,
        k=fa_k,
        v=fa_v,
        out=fa_out,
        cu_seqlens_q=cu_query_lens,
        max_seqlen_q=max_query_len,
        seqused_k=kv_lens_t,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=list(window_size),
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        fa_version=fa_version,
    )

    # ---- Triton DiffKV --------------------------------------------------
    segm_output, segm_max, segm_expsum = _alloc_segm_buffers(
        seq_threshold_3D, num_query_heads, head_size_v
    )
    triton_out = torch.empty(sum(query_lens), num_query_heads, head_size_v, dtype=dtype)
    unified_attention_diffkv(
        q=query,
        k=key_cache,
        v=value_cache,
        out=triton_out,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        max_seqlen_q=max_query_len,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
    )

    (
        torch.testing.assert_close(triton_out, fa_out, atol=2e-2, rtol=2e-2),
        f"triton vs FA max abs diff: {torch.max(torch.abs(triton_out - fa_out))}",
    )
