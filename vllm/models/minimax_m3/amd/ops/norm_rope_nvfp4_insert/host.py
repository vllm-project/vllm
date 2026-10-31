# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch wrapper of the fused norm / RoPE / NVFP4 or MXFP4 insert."""

import functools
import struct

import torch

from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    run_compiled,
)
from vllm.utils.torch_utils import mxfp4_kv_cache_full_dim, nvfp4_kv_cache_full_dim

from . import kernel as _kernel


@functools.cache
def _get_kernel(nq: int, nkv: int, niq: int, rotary_dim: int, tp: int, mx: bool):
    return _kernel.compile_norm_rope_insert(nq, nkv, niq, rotary_dim, tp, mx)


def _token_pairs_per_wave(n: int) -> int:
    return 1 if n <= 512 else 4


def minimax_m3_norm_rope_nvfp4_insert(
    qkv: torch.Tensor,  # [n, (nq + 2 nkv + niq + 1) * 128] bf16
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim] bf16
    positions: torch.Tensor,  # [n] int64
    slot_mapping: torch.Tensor,  # [n] int64, -1 skips the token
    index_slot_mapping: torch.Tensor,  # [n] int64, -1 skips the token
    kv_cache: torch.Tensor,  # [blocks, 2 nkv, 128, 72 (MXFP4: 68)] uint8
    index_cache: torch.Tensor,  # [blocks, 128, 72 (MXFP4: 68)] uint8
    k_scale: torch.Tensor,  # fp32 global scales, unused for MXFP4
    v_scale: torch.Tensor,
    q_out: torch.Tensor,  # [n, nq * 128] bf16
    index_q_out: torch.Tensor,  # [n, niq * 128] e4m3, row stride free
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    eps: float,
    mxfp4: bool = False,
) -> None:
    """Norm + RoPE of q / k / index-q / index-k and the NVFP4 (``mxfp4``: MXFP4)
    K/V and index-K inserts in one launch; k and index-k are not written back
    into ``qkv``."""
    n = qkv.shape[0]
    if n == 0:
        return
    nq, nkv, niq = num_heads, num_kv_heads, num_index_heads
    slot_w = (mxfp4_kv_cache_full_dim if mxfp4 else nvfp4_kv_cache_full_dim)(_kernel.D)
    rotary_dim = cos_sin_cache.shape[1]
    assert eps >= 2.0**-126  # the kernel's rsqrt skips ocml's denormal scaling
    assert qkv.dtype == torch.bfloat16 and qkv.is_contiguous()
    assert qkv.shape[1] == (nq + 2 * nkv + niq + 1) * _kernel.D
    assert cos_sin_cache.dtype == torch.bfloat16 and cos_sin_cache.is_contiguous()
    for w in (q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight):
        assert w.dtype == torch.bfloat16 and w.is_contiguous()
    assert q_out.dtype == torch.bfloat16 and q_out.is_contiguous()
    assert index_q_out.dtype == torch.float8_e4m3fn and index_q_out.stride(1) == 1
    assert index_q_out.stride(0) % 4 == 0
    assert kv_cache.shape[1:] == (2 * nkv, _kernel.BLK, slot_w)
    assert kv_cache.stride(1) == _kernel.BLK * slot_w and kv_cache.stride(3) == 1
    assert index_cache.shape[1:] == (_kernel.BLK, slot_w) and index_cache.stride(2) == 1
    assert index_cache.stride(1) == slot_w
    run_compiled(
        _get_kernel(nq, nkv, niq, rotary_dim, _token_pairs_per_wave(n), mxfp4),
        qkv.data_ptr(),
        q_out.data_ptr(),
        index_q_out.data_ptr(),
        q_norm_weight.data_ptr(),
        k_norm_weight.data_ptr(),
        index_q_norm_weight.data_ptr(),
        index_k_norm_weight.data_ptr(),
        cos_sin_cache.data_ptr(),
        positions.data_ptr(),
        slot_mapping.data_ptr(),
        index_slot_mapping.data_ptr(),
        kv_cache.data_ptr(),
        index_cache.data_ptr(),
        k_scale.data_ptr(),
        v_scale.data_ptr(),
        n,
        index_q_out.stride(0),
        kv_cache.stride(0),
        index_cache.stride(0),
        cos_sin_cache.shape[0],
        struct.unpack("<i", struct.pack("<f", eps))[0],
        torch.cuda.current_stream(),
    )
