# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test:

* Tests for MMEncoderAttention layer
"""

import itertools
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.config import get_current_vllm_config
from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.platforms import current_platform
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.rocm import RocmPlatform
from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import _cached_get_attn_backend


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching."""
    _cached_get_attn_backend.cache_clear()


devices = ["cpu"]
if current_platform.is_cuda():
    devices.append("cuda")
if current_platform.is_rocm():
    devices.append("hip")


@pytest.mark.parametrize("device", devices)
def test_mha_attn_platform(default_vllm_config, device: str):
    """
    Test the attention selector between different platform and device.
    """
    torch.set_default_dtype(torch.float16)

    if device == "cpu":
        with (
            patch("vllm.model_executor.models.vision.current_platform", CpuPlatform()),
        ):
            attn = MMEncoderAttention(16, 64, scale=1)
            assert attn.attn_backend == AttentionBackendEnum.TORCH_SDPA
    elif device == "hip":
        with (
            patch("vllm.model_executor.models.vision.current_platform", RocmPlatform()),
        ):
            attn = MMEncoderAttention(16, 64, scale=1)
            assert attn.attn_backend == AttentionBackendEnum.FLASH_ATTN
    else:
        # Test CUDA with head_size=64 (divisible by 32)
        # - should use vLLM's FlashAttention
        with (
            patch("vllm.model_executor.models.vision.current_platform", CudaPlatform()),
        ):
            attn = MMEncoderAttention(16, 64, scale=1)
            assert attn.attn_backend == AttentionBackendEnum.FLASH_ATTN

        # Test CUDA with head_size=72 (not divisible by 32)
        # - should use vLLM's FlashAttention
        with (
            patch("vllm.model_executor.models.vision.current_platform", CudaPlatform()),
        ):
            attn = MMEncoderAttention(16, 72, scale=1)
            assert attn.attn_backend == AttentionBackendEnum.FLASH_ATTN

        # Test CUDA with head_size=72 (not divisible by 32)
        # - should use vLLM's FlashAttention
        with (
            patch("vllm.model_executor.models.vision.current_platform", CudaPlatform()),
            set_default_torch_dtype(torch.float32),
        ):
            attn = MMEncoderAttention(16, 72, scale=1)
            assert attn.attn_backend == AttentionBackendEnum.TRITON_ATTN


def ref_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Native implementation of scaled dot product attention without mask:
    - query, key, value: [batch_size, seq_len, num_heads, head_size]
    - attn_mask: [batch_size, seq_len, seq_len]
    """
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    attn_weights = scale * torch.matmul(query, key.transpose(2, 3))
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.matmul(attn_weights, value).transpose(1, 2)
    return out


BATCH_SIZES = [1, 16]
SEQ_LENS = [1]
VAR_SEQ_LENS = [
    [2, 2],
    [2, 3, 4],
]
NUM_HEADS = [1, 16]
NUM_KV_HEADS = [1]
HEAD_SIZES = [64, 80]
# flshattF and tritonflashattF supported: {torch.float16, torch.bfloat16}
DTYPES = (
    [torch.half, torch.bfloat16, torch.float]
    if not current_platform.is_rocm()
    else [torch.half, torch.bfloat16]
)
CUDA_DEVICES = ["cuda"]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_mha_attn_forward(
    default_vllm_config,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: str,
):
    set_random_seed(0)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    q = torch.randn(batch_size, seq_len, num_heads * head_size)
    k = torch.randn(batch_size, seq_len, num_kv_heads * head_size)
    v = torch.randn(batch_size, seq_len, num_kv_heads * head_size)
    scale = 1.0 / head_size**0.5
    attn = MMEncoderAttention(
        num_heads, head_size, scale=scale, num_kv_heads=num_kv_heads
    )
    output = attn(q, k, v)

    assert num_heads % num_kv_heads == 0
    num_queries_per_kv = num_heads // num_kv_heads
    q = q.reshape(batch_size, seq_len, num_heads, head_size)
    k = k.reshape(batch_size, seq_len, num_kv_heads, head_size)
    v = v.reshape(batch_size, seq_len, num_kv_heads, head_size)
    if num_queries_per_kv > 1:
        k = torch.repeat_interleave(k, num_queries_per_kv, dim=2)
        v = torch.repeat_interleave(v, num_queries_per_kv, dim=2)

    ref_output = ref_attention(
        q,
        k,
        v,
        scale=scale,
    ).reshape(batch_size, seq_len, num_heads * head_size)
    tol_kwargs = (
        dict(rtol=1e-3, atol=1e-3)
        if attn.attn_backend == AttentionBackendEnum.TRITON_ATTN
        else {}
    )
    torch.testing.assert_close(output, ref_output, **tol_kwargs)


@pytest.mark.parametrize("var_seq_len", VAR_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_mha_attn_varlen_forward(
    default_vllm_config,
    var_seq_len: list[int],
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: str,
):
    set_random_seed(0)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    q = torch.randn(1, sum(var_seq_len), num_heads, head_size)
    k = torch.randn(1, sum(var_seq_len), num_kv_heads, head_size)
    v = torch.randn(1, sum(var_seq_len), num_kv_heads, head_size)
    cu_seqlens = torch.tensor(
        [0] + list(itertools.accumulate(var_seq_len)), dtype=torch.int32
    )
    scale = 1.0 / head_size**0.5
    attn = MMEncoderAttention(
        num_heads, head_size, scale=scale, num_kv_heads=num_kv_heads
    )
    output = attn(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=torch.tensor(max(var_seq_len))
    )

    assert num_heads % num_kv_heads == 0
    num_queries_per_kv = num_heads // num_kv_heads
    if num_queries_per_kv > 1:
        k = torch.repeat_interleave(k, num_queries_per_kv, dim=2)
        v = torch.repeat_interleave(v, num_queries_per_kv, dim=2)

    ref_output = []
    for q_i, k_i, v_i in zip(
        torch.split(q, var_seq_len, dim=1),
        torch.split(k, var_seq_len, dim=1),
        torch.split(v, var_seq_len, dim=1),
    ):
        output_i = ref_attention(
            q_i,
            k_i,
            v_i,
            scale=scale,
        )
        ref_output.append(output_i)
    ref_output = torch.cat(ref_output, dim=1)
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("var_seq_len", VAR_SEQ_LENS)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.half],
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_mha_attn_varlen_forward_flashinfer(
    default_vllm_config,
    var_seq_len: list[int],
    dtype: torch.dtype,
    device: str,
):
    """Test MMEncoderAttention varlen forward with FLASHINFER backend (head_size=72).

    Exercises the path that uses --mm-encoder-attn-backend=FLASHINFER with
    recomputed cu_seqlens, max_seqlen, and sequence_lengths as in qwen3_vl
    vision encoder.
    """
    pytest.importorskip("flashinfer")

    num_heads = 16
    head_size = 72
    set_random_seed(0)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    # Override vllm config so get_vit_attn_backend returns FLASHINFER (simulates
    # --mm-encoder-attn-backend=FLASHINFER).
    vllm_config = get_current_vllm_config()
    old_model_config = getattr(vllm_config, "model_config", None)
    minimal_model_config = type(
        "MinimalModelConfig",
        (),
        {
            "multimodal_config": MultiModalConfig(
                mm_encoder_attn_backend=AttentionBackendEnum.FLASHINFER
            ),
        },
    )()
    vllm_config.model_config = minimal_model_config
    try:
        total_len = sum(var_seq_len)
        # Stride of second dim = 3 * num_heads * head_size (same as qwen2_5_vl
        # after qkv rearrange and unbind: qkv shape (b, s, 3, head, head_dim)).
        qkv = torch.randn(1, total_len, 3, num_heads, head_size)
        q, k, v = qkv.unbind(dim=2)

        cu_seqlens_np = np.array(
            [0] + list(itertools.accumulate(var_seq_len)), dtype=np.int32
        )
        hidden_size = num_heads * head_size
        tp_size = 1

        sequence_lengths_np = MMEncoderAttention.maybe_compute_sequence_lengths(
            AttentionBackendEnum.FLASHINFER, cu_seqlens_np
        )
        sequence_lengths = torch.from_numpy(sequence_lengths_np).to(
            device, dtype=torch.int32, non_blocking=True
        )

        max_seqlen_val = MMEncoderAttention.compute_max_seqlen(
            AttentionBackendEnum.FLASHINFER, cu_seqlens_np
        )
        max_seqlen = torch.tensor(max_seqlen_val, device=device, dtype=torch.int32)

        cu_seqlens_np = MMEncoderAttention.maybe_recompute_cu_seqlens(
            AttentionBackendEnum.FLASHINFER,
            cu_seqlens_np,
            hidden_size,
            tp_size,
            rotary_pos_emb_cos=None,
            rotary_pos_emb_sin=None,
        )
        cu_seqlens = torch.from_numpy(cu_seqlens_np).to(
            device, dtype=torch.int32, non_blocking=True
        )

        scale = 1.0 / head_size**0.5
        attn = MMEncoderAttention(
            num_heads,
            head_size,
            scale=scale,
            num_kv_heads=num_heads,
        )
        assert attn.attn_backend == AttentionBackendEnum.FLASHINFER

        output = attn(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

        ref_output = []
        for q_i, k_i, v_i in zip(
            torch.split(q, var_seq_len, dim=1),
            torch.split(k, var_seq_len, dim=1),
            torch.split(v, var_seq_len, dim=1),
        ):
            output_i = ref_attention(q_i, k_i, v_i, scale=scale)
            ref_output.append(output_i)
        ref_output = torch.cat(ref_output, dim=1)
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
    finally:
        vllm_config.model_config = old_model_config
