# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test:

* Tests for MMEncoderAttention layer
"""

import itertools
import sys
import types
from typing import Any
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
from vllm.platforms.interface import DeviceCapability
from vllm.platforms.rocm import RocmPlatform, on_mi3xx
from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import _cached_get_attn_backend


@pytest.fixture(autouse=True)
def reset_test_state():
    """Clear cached selectors and restore process-wide torch defaults."""
    default_device = torch.get_default_device()
    default_dtype = torch.get_default_dtype()
    _cached_get_attn_backend.cache_clear()
    try:
        yield
    finally:
        torch.set_default_device(default_device)
        torch.set_default_dtype(default_dtype)
        _cached_get_attn_backend.cache_clear()


devices = ["cpu"]
if current_platform.is_cuda():
    devices.append("cuda")
if current_platform.is_rocm():
    devices.append("hip")


@pytest.mark.parametrize("device", devices)
def test_mha_attn_platform(default_vllm_config, device: str):
    """Test the attention selector between different platform and device."""
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

        # Test Turing (pre-Ampere, sm_75): FlashAttention requires sm>=80,
        # and Triton no longer supports MMA on Turing, so we expect that
        # TORCH_SDPA is used for MMEncoderAttention.
        with (
            patch("vllm.model_executor.models.vision.current_platform", CudaPlatform()),
            patch.object(
                CudaPlatform,
                "get_device_capability",
                return_value=DeviceCapability(major=7, minor=5),
            ),
        ):
            attn = MMEncoderAttention(16, 64, scale=1)
            assert attn.attn_backend == AttentionBackendEnum.TORCH_SDPA


def ref_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Native implementation of scaled dot product attention without mask:
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
    old_model_config = vllm_config.model_config
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

        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
            AttentionBackendEnum.FLASHINFER,
            cu_seqlens_np,
            device,
        )

        max_seqlen_val = MMEncoderAttention.compute_max_seqlen(
            AttentionBackendEnum.FLASHINFER, cu_seqlens_np
        )
        max_seqlen = torch.tensor(max_seqlen_val, device=device, dtype=torch.int32)

        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            AttentionBackendEnum.FLASHINFER,
            cu_seqlens_np,
            hidden_size,
            tp_size,
            device,
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


@pytest.mark.skipif(
    not current_platform.is_rocm() or not on_mi3xx(),
    reason="AITER FP8 attention requires MI300/MI350",
)
@pytest.mark.parametrize("var_seq_len", [[128, 193], [256, 384, 511]])
@pytest.mark.parametrize("head_size", [64, 72, 80])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
def test_mha_attn_varlen_forward_aiter_fp8(
    default_vllm_config,
    var_seq_len: list[int],
    head_size: int,
    dtype: torch.dtype,
):
    """Compare packed AITER FP8 ViT attention with a BF16/FP16 reference."""
    aiter = pytest.importorskip("aiter")
    if not hasattr(aiter, "flash_attn_varlen_fp8_pertensor_func"):
        pytest.skip("Installed AITER does not provide varlen FP8 attention")

    num_heads = 4
    set_random_seed(0)
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)

    vllm_config = get_current_vllm_config()
    old_model_config = vllm_config.model_config
    minimal_model_config = type(
        "MinimalModelConfig",
        (),
        {
            "multimodal_config": MultiModalConfig(
                mm_encoder_attn_backend=AttentionBackendEnum.ROCM_AITER_FA,
                mm_encoder_attn_dtype="fp8",
            ),
        },
    )()
    vllm_config.model_config = minimal_model_config
    try:
        total_len = sum(var_seq_len)
        # Keep Q/K/V as non-contiguous views, matching interleaved ViT QKV.
        qkv = torch.randn(1, total_len, 3, num_heads, head_size)
        q, k, v = qkv.unbind(dim=2)
        cu_seqlens = torch.tensor(
            [0] + list(itertools.accumulate(var_seq_len)), dtype=torch.int32
        )
        max_seqlen = torch.tensor(max(var_seq_len), dtype=torch.int32)
        scale = 1.0 / head_size**0.5

        attn = MMEncoderAttention(num_heads, head_size, scale=scale)
        assert attn.attn_backend == AttentionBackendEnum.ROCM_AITER_FA
        assert attn.fp8_enabled
        output = attn(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        ref_output = torch.cat(
            [
                ref_attention(q_i, k_i, v_i, scale=scale)
                for q_i, k_i, v_i in zip(
                    torch.split(q, var_seq_len, dim=1),
                    torch.split(k, var_seq_len, dim=1),
                    torch.split(v, var_seq_len, dim=1),
                )
            ],
            dim=1,
        )
        diff = (output.float() - ref_output.float()).abs()
        cosine = torch.nn.functional.cosine_similarity(
            output.float().flatten(),
            ref_output.float().flatten(),
            dim=0,
        )

        assert output.shape == ref_output.shape
        assert output.dtype == dtype
        assert cosine.item() >= 0.99
        assert diff.mean().item() <= 0.03
        assert diff.max().item() <= 0.30
    finally:
        vllm_config.model_config = old_model_config


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="AITER FP8 attention requires ROCm"
)
def test_mha_attn_aiter_fp8_rejects_unsupported_arch(default_vllm_config):
    vllm_config = get_current_vllm_config()
    old_model_config = vllm_config.model_config
    vllm_config.model_config = type(
        "MinimalModelConfig",
        (),
        {
            "multimodal_config": MultiModalConfig(
                mm_encoder_attn_backend=AttentionBackendEnum.ROCM_AITER_FA,
                mm_encoder_attn_dtype="fp8",
            ),
        },
    )()
    try:
        with (
            patch("vllm.platforms.rocm.on_mi3xx", return_value=False),
            pytest.raises(ValueError, match="gfx942 or gfx950"),
        ):
            MMEncoderAttention(4, 64)
    finally:
        vllm_config.model_config = old_model_config


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="AITER FP8 attention requires ROCm"
)
def test_mha_attn_fp8_rejects_wrong_backend(default_vllm_config):
    vllm_config = get_current_vllm_config()
    old_model_config = vllm_config.model_config
    vllm_config.model_config = type(
        "MinimalModelConfig",
        (),
        {
            "multimodal_config": MultiModalConfig(
                mm_encoder_attn_backend=AttentionBackendEnum.FLASH_ATTN,
                mm_encoder_attn_dtype="fp8",
            ),
        },
    )()
    try:
        with pytest.raises(ValueError, match="requires either"):
            MMEncoderAttention(4, 64)
    finally:
        vllm_config.model_config = old_model_config


@pytest.mark.parametrize(
    "fa_version,head_size,is_sm90,fa4_supported,pinned,expected",
    [
        (3, 80, True, True, None, 4),
        (3, 64, True, True, None, None),
        (3, 72, True, True, None, None),
        (3, 128, True, True, None, None),
        (2, 80, True, True, None, None),
        (3, 80, False, True, None, None),
        (3, 80, True, False, None, None),
        (3, 80, True, True, 3, None),
    ],
)
def test_vit_long_seq_fa_version(
    fa_version, head_size, is_sm90, fa4_supported, pinned, expected
):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.attention import AttentionConfig
    from vllm.v1.attention.backends import fa_utils

    # `vllm.vllm_flash_attn` is CUDA-only, so patching through it by name would
    # raise AttributeError on ROCm before the test body runs. Stub the module in
    # sys.modules instead, which the lazy `from ... import` inside
    # get_vit_long_seq_fa_version picks up on every platform.
    fa_pkg: Any = types.ModuleType("vllm.vllm_flash_attn")
    fa_mod: Any = types.ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
    fa_mod.is_fa_version_supported = lambda *a, **kw: fa4_supported
    fa_pkg.flash_attn_interface = fa_mod

    config = VllmConfig(attention_config=AttentionConfig(flash_attn_version=pinned))
    with (
        set_current_vllm_config(config),
        patch.object(
            fa_utils.current_platform, "is_device_capability", return_value=is_sm90
        ),
        patch.dict(
            sys.modules,
            {
                "vllm.vllm_flash_attn": fa_pkg,
                "vllm.vllm_flash_attn.flash_attn_interface": fa_mod,
            },
        ),
    ):
        assert fa_utils.get_vit_long_seq_fa_version(fa_version, head_size) == expected


@pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="FA3/FA4 on SM90 only"
)
@pytest.mark.parametrize("seqlen", [4096, 8192])
def test_vit_long_seq_fa4_dispatch(seqlen: int):
    from vllm.v1.attention.ops.vit_attn_wrappers import (
        VIT_LONG_SEQ_MIN_SEQLEN,
        flash_attn_maxseqlen_wrapper,
    )
    from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported

    if not is_fa_version_supported(4):
        pytest.skip("FA4 is not available in this build")

    set_random_seed(0)
    num_heads, head_size = 16, 80
    q, k, v = (
        torch.randn(
            1, seqlen, num_heads, head_size, device="cuda", dtype=torch.bfloat16
        )
        for _ in range(3)
    )
    max_seqlen = torch.tensor(seqlen, dtype=torch.int32)

    def attn(long_seq_fa_version):
        return flash_attn_maxseqlen_wrapper(
            q,
            k,
            v,
            1,
            False,
            3,
            head_size**-0.5,
            max_seqlen=max_seqlen,
            long_seq_fa_version=long_seq_fa_version,
        )

    out_fa3, out_gated = attn(None), attn(4)
    if seqlen < VIT_LONG_SEQ_MIN_SEQLEN:
        # Below the threshold the gated call must still run FA3.
        torch.testing.assert_close(out_gated, out_fa3, rtol=0, atol=0)
        return

    ref = torch.nn.functional.scaled_dot_product_attention(
        *(x.float().transpose(1, 2) for x in (q, k, v)), scale=head_size**-0.5
    ).transpose(1, 2)
    err = lambda o: ((o.float() - ref).norm() / ref.norm()).item()  # noqa: E731
    assert err(out_gated) <= 1.05 * err(out_fa3)
