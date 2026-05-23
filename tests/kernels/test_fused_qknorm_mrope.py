# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fused QK-RMSNorm + mRoPE kernel (Qwen3-VL style)."""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.mrope import (
    MRotaryEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float16]
IS_NEOX = [True, False]
EPS_VALUES = [1e-5, 1e-6]
SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]

# Qwen3-VL typical config: head_dim=128, mrope_section=[16,24,24]
HEAD_CONFIGS = [
    (16, 4, 128, [16, 24, 24]),  # Qwen3-VL-like
    (8, 2, 64, [8, 12, 12]),  # smaller head_dim=64
]


def _reference_qk_norm_mrope(
    qkv: torch.Tensor,
    mrope: MRotaryEmbedding,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    positions: torch.Tensor,  # [3, num_tokens]
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> torch.Tensor:
    """Unfused reference: separate RMSNorm + Triton mRoPE."""
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    # QK RMSNorm (float32 reference for numerical stability)
    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
    q_by_head = q_norm.forward_native(q_by_head)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
    k_by_head = k_norm.forward_native(k_by_head)
    k = k_by_head.view(k.shape)

    # mRoPE via Triton kernel
    q, k = mrope.forward_cuda(positions, q, k)

    return torch.cat([q, k, v], dim=-1)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_norm_mrope requires CUDA or ROCm",
)
@pytest.mark.skipif(
    not hasattr(torch.ops._C, "fused_qk_norm_mrope"),
    reason="fused_qk_norm_mrope custom op not compiled",
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("eps", EPS_VALUES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "num_heads_q,num_heads_kv,head_dim,mrope_section", HEAD_CONFIGS
)
@torch.inference_mode()
def test_fused_qk_norm_mrope_matches_reference(
    default_vllm_config,
    device: str,
    dtype: torch.dtype,
    is_neox: bool,
    eps: float,
    seed: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    mrope_section: list[int],
):
    torch.set_default_device(device)
    set_random_seed(seed)
    num_tokens = 8

    total_dim = (num_heads_q + 2 * num_heads_kv) * head_dim
    qkv_base = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    qkv_fused = qkv_base.clone()

    # 2-D mRoPE positions: [3, num_tokens] — random small values
    positions = torch.stack(
        [
            torch.arange(num_tokens, device=device, dtype=torch.long),
            torch.arange(num_tokens, device=device, dtype=torch.long),
            torch.arange(num_tokens, device=device, dtype=torch.long),
        ]
    )  # [3, num_tokens]

    q_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    k_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    q_norm.weight.data.normal_(mean=1.0, std=0.1)
    k_norm.weight.data.normal_(mean=1.0, std=0.1)

    rotary_dim = head_dim  # full rotary for simplicity
    mrope = MRotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=4096,
        base=10000.0,
        is_neox_style=is_neox,
        dtype=dtype,
        mrope_section=mrope_section,
    ).to(device)

    # Reference: unfused path
    ref = _reference_qk_norm_mrope(
        qkv=qkv_base,
        mrope=mrope,
        q_norm=q_norm,
        k_norm=k_norm,
        positions=positions,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
    )

    # Prepare cos/sin for the fused kernel
    cos_sin_cache = mrope._match_cos_sin_cache_dtype(qkv_fused)
    cos_sin = cos_sin_cache[positions]  # [3, num_tokens, rotary_dim]
    cos, sin = cos_sin.chunk(2, dim=-1)  # each [3, num_tokens, half_rd]
    cos = cos.contiguous()
    sin = sin.contiguous()

    # opcheck verifies aliasing / schema correctness
    opcheck(
        torch.ops._C.fused_qk_norm_mrope,
        (
            qkv_fused.clone(),
            num_heads_q,
            num_heads_kv,
            num_heads_kv,
            head_dim,
            eps,
            q_norm.weight,
            k_norm.weight,
            cos,
            sin,
            is_neox,
            mrope_section[0],
            mrope_section[1],
        ),
    )

    torch.ops._C.fused_qk_norm_mrope(
        qkv_fused,
        num_heads_q,
        num_heads_kv,
        num_heads_kv,
        head_dim,
        eps,
        q_norm.weight,
        k_norm.weight,
        cos,
        sin,
        is_neox,
        mrope_section[0],
        mrope_section[1],
    )

    atol, rtol = (2e-3, 2e-3) if dtype == torch.float16 else (1e-2, 1e-2)
    torch.testing.assert_close(qkv_fused, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_norm_mrope requires CUDA or ROCm",
)
@pytest.mark.skipif(
    not hasattr(torch.ops._C, "fused_qk_norm_mrope"),
    reason="fused_qk_norm_mrope custom op not compiled",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 32, 256])
@torch.inference_mode()
def test_fused_qk_norm_mrope_varying_lengths(
    default_vllm_config,
    num_tokens: int,
):
    """Verify the kernel handles varying sequence lengths correctly."""
    device = "cuda:0"
    dtype = torch.bfloat16
    num_heads_q, num_heads_kv, head_dim = 16, 4, 128
    mrope_section = [16, 24, 24]
    eps = 1e-6
    rotary_dim = head_dim

    torch.set_default_device(device)
    set_random_seed(0)

    total_dim = (num_heads_q + 2 * num_heads_kv) * head_dim
    qkv_base = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    qkv_fused = qkv_base.clone()

    positions = torch.stack(
        [
            torch.arange(num_tokens, device=device, dtype=torch.long),
            torch.arange(num_tokens, device=device, dtype=torch.long),
            torch.arange(num_tokens, device=device, dtype=torch.long),
        ]
    )

    q_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    k_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)

    mrope = MRotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=4096,
        base=10000.0,
        is_neox_style=True,
        dtype=dtype,
        mrope_section=mrope_section,
    ).to(device)

    ref = _reference_qk_norm_mrope(
        qkv=qkv_base,
        mrope=mrope,
        q_norm=q_norm,
        k_norm=k_norm,
        positions=positions,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
    )

    cos_sin_cache = mrope._match_cos_sin_cache_dtype(qkv_fused)
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.contiguous()
    sin = sin.contiguous()

    torch.ops._C.fused_qk_norm_mrope(
        qkv_fused,
        num_heads_q,
        num_heads_kv,
        num_heads_kv,
        head_dim,
        eps,
        q_norm.weight,
        k_norm.weight,
        cos,
        sin,
        True,
        mrope_section[0],
        mrope_section[1],
    )

    torch.testing.assert_close(qkv_fused, ref, atol=1e-2, rtol=1e-2)
