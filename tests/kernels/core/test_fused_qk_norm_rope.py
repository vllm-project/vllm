# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float16]
IS_NEOX = [True, False]
EPS_VALUES = [1e-5, 1e-6]
SEEDS = [13]
PARTIAL_ROPE = [True, False]
CUDA_DEVICES = ["cuda:0"]


def _apply_qk_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    rope: RotaryEmbedding,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> torch.Tensor:
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
    q_by_head = q_norm.forward_native(q_by_head)
    assert isinstance(q_by_head, torch.Tensor)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
    k_by_head = k_norm.forward_native(k_by_head)
    assert isinstance(k_by_head, torch.Tensor)
    k = k_by_head.view(k.shape)

    q, k = rope.forward_native(positions, q, k)
    return torch.cat([q, k, v], dim=-1)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_norm_rope custom op requires cuda and rocm platform",
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("eps", EPS_VALUES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("rotary_ratio", [1.0, 0.5, 0.25])
@torch.inference_mode()
def test_fused_qk_norm_rope_matches_reference(
    default_vllm_config,
    device: str,
    dtype: torch.dtype,
    is_neox: bool,
    eps: float,
    seed: int,
    rotary_ratio: float,
):
    torch.set_default_device(device)
    set_random_seed(seed)
    num_heads, num_kv_heads, head_dim = 16, 4, 128
    num_tokens = 4

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv_base = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    qkv_fused = qkv_base.clone()
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    q_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    k_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    q_norm.weight.data.normal_(mean=1.0, std=0.1)
    k_norm.weight.data.normal_(mean=1.0, std=0.1)
    q_weight = q_norm.weight.data
    k_weight = k_norm.weight.data
    rotary_dim = int(head_dim * rotary_ratio)
    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=4096,
        base=10000.0,
        is_neox_style=is_neox,
        dtype=dtype,
    ).to(device)

    ref_result = _apply_qk_norm_rope(
        qkv=qkv_base,
        positions=positions,
        q_norm=q_norm,
        k_norm=k_norm,
        rope=rope,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
    )

    opcheck(
        torch.ops._C.fused_qk_norm_rope,
        (
            qkv_fused.clone(),
            num_heads,
            num_kv_heads,
            num_kv_heads,
            head_dim,
            eps,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            is_neox,
            positions.view(-1),
        ),
    )

    torch.ops._C.fused_qk_norm_rope(
        qkv_fused,
        num_heads,
        num_kv_heads,
        num_kv_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        rope.cos_sin_cache,
        is_neox,
        positions.view(-1),
    )

    if dtype == torch.float16:
        ATOL, RTOL = (2e-3, 2e-3)
    else:
        ATOL, RTOL = (1e-2, 1e-2)

    torch.testing.assert_close(
        qkv_fused,
        ref_result,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused gate kernel requires cuda or rocm platform",
)
@pytest.mark.parametrize("num_tokens", [1, 8, 64, 512])
@torch.inference_mode()
def test_fused_qk_norm_rope_gate_matches_eager(
    default_vllm_config,
    num_tokens: int,
):
    """Gated Triton kernel vs. ``Qwen3NextAttention._project_qkv_gate``'s eager
    branch -- the parity guarantee behind enabling the kernel on ROCm."""
    device = "cuda:0"
    dtype = torch.bfloat16
    torch.set_default_device(device)
    set_random_seed(13)

    num_q_heads, num_kv_heads, head_dim = 24, 4, 256
    partial_rotary_factor = 0.25  # rotary_dim = 64
    eps = 1e-6

    q_norm = GemmaRMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    k_norm = GemmaRMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    # GemmaRMSNorm applies x * (1 + weight), so weight is centered on 0.
    q_norm.weight.data.normal_(mean=0.0, std=0.1)
    k_norm.weight.data.normal_(mean=0.0, std=0.1)
    rotary_emb = get_rope(
        head_size=head_dim,
        max_position=8192,
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": partial_rotary_factor,
        },
    ).to(device)

    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q_gate = torch.randn(num_tokens, q_size * 2, dtype=dtype, device=device)
    k_in = torch.randn(num_tokens, kv_size, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    # Fused path (mirrors _project_qkv_gate fused branch).
    q_f, k_f, gate_f = fused_qk_rmsnorm_rope_gate(
        q_gate.clone(),
        k_in.clone(),
        q_norm.weight.float() + 1.0,
        k_norm.weight.float() + 1.0,
        rotary_emb.cos_sin_cache,
        positions,
        q_norm.variance_epsilon,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_emb.rotary_dim,
    )

    # Eager path (mirrors _project_qkv_gate eager branch exactly).
    qg = q_gate.view(num_tokens, num_q_heads, -1)
    q_e, gate_e = torch.chunk(qg, 2, dim=-1)
    q_e = q_e.reshape(num_tokens, -1)
    gate_e = gate_e.reshape(num_tokens, -1)
    q_e = q_norm(q_e.view(-1, num_q_heads, head_dim)).view(-1, q_size)
    k_e = k_norm(k_in.view(-1, num_kv_heads, head_dim)).view(-1, kv_size)
    q_e, k_e = rotary_emb(positions, q_e.clone(), k_e.clone())

    # bf16 rope materializes ~1-2 ULP off the fused kernel's round-trip; the
    # non-rotary head region is bit-identical. 5e-2 covers the rotary ULP.
    torch.testing.assert_close(q_f, q_e, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(k_f, k_e, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(gate_f, gate_e, atol=0, rtol=0)
