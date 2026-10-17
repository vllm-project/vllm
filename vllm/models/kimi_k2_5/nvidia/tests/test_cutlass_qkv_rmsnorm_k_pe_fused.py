# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import get_rope

if not torch.cuda.is_available():
    pytest.skip("This test only runs on CUDA GPUs.", allow_module_level=True)

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.torch")

from vllm.models.kimi_k2_5.nvidia.ops.qkv_rmsnorm_k_pe_fused import (  # noqa: E402
    qkv_rmsnorm_k_pe_fused,
)

# Default bf16 tolerances, from PyTorch's test_transformers.py
# (https://github.com/pytorch/pytorch/blob/6d96beb/test/test_transformers.py#L67).
_BF16_ATOL = 1e-3
_BF16_RTOL = 1.6e-2

Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
TOTAL_LORA_RANK = Q_LORA_RANK + KV_LORA_RANK
MAX_POSITION = 262144
KERNEL_ATOL = 2e-2
ROPE_PARAMETERS = {
    "rope_type": "deepseek_yarn",
    "rope_theta": 50000.0,
    "factor": 64.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
}


def _pytorch_rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_fp32 = x.float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * inv_rms).to(torch.bfloat16) * weight


def _make_mla_like_qkv_a(num_tokens: int) -> torch.Tensor:
    # The fused QKV-A projection output: [q_lora | kv_lora | qk_rope]. The kernel
    # operates on strided views of this buffer, exactly as the model does.
    return torch.randn(
        num_tokens,
        TOTAL_LORA_RANK + QK_ROPE_HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("num_tokens", [7, 64])
@torch.inference_mode()
def test_kimik25_qkv_rmsnorm_k_pe_fused_matches_reference(
    default_vllm_config,
    num_tokens: int,
) -> None:
    torch.manual_seed(0)

    rope = get_rope(
        head_size=QK_ROPE_HEAD_DIM,
        max_position=MAX_POSITION,
        is_neox_style=False,
        rope_parameters=ROPE_PARAMETERS,
        dtype=torch.bfloat16,
    ).to(device="cuda", dtype=torch.bfloat16)

    positions = torch.randint(
        0, MAX_POSITION, (num_tokens,), device="cuda", dtype=torch.long
    )
    q_weight = torch.randn(Q_LORA_RANK, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(KV_LORA_RANK, device="cuda", dtype=torch.bfloat16)
    eps_q = 1e-5
    eps_kv = 1e-5

    qkv_a = _make_mla_like_qkv_a(num_tokens)
    data, k_pe = qkv_a.split([TOTAL_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1)
    assert not data.is_contiguous()
    assert data.stride() == (TOTAL_LORA_RANK + QK_ROPE_HEAD_DIM, 1)

    # References are computed from copies, since the kernel works in place.
    data_in = data.clone()
    q_x, kv_x = data_in.split([Q_LORA_RANK, KV_LORA_RANK], dim=-1)
    expected_q = _pytorch_rmsnorm_reference(q_x, q_weight, eps_q)
    expected_kv = _pytorch_rmsnorm_reference(kv_x, kv_weight, eps_kv)
    _, expected_k_pe = rope.forward_native(
        positions,
        torch.zeros(
            num_tokens, 1, QK_ROPE_HEAD_DIM, device="cuda", dtype=torch.bfloat16
        ),
        k_pe.clone().reshape(num_tokens, 1, QK_ROPE_HEAD_DIM),
    )
    expected_k_pe = expected_k_pe.reshape(num_tokens, QK_ROPE_HEAD_DIM)

    qkv_rmsnorm_k_pe_fused(
        data=data,
        positions=positions,
        k_pe=k_pe,
        cos_sin_cache=rope.cos_sin_cache,
        weights_q=q_weight,
        weights_kv=kv_weight,
        lora_dim_q=Q_LORA_RANK,
        lora_dim_kv=KV_LORA_RANK,
        pe_dim=QK_ROPE_HEAD_DIM,
        eps_q=eps_q,
        eps_kv=eps_kv,
    )
    torch.cuda.synchronize()

    actual_q, actual_kv = data.split([Q_LORA_RANK, KV_LORA_RANK], dim=-1)
    torch.testing.assert_close(
        actual_q.float(), expected_q.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        actual_kv.float(), expected_kv.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        k_pe,
        expected_k_pe,
        atol=max(_BF16_ATOL, KERNEL_ATOL),
        rtol=_BF16_RTOL,
    )
