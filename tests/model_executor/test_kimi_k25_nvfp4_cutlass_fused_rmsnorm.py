# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if not torch.cuda.is_available() or not current_platform.is_device_capability_family(
    100
):
    pytest.skip(
        "This test only runs on Blackwell GPUs (SM10x).", allow_module_level=True
    )

cutlass_torch = pytest.importorskip("cutlass.torch")
from_dlpack = pytest.importorskip("cutlass.cute.runtime").from_dlpack

from vllm.model_executor.specialized_models.kimi_k2_5_nvfp4.model import (  # noqa: E402
    kimik25_rmsnorm_special_qkv_split,
)

Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
TOTAL_LORA_RANK = Q_LORA_RANK + KV_LORA_RANK


def _pytorch_rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_fp32 = x.float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * inv_rms).to(torch.bfloat16) * weight


def _pytorch_split_rmsnorm_reference(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps_q: float,
    eps_kv: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_x, kv_x = x.split([Q_LORA_RANK, KV_LORA_RANK], dim=-1)
    q_expected = _pytorch_rmsnorm_reference(q_x, q_weight, eps_q)
    kv_expected = _pytorch_rmsnorm_reference(kv_x, kv_weight, eps_kv)
    return q_expected, kv_expected


def _make_mla_like_qkv_c_view(num_tokens: int) -> torch.Tensor:
    qkv_a = torch.randn(
        num_tokens,
        TOTAL_LORA_RANK + QK_ROPE_HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    qkv_c, _ = qkv_a.split([TOTAL_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1)
    return qkv_c


@pytest.mark.parametrize("num_tokens", [7, 64])
def test_kimik25_cutlass_fused_rmsnorm_matches_two_pytorch_rmsnorms(
    num_tokens: int,
) -> None:
    torch.manual_seed(0)

    q_weight = torch.randn(Q_LORA_RANK, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(KV_LORA_RANK, device="cuda", dtype=torch.bfloat16)
    eps_q = 1e-5
    eps_kv = 1e-5

    reference_input = _make_mla_like_qkv_c_view(num_tokens)
    expected_q, expected_kv = _pytorch_split_rmsnorm_reference(
        reference_input,
        q_weight,
        kv_weight,
        eps_q,
        eps_kv,
    )

    actual_input = _make_mla_like_qkv_c_view(num_tokens)
    actual_input.copy_(reference_input)

    assert actual_input.shape == (num_tokens, TOTAL_LORA_RANK)
    assert actual_input.stride() == (TOTAL_LORA_RANK + QK_ROPE_HEAD_DIM, 1)
    assert not actual_input.is_contiguous()

    kimik25_rmsnorm_special_qkv_split(
        data=from_dlpack(actual_input).mark_layout_dynamic(),
        weights_q=from_dlpack(q_weight),
        weights_kv=from_dlpack(kv_weight),
        lora_dim_q=Q_LORA_RANK,
        lora_dim_kv=KV_LORA_RANK,
        eps_q=eps_q,
        eps_kv=eps_kv,
        stream=cutlass_torch.current_stream(),
    )
    torch.cuda.synchronize()

    actual_q, actual_kv = actual_input.split([Q_LORA_RANK, KV_LORA_RANK], dim=-1)
    torch.testing.assert_close(
        actual_q.float(),
        expected_q.float(),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        actual_kv.float(),
        expected_kv.float(),
        atol=2e-2,
        rtol=2e-2,
    )
