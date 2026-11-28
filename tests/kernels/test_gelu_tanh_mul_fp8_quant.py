# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.model_executor.layers.fusion import gelu_tanh_mul_fp8_quant  # noqa: F401


def _reference(x: torch.Tensor):
    d = x.shape[-1] // 2
    act = torch.nn.functional.gelu(x[:, :d].float(), approximate="tanh")
    v = (act.to(torch.bfloat16).float() * x[:, d:].float()).to(torch.bfloat16).float()
    scale = v.abs().amax(dim=-1, keepdim=True) / 448.0
    q = (v / scale).clamp(-448.0, 448.0).to(current_platform.fp8_dtype())
    return q, scale


@pytest.mark.parametrize("num_tokens", [1, 7, 256, 1000])
@pytest.mark.parametrize("d", [1024, 10240])
def test_gelu_tanh_mul_fp8_per_token_quant(num_tokens: int, d: int):
    torch.manual_seed(0)
    x = (torch.randn(num_tokens, 2 * d, device="cuda") * 3).to(torch.bfloat16)
    q, scale = torch.ops.vllm.gelu_tanh_mul_fp8_per_token_quant(x)
    q_ref, scale_ref = _reference(x)
    assert q.shape == (num_tokens, d) and scale.shape == (num_tokens, 1)
    torch.testing.assert_close(scale, scale_ref, rtol=1e-3, atol=0)
    # Dequantized values agree to one fp8 step (3 mantissa bits).
    torch.testing.assert_close(
        q.float() * scale, q_ref.float() * scale_ref, rtol=0.14, atol=1e-3
    )
