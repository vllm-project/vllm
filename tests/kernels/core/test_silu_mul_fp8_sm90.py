# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(90),
    reason="FlashInfer's block-scaled FP8 oracle requires SM90",
)
@pytest.mark.parametrize("num_tokens", [1, 3, 6, 31])
@pytest.mark.parametrize("amplitude", [0.0, 1e-5, 1.0, 50.0])
@pytest.mark.parametrize("round_silu", [False, True])
@torch.inference_mode()
def test_silu_mul_fp8_sm90_matches_flashinfer(
    num_tokens: int, amplitude: float, round_silu: bool
) -> None:
    from flashinfer.gemm.gemm_base import get_fp8_blockscale_gemm_runner_sm90

    torch.manual_seed(42)
    k = 17408
    x = torch.randn(num_tokens, 2 * k, device="cuda", dtype=torch.bfloat16)
    x *= amplitude
    if round_silu:
        activation = torch.empty(num_tokens, k, device=x.device, dtype=x.dtype)
        torch.ops._C.silu_and_mul(activation, x)
    else:
        activation = _compiled_sm90_activation(x)
    q = torch.empty_like(activation, dtype=torch.float8_e4m3fn)
    scales = torch.empty_strided(
        (num_tokens, k // 128),
        (1, ((num_tokens + 3) // 4) * 4),
        device=x.device,
        dtype=torch.float32,
    )
    ref_q, ref_scales = (
        torch.empty_like(q),
        torch.empty_strided(
            scales.shape, scales.stride(), device=x.device, dtype=torch.float32
        ),
    )
    get_fp8_blockscale_gemm_runner_sm90().fp8_quantize_1x128(
        activation, ref_q, ref_scales, False
    )
    torch.ops._C.silu_mul_fp8_sm90(q, x, scales, round_silu)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(scales, ref_scales)


@torch.compile(fullgraph=True)
def _compiled_sm90_activation(x: torch.Tensor) -> torch.Tensor:
    k = x.shape[-1] // 2
    return F.silu(x[..., :k]) * x[..., k:]
