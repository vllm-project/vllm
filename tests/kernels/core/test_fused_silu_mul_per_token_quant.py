# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()

DTYPES = [torch.float16, torch.bfloat16]
NUM_TOKENS_HIDDEN_SIZES = [
    *[(1, h) for h in [64, 1024, 2048, 4096, 5120, 7168]],
    *[(16, h) for h in [64, 1024, 4096, 5120]],
    *[(128, h) for h in [64, 1024, 4096, 7168]],
    *[(512, h) for h in [64, 1024, 5120]],
    *[(2048, h) for h in [1024, 4096, 5120]],
]
SEEDS = [0]
CUDA_DEVICES = [
    i for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


def ref_silu_and_mul_per_token_quant(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: unfused SiLU+Mul then per-token dynamic FP8 quant."""
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)
    silu_out = F.silu(gate.float()) * up.float()

    # Per-token dynamic quant (matches C++ dynamic_per_token_scaled_fp8_quant
    # convention: scale = amax(row), output = clamp(input / scale * FP8_MAX))
    fp8_max = torch.finfo(quant_dtype).max
    scales = silu_out.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    quantized = (silu_out / scales * fp8_max).clamp(-fp8_max, fp8_max).to(
        quant_dtype
    )
    return quantized, scales


@pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device_idx", CUDA_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_per_token_quant(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    device_idx: int,
) -> None:
    """Test fused SiLU+Mul + per-token FP8 quant kernel correctness."""
    torch.random.manual_seed(seed)
    device = f"cuda:{device_idx}"
    torch.set_default_device(device)

    scale = 1 / hidden_size
    x = torch.randn(
        num_tokens, hidden_size * 2, dtype=dtype, device=device
    ) * scale

    # Reference
    ref_out, ref_scales = ref_silu_and_mul_per_token_quant(x, FP8_DTYPE)

    # Fused kernel
    fused_out, fused_scales = ops.fused_silu_mul_per_token_quant(
        x, FP8_DTYPE
    )

    # Check for NaN/Inf
    assert not torch.isnan(fused_out.float()).any(), "Output contains NaN"
    assert not torch.isinf(fused_out.float()).any(), "Output contains Inf"
    assert not torch.isnan(fused_scales).any(), "Scales contain NaN"
    assert not torch.isinf(fused_scales).any(), "Scales contain Inf"

    # Dtype checks
    assert fused_out.dtype == FP8_DTYPE
    assert fused_scales.dtype == torch.float32

    # Shape checks
    assert fused_out.shape == (num_tokens, hidden_size)
    assert fused_scales.shape == (num_tokens, 1)

    # Scales match
    torch.testing.assert_close(fused_scales, ref_scales, rtol=1e-3, atol=1e-6)

    # Dequantized output match
    ref_deq = ref_out.to(torch.float32) * ref_scales
    fused_deq = fused_out.to(torch.float32) * fused_scales
    torch.testing.assert_close(ref_deq, fused_deq, atol=5e-2, rtol=5e-2)

    # opcheck
    opcheck(
        torch.ops.vllm.fused_silu_mul_per_token_quant,
        (x, FP8_DTYPE, None),
    )


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("num_tokens", [1, 128])
def test_silu_per_token_quant_shapes(
    default_vllm_config,
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
):
    """Test output shapes are correct."""
    torch.set_default_device("cuda")
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device="cuda")

    out, scales = ops.fused_silu_mul_per_token_quant(x, FP8_DTYPE)

    assert out.shape == (num_tokens, hidden_size)
    assert out.dtype == FP8_DTYPE
    assert scales.shape == (num_tokens, 1)
    assert scales.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 16, 256])
@pytest.mark.parametrize("hidden_size", [1024, 5120, 7168])
def test_silu_per_token_quant_edge_cases(
    default_vllm_config,
    dtype: torch.dtype,
    batch_size: int,
    hidden_size: int,
):
    """Test edge cases: single token, large batch, large hidden."""
    torch.set_default_device("cuda")
    x = torch.randn(batch_size, hidden_size * 2, dtype=dtype, device="cuda")

    out, scales = ops.fused_silu_mul_per_token_quant(x, FP8_DTYPE)

    assert out.shape == (batch_size, hidden_size)
    assert out.dtype == FP8_DTYPE
    assert scales.dtype == torch.float32
    assert not torch.isnan(out.float()).any()
    assert not torch.isnan(scales).any()
    assert (scales > 0).all()
