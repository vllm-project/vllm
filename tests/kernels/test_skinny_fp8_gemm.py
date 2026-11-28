# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the W8A16 block-FP8 skinny GEMM and its SM90 plans."""

import pytest
import torch

from vllm.model_executor.kernels.linear.cute_dsl import (
    skinny_gemm_plans as skinny_plans,
)
from vllm.model_executor.kernels.linear.cute_dsl.skinny_fp8_gemm import (
    QUANT_BLOCK,
    can_use_skinny_fp8_gemm,
    shape_dynamic_skinny_fp8_gemm,
)
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm import flashinfer as fi_kernels

SM90_FP8_CASES = [
    pytest.param(n, k, m, config, id=f"{n}x{k}-m{m}")
    for (n, k), plan in skinny_plans.SM90_SKINNY_FP8_GEMM_PLANS.items()
    for m, config in plan.items()
]


def _block_fp8_weight(n: int, k: int, device: str):
    """Random weight quantized per 128x128 block with power-of-two scales."""
    w = torch.randn(n, k, device=device) * 0.02
    n_blocks = (n + QUANT_BLOCK - 1) // QUANT_BLOCK
    padded = torch.nn.functional.pad(w, (0, 0, 0, n_blocks * QUANT_BLOCK - n))
    scale = padded.view(n_blocks, QUANT_BLOCK, k // QUANT_BLOCK, QUANT_BLOCK)
    scale = (scale.abs().amax(dim=(1, 3)) / 448.0).clamp(min=1e-8)
    scale = torch.exp2(torch.ceil(torch.log2(scale))).float().contiguous()
    full_scale = scale.repeat_interleave(QUANT_BLOCK, 0)[:n]
    full_scale = full_scale.repeat_interleave(QUANT_BLOCK, 1)
    wq = (w / full_scale).to(torch.float8_e4m3fn).contiguous()
    return wq, scale, wq.float() * full_scale


def _require_sm90_and_cute() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 block-FP8 skinny plans require SM90")
    if not shape_dynamic_skinny_fp8_gemm.is_available():
        pytest.skip("CuTe DSL is not available")


@pytest.mark.parametrize("n,k,num_tokens,config", SM90_FP8_CASES)
def test_skinny_fp8_plans_are_valid(
    n: int, k: int, num_tokens: int, config: SkinnyGemmConfig
) -> None:
    """Every SM90 plan satisfies the kernel's shape constraints."""
    assert config.num_rows == num_tokens
    assert num_tokens <= skinny_plans.SKINNY_FP8_MAX_TOKENS
    assert config.static_k is None
    assert config.block_size % 32 == 0
    assert config.vector_width in (8, 16)
    assert QUANT_BLOCK % config.outputs_per_block == 0
    assert n % config.outputs_per_block == 0
    assert k % QUANT_BLOCK == 0
    assert k % (config.block_size * config.vector_width) == 0


@pytest.mark.parametrize("n,k,num_tokens,config", SM90_FP8_CASES)
def test_sm90_skinny_fp8_plans_match_reference(
    n: int, k: int, num_tokens: int, config: SkinnyGemmConfig
) -> None:
    """Each SM90 plan matches a dequantized FP32 reference on BF16 input."""
    _require_sm90_and_cute()
    torch.manual_seed(42 + num_tokens)
    weight, scale, ref_weight = _block_fp8_weight(n, k, "cuda")
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")

    output = shape_dynamic_skinny_fp8_gemm(x, weight, scale, config)
    reference = x.float() @ ref_weight.t()

    assert output.dtype == torch.bfloat16
    assert output.shape == (num_tokens, n)
    max_err = (output.float() - reference).abs().max().item()
    assert max_err <= 1e-2 * reference.abs().max().item()


def test_sm90_dynamic_op_uses_skinny_for_planned_shape() -> None:
    """The FlashInfer/DeepGEMM op routes a planned shape through the skinny
    GEMM, which then matches the reference more closely than the W8A8 path."""
    _require_sm90_and_cute()
    if not SM90_FP8_CASES:
        pytest.skip("no SM90 block-FP8 plans")
    n, k, num_tokens, _ = SM90_FP8_CASES[0].values
    torch.manual_seed(0)
    weight, scale, ref_weight = _block_fp8_weight(n, k, "cuda")
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    reference = x.float() @ ref_weight.t()

    assert fi_kernels._skinny_fp8_config(x, weight, scale, 128) is not None
    output = torch.ops.vllm.dynamic_flashinfer_deepgemm_blockscale_gemm(
        x, weight, scale, 128, False
    )
    skinny = shape_dynamic_skinny_fp8_gemm(
        x, weight, scale, fi_kernels._skinny_fp8_config(x, weight, scale, 128)
    )
    torch.testing.assert_close(output, skinny, rtol=0, atol=0)
    max_err = (output.float() - reference).abs().max().item()
    assert max_err <= 1e-2 * reference.abs().max().item()


def test_skinny_fp8_dispatch_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planned token counts select the skinny GEMM; everything else falls back."""
    n, k = 4096, 4096
    config = SkinnyGemmConfig(1, 128, 4, vector_width=8, k_unroll=2)
    monkeypatch.setattr(
        skinny_plans,
        "SKINNY_FP8_GEMM_PLANS_BY_CAPABILITY",
        {(9, 0): {(n, k): {1: config}}},
    )
    monkeypatch.setattr(
        skinny_plans.current_platform, "get_device_capability", lambda: (9, 0)
    )
    monkeypatch.setattr(
        skinny_plans.shape_dynamic_skinny_fp8_gemm, "is_available", lambda: True
    )
    weight = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
    scale = torch.ones(n // QUANT_BLOCK, k // QUANT_BLOCK, dtype=torch.float32)
    x1 = torch.zeros(1, k, dtype=torch.bfloat16)
    x2 = torch.zeros(2, k, dtype=torch.bfloat16)

    assert fi_kernels._skinny_fp8_config(x1, weight, scale, 128) == config
    # Token count without a plan entry.
    assert fi_kernels._skinny_fp8_config(x2, weight, scale, 128) is None
    # Other quantization block size.
    assert fi_kernels._skinny_fp8_config(x1, weight, scale, 64) is None
    # Scale layout other than row-major (N / 128, K / 128).
    assert fi_kernels._skinny_fp8_config(x1, weight, scale.t(), 128) is None
    # Unplanned weight shape.
    assert fi_kernels._skinny_fp8_config(x1, weight[:, :2048], scale, 128) is None


def test_can_use_skinny_fp8_gemm_rejects_bad_inputs() -> None:
    config = SkinnyGemmConfig(1, 128, 4, vector_width=8)
    n, k = 512, 1024
    weight = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
    scale = torch.ones(n // QUANT_BLOCK, k // QUANT_BLOCK, dtype=torch.float32)
    x = torch.zeros(1, k, dtype=torch.bfloat16)
    assert can_use_skinny_fp8_gemm(x, weight, scale, config)
    assert not can_use_skinny_fp8_gemm(x.half(), weight, scale, config)
    assert not can_use_skinny_fp8_gemm(x, weight.to(torch.bfloat16), scale, config)
    assert not can_use_skinny_fp8_gemm(x, weight, scale.half(), config)
    assert not can_use_skinny_fp8_gemm(
        torch.zeros(2, k, dtype=x.dtype), weight, scale, config
    )
    # static_k and other vector widths are BF16-kernel-only features.
    assert not can_use_skinny_fp8_gemm(
        x, weight, scale, SkinnyGemmConfig(1, 128, 4, static_k=k)
    )
    assert not can_use_skinny_fp8_gemm(
        x, weight, scale, SkinnyGemmConfig(1, 128, 4, vector_width=4)
    )
    # K not a multiple of block_size * vector_width.
    assert not can_use_skinny_fp8_gemm(
        torch.zeros(1, 1152, dtype=x.dtype),
        torch.zeros(n, 1152, dtype=weight.dtype),
        torch.ones(n // QUANT_BLOCK, 9),
        config,
    )
