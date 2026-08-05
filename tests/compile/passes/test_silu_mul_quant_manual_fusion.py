# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for manual fusion via maybe_fused_act_quant.

Tests all fusion paths in _FUSED_ACT_QUANT:
- kFp8StaticTensorSym: all platforms
- kFp8Dynamic128Sym: CUDA only
- kNvfp4Dynamic: CUDA SM100+ only
"""

import pytest
import torch

import vllm.envs as envs
from tests.utils import TestFP8Layer
from vllm.config import (
    CompilationConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.kernels.linear import (
    CutlassFP8ScaledMMLinearKernel,
    FlashInferFP8ScaledMMLinearKernel,
    FP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    ROCmFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fusion.fused_act_quant import (
    _FUSED_ACT_QUANT,
    maybe_fused_act_quant,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    expose_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform


# Mock linear layer for testing fusion paths that don't have real kernel support
class MockLinearForFusion(torch.nn.Module):
    """Mock linear layer that exposes input_quant_key for fusion testing."""

    def __init__(self, quant_key, input_scale=None, input_global_scale=None):
        super().__init__()
        self.input_quant_key = quant_key
        if input_scale is not None:
            self.input_scale = input_scale
        if input_global_scale is not None:
            self.input_global_scale = input_global_scale


ROCM_KERNELS = [ROCmFP8ScaledMMLinearKernel, PerTensorTorchFP8ScaledMMLinearKernel]
CUDA_KERNELS = [
    FlashInferFP8ScaledMMLinearKernel,
    CutlassFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
]
TEST_KERNELS = ROCM_KERNELS if current_platform.is_rocm() else CUDA_KERNELS


@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("force_kernel", TEST_KERNELS)
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"], reason="Only test on CUDA and ROCm"
)
def test_manual_fusion_fp8_static_with_linear(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    force_kernel: FP8ScaledMMLinearKernel,
):
    """Test manual fusion with real FP8 linear layer (kFp8StaticTensorSym).

    This is an end-to-end test that verifies the full flow:
    unfused (silu_and_mul -> in-kernel quant) vs fused (silu_and_mul_quant).
    """
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)

    x = torch.rand(num_tokens, hidden_size * 2)

    config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["none"]),
    )

    with set_current_vllm_config(config):
        silu_and_mul = SiluAndMul()
        fp8_linear = TestFP8Layer(
            weight_shape=(hidden_size, hidden_size),
            activation_quant_key=kFp8StaticTensorSym,
            weight_quant_key=kFp8StaticTensorSym,
            force_kernel=force_kernel,
            input_dtype=dtype,
        )

        # Run without fusion: silu_and_mul returns plain tensor
        y_unfused = maybe_fused_act_quant(silu_and_mul, x, fp8_linear)
        assert isinstance(y_unfused, torch.Tensor)
        result_unfused = fp8_linear(y_unfused)

        # Enable fusion
        expose_input_quant_key(fp8_linear, fp8_linear.kernel)

        if not hasattr(fp8_linear, "input_quant_key"):
            pytest.skip(
                f"Kernel {force_kernel.__name__} doesn't support input_quant_key"
            )

        # Run with fusion: silu_and_mul returns QuantizedActivation
        y_fused = maybe_fused_act_quant(silu_and_mul, x, fp8_linear)
        assert isinstance(y_fused, QuantizedActivation)
        assert y_fused.quant_key == kFp8StaticTensorSym
        assert y_fused.data.dtype == current_platform.fp8_dtype()
        assert y_fused.data.shape == (num_tokens, hidden_size)
        assert y_fused.orig_dtype == dtype
        assert y_fused.orig_shape == (num_tokens, hidden_size)

        result_fused = fp8_linear(y_fused)

        torch.testing.assert_close(
            result_fused.to(dtype=dtype),
            result_unfused.to(dtype=dtype),
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Dynamic block quant CUDA only"
)
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"], reason="Only test on CUDA and ROCm"
)
def test_manual_fusion_fp8_dynamic_128(dtype: torch.dtype):
    """Test kFp8Dynamic128Sym fusion path (group_size=128).

    Compares fused (silu_and_mul_per_block_quant) vs unfused (silu_and_mul)
    by dequantizing the fused result and comparing with unfused.
    """
    if (SiluAndMul, kFp8Dynamic128Sym) not in _FUSED_ACT_QUANT:
        pytest.skip("kFp8Dynamic128Sym fusion not available")

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)

    # hidden_size must be divisible by group_size (128)
    num_tokens, hidden_size = 32, 256
    group_size = 128
    x = torch.rand(num_tokens, hidden_size * 2)

    config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["none"]),
    )

    with set_current_vllm_config(config):
        silu_and_mul = SiluAndMul()

        # Unfused path: just apply silu_and_mul
        mock_linear_no_key = torch.nn.Linear(hidden_size, hidden_size)
        result_unfused = maybe_fused_act_quant(silu_and_mul, x, mock_linear_no_key)
        assert isinstance(result_unfused, torch.Tensor)

        # Fused path: apply silu_and_mul + per-block quantization
        mock_linear_with_key = MockLinearForFusion(kFp8Dynamic128Sym)
        result_fused = maybe_fused_act_quant(silu_and_mul, x, mock_linear_with_key)

        # Verify fused result structure
        assert isinstance(result_fused, QuantizedActivation)
        assert result_fused.quant_key == kFp8Dynamic128Sym
        assert result_fused.data.dtype == current_platform.fp8_dtype()
        assert result_fused.data.shape == (num_tokens, hidden_size)
        assert result_fused.orig_dtype == dtype

        # Check scale shape
        expected_num_groups = hidden_size // group_size
        assert result_fused.scale.shape == (num_tokens, expected_num_groups)

        # Dequantize fused result and compare with unfused
        # Per-block dequant: data * scale (broadcast scale across group)
        dequant_data = result_fused.data.to(dtype).view(
            num_tokens, expected_num_groups, group_size
        )
        scales_expanded = result_fused.scale.unsqueeze(
            -1
        )  # (num_tokens, num_groups, 1)
        dequant_result = (
            (dequant_data * scales_expanded).view(num_tokens, hidden_size).to(dtype)
        )

        torch.testing.assert_close(
            dequant_result,
            result_unfused,
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVFP4 CUDA only")
@pytest.mark.skipif(
    not current_platform.has_device_capability(100), reason="NVFP4 requires SM100+"
)
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"], reason="Only test on CUDA and ROCm"
)
def test_manual_fusion_nvfp4_dynamic(dtype: torch.dtype):
    """Test kNvfp4Dynamic fusion path.

    Compares fused (silu_and_mul_nvfp4_quant) vs unfused (silu_and_mul)
    by dequantizing the fused result and comparing with unfused.
    """
    if (SiluAndMul, kNvfp4Dynamic) not in _FUSED_ACT_QUANT:
        pytest.skip("kNvfp4Dynamic fusion not available")

    from tests.kernels.quantization.nvfp4_utils import dequantize_nvfp4_to_dtype

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)

    # NVFP4 requires hidden_size divisible by 16 (block size) and by 2 (packing)
    num_tokens, hidden_size = 32, 128
    x = torch.rand(num_tokens, hidden_size * 2)
    input_global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["none"]),
    )

    with set_current_vllm_config(config):
        silu_and_mul = SiluAndMul()

        # Unfused path: just apply silu_and_mul
        mock_linear_no_key = torch.nn.Linear(hidden_size, hidden_size)
        result_unfused = maybe_fused_act_quant(silu_and_mul, x, mock_linear_no_key)
        assert isinstance(result_unfused, torch.Tensor)

        # Fused path: apply silu_and_mul + NVFP4 quantization
        mock_linear_with_key = MockLinearForFusion(
            kNvfp4Dynamic, input_global_scale=input_global_scale
        )
        result_fused = maybe_fused_act_quant(silu_and_mul, x, mock_linear_with_key)

        # Verify fused result structure
        assert isinstance(result_fused, QuantizedActivation)
        assert result_fused.quant_key == kNvfp4Dynamic
        # NVFP4 packs 2 values into 1 byte
        assert result_fused.data.dtype == torch.uint8
        assert result_fused.data.shape == (num_tokens, hidden_size // 2)
        assert result_fused.orig_dtype == dtype
        assert result_fused.orig_shape == (num_tokens, hidden_size)

        # Dequantize fused result and compare with unfused
        dequant_result = dequantize_nvfp4_to_dtype(
            tensor_fp4=result_fused.data,
            tensor_sf=result_fused.scale,
            global_scale=input_global_scale,
            dtype=dtype,
            device="cuda",
            block_size=16,
            is_sf_128x4_layout=True,
        )

        torch.testing.assert_close(
            dequant_result,
            result_unfused,
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"], reason="Only test on CUDA and ROCm"
)
def test_manual_fusion_fallback_no_key(dtype: torch.dtype):
    """Test that maybe_fused_act_quant falls back when no input_quant_key."""
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)

    x = torch.rand(32, 256)

    config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["none"]),
    )

    with set_current_vllm_config(config):
        silu_and_mul = SiluAndMul()
        # Linear without input_quant_key attribute
        mock_linear = torch.nn.Linear(128, 128)

        result = maybe_fused_act_quant(silu_and_mul, x, mock_linear)

        # Should fall back to plain silu_and_mul
        assert isinstance(result, torch.Tensor)
        assert result.shape == (32, 128)
        assert result.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"], reason="Only test on CUDA and ROCm"
)
def test_manual_fusion_fallback_unsupported_key(dtype: torch.dtype):
    """Test that maybe_fused_act_quant falls back for unsupported quant keys."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        QuantKey,
        ScaleDesc,
    )

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)

    x = torch.rand(32, 256)

    config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["none"]),
    )

    with set_current_vllm_config(config):
        # Create an unsupported quant key
        unsupported_key = QuantKey(
            dtype=torch.int8,
            scale=ScaleDesc(
                dtype=torch.float32, static=False, group_shape=GroupShape(1, 1)
            ),
        )

        silu_and_mul = SiluAndMul()
        mock_linear = MockLinearForFusion(unsupported_key)

        result = maybe_fused_act_quant(silu_and_mul, x, mock_linear)

        # Should fall back to plain silu_and_mul since key not in _FUSED_ACT_QUANT
        assert isinstance(result, torch.Tensor)
        assert result.shape == (32, 128)
        assert result.dtype == dtype
