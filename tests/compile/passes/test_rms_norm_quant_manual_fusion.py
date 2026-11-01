# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for manual fusion via maybe_fused_norm_quant.

Tests all fusion paths in _FUSED_NORM_QUANT:
- kFp8StaticTensorSym (with and without residual)
- kFp8DynamicTokenSym (with and without residual)
- kFp8Dynamic128Sym (with and without residual)
- kNvfp4Dynamic (with residual on SM100+)
- Fallback when consumer requires unquantized input (e.g. LoRA) or has no key
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
from vllm.model_executor.layers.fusion.fused_norm_quant import (
    _FUSED_NORM_QUANT,
    maybe_fused_norm_quant,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    expose_input_quant_key,
    get_input_quant_key,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform


class MockLinearForNormFusion(torch.nn.Module):
    """Mock linear layer that exposes an input key and scale for norm fusion tests."""

    def __init__(
        self,
        quant_key,
        input_scale=None,
        input_global_scale_inv=None,
    ):
        super().__init__()
        self._input_quant_key = quant_key
        if input_scale is not None:
            self.input_scale = input_scale
        if input_global_scale_inv is not None:
            self.input_global_scale_inv = input_global_scale_inv


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
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("eps", [1e-6, 1e-4])
@pytest.mark.parametrize("force_kernel", TEST_KERNELS)
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"], reason="Only test on CUDA and ROCm"
)
def test_manual_norm_fusion_fp8_static_with_linear(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    has_residual: bool,
    eps: float,
    force_kernel: type[FP8ScaledMMLinearKernel],
):
    """End-to-end test: RMSNorm + static FP8 quant feeding a real FP8 linear layer."""
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)

    x = torch.randn(num_tokens, hidden_size)
    res_unfused = torch.randn(num_tokens, hidden_size) if has_residual else None
    res_fused = res_unfused.clone() if res_unfused is not None else None

    config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["none"]),
    )

    with set_current_vllm_config(config):
        rms_norm = RMSNorm(hidden_size, eps=eps)
        fp8_linear = TestFP8Layer(
            weight_shape=(hidden_size, hidden_size),
            activation_quant_key=kFp8StaticTensorSym,
            weight_quant_key=kFp8StaticTensorSym,
            force_kernel=force_kernel,
            input_dtype=dtype,
        )

        # 1. Run without fusion (before expose_input_quant_key)
        if res_unfused is None:
            y_unfused = maybe_fused_norm_quant(rms_norm, x, fp8_linear)
        else:
            y_unfused, res_unfused = maybe_fused_norm_quant(
                rms_norm, x, fp8_linear, residual=res_unfused
            )
        assert isinstance(y_unfused, torch.Tensor)
        result_unfused = fp8_linear(y_unfused)

        # 2. Enable fusion via contract
        expose_input_quant_key(fp8_linear, fp8_linear.kernel)
        if get_input_quant_key(fp8_linear) is None:
            pytest.skip(
                f"Kernel {force_kernel.__name__} doesn't support input_quant_key"
            )

        if res_fused is None:
            y_fused = maybe_fused_norm_quant(rms_norm, x, fp8_linear)
        else:
            y_fused, res_fused = maybe_fused_norm_quant(
                rms_norm, x, fp8_linear, residual=res_fused
            )
            assert res_fused.dtype == dtype
            torch.testing.assert_close(res_fused, res_unfused, atol=1e-3, rtol=1e-3)

        assert isinstance(y_fused, QuantizedActivation)
        assert y_fused.quant_key == kFp8StaticTensorSym
        assert y_fused.data.dtype == current_platform.fp8_dtype()
        result_fused = fp8_linear(y_fused)

        torch.testing.assert_close(
            result_fused.to(dtype=dtype),
            result_unfused.to(dtype=dtype),
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 7, 32])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Dynamic block quant CUDA/ROCm only"
)
def test_manual_norm_fusion_fp8_dynamic_128(
    dtype: torch.dtype, has_residual: bool, num_tokens: int
):
    """Test kFp8Dynamic128Sym norm fusion path and verify residual preservation."""
    if (RMSNorm, kFp8Dynamic128Sym) not in _FUSED_NORM_QUANT:
        pytest.skip("kFp8Dynamic128Sym norm fusion not available")

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)

    hidden_size, group_size = 256, 128
    x = torch.randn(num_tokens, hidden_size)
    res_unfused = torch.randn(num_tokens, hidden_size) if has_residual else None
    res_fused = res_unfused.clone() if res_unfused is not None else None

    config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["none"]),
    )

    with set_current_vllm_config(config):
        rms_norm = RMSNorm(hidden_size, eps=1e-6)
        mock_linear_no_key = torch.nn.Linear(hidden_size, hidden_size)
        mock_linear_with_key = MockLinearForNormFusion(kFp8Dynamic128Sym)

        if res_unfused is None:
            result_unfused = maybe_fused_norm_quant(rms_norm, x, mock_linear_no_key)
            result_fused = maybe_fused_norm_quant(rms_norm, x, mock_linear_with_key)
        else:
            result_unfused, res_unfused = maybe_fused_norm_quant(
                rms_norm, x, mock_linear_no_key, residual=res_unfused
            )
            result_fused, res_fused = maybe_fused_norm_quant(
                rms_norm, x, mock_linear_with_key, residual=res_fused
            )
            torch.testing.assert_close(res_fused, res_unfused, atol=1e-3, rtol=1e-3)

        assert isinstance(result_fused, QuantizedActivation)
        expected_num_groups = hidden_size // group_size
        tma_aligned_m = (num_tokens + 3) // 4 * 4
        assert result_fused.scale.shape == (num_tokens, expected_num_groups)
        assert result_fused.scale.stride() == (1, tma_aligned_m)
        dequant_data = result_fused.data.to(dtype).view(
            num_tokens, expected_num_groups, group_size
        )
        dequant_result = (
            (dequant_data * result_fused.scale.unsqueeze(-1))
            .view(num_tokens, hidden_size)
            .to(dtype)
        )
        torch.testing.assert_close(dequant_result, result_unfused, atol=5e-2, rtol=5e-2)


def test_rocm_aiter_deferral_guard(monkeypatch):
    """Verify dynamic FP8 norm+quant fusion defers when ROCm AITER is enabled."""
    import vllm._aiter_ops as aiter_mod
    import vllm.model_executor.layers.fusion.fused_norm_quant as fnq

    monkeypatch.setattr(fnq.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(
        aiter_mod.rocm_aiter_ops, "is_enabled", classmethod(lambda cls: True)
    )
    monkeypatch.setattr(
        aiter_mod.rocm_aiter_ops,
        "is_rdna_aiter_enabled",
        classmethod(lambda cls: False),
    )
    assert fnq._should_defer_to_rocm_aiter_rms_fusion() is True

    monkeypatch.setattr(
        aiter_mod.rocm_aiter_ops, "is_enabled", classmethod(lambda cls: False)
    )
    assert fnq._should_defer_to_rocm_aiter_rms_fusion() is False
