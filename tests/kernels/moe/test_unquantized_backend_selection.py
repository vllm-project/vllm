# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    select_unquantized_moe_backend,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "platform_method,expected_backend",
    [
        ("is_cuda", UnquantizedMoeBackend.TRITON),  # Default CUDA without FlashInfer
        ("is_rocm", UnquantizedMoeBackend.TRITON),  # ROCm without AITER
        ("is_cpu", UnquantizedMoeBackend.CPU),
        ("is_xpu", UnquantizedMoeBackend.XPU),
        ("is_tpu", UnquantizedMoeBackend.TPU),
        ("is_out_of_tree", UnquantizedMoeBackend.OOT),
    ],
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer",
    return_value=False,
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.rocm_aiter_ops.is_fused_moe_enabled",
    return_value=False,
)
def test_select_default_backend_by_platform(
    mock_aiter_enabled,
    mock_has_flashinfer,
    monkeypatch,
    platform_method,
    expected_backend,
):
    """Test default backend selection per platform with all optional
    accelerators (FlashInfer, AITER) disabled."""
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform"
    ) as mock_platform:
        # Set all platform checks to False
        mock_platform.is_cuda.return_value = False
        mock_platform.is_rocm.return_value = False
        mock_platform.is_cpu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_tpu.return_value = False
        mock_platform.is_out_of_tree.return_value = False

        # Set only the specified platform to True
        getattr(mock_platform, platform_method).return_value = True

        moe_config = make_dummy_moe_config()
        selected_backend = select_unquantized_moe_backend(
            moe_config=moe_config,
            use_dp=False,
        )

        assert selected_backend == expected_backend


@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer",
    return_value=False,
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.rocm_aiter_ops.is_fused_moe_enabled",
    return_value=True,
)
@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific backend selection test"
)
def test_select_rocm_aiter_backend(mock_aiter_enabled, mock_has_flashinfer):
    """Test ROCm backend selection when AITER is available."""
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform"
    ) as mock_platform:
        mock_platform.is_cuda.return_value = False
        mock_platform.is_rocm.return_value = True
        mock_platform.is_cpu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_tpu.return_value = False
        mock_platform.is_out_of_tree.return_value = False

        moe_config = make_dummy_moe_config()
        selected_backend = select_unquantized_moe_backend(
            moe_config=moe_config,
            use_dp=False,
        )

        assert selected_backend == UnquantizedMoeBackend.AITER


@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer",
    return_value=True,
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.is_supported_config_trtllm_bf16",
    return_value=(True, None),
)
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Only supported on NVIDIA platforms."
)
def test_select_cuda_flashinfer_trtllm_backend(
    mock_has_flashinfer, mock_is_supported_trtllm, monkeypatch
):
    """Test CUDA backend selection when FlashInfer TRTLLM is available and enabled."""
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform"
    ) as mock_platform:
        # Set as CUDA platform
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False
        mock_platform.is_cpu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_tpu.return_value = False
        mock_platform.is_out_of_tree.return_value = False

        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")

        moe_config = make_dummy_moe_config()

        selected_backend = select_unquantized_moe_backend(
            moe_config=moe_config,
            use_dp=False,
        )

        assert selected_backend == UnquantizedMoeBackend.FLASHINFER_TRTLLM


@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer",
    return_value=True,
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.is_supported_config_trtllm_bf16",
    return_value=(False, None),
)
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Only supported on NVIDIA platforms."
)
def test_select_cuda_flashinfer_cutlass_backend(
    mock_has_flashinfer, mock_is_supported_trtllm, monkeypatch
):
    """Test CUDA backend selection when FlashInfer TRTLLM is not available
    and FlashInfer CUTLASS is available."""
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform"
    ) as mock_platform:
        # Set as CUDA platform with Hopper capability
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False
        mock_platform.is_cpu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_tpu.return_value = False
        mock_platform.is_out_of_tree.return_value = False
        mock_platform.has_device_capability.return_value = True  # SM90+

        # Enable FlashInfer via env var
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")

        moe_config = make_dummy_moe_config()

        selected_backend = select_unquantized_moe_backend(
            moe_config=moe_config,
            use_dp=False,
        )

        assert selected_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS


@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer",
    return_value=True,
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.is_supported_config_trtllm_bf16",
    return_value=(False, None),
)
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Only supported on NVIDIA platforms."
)
def test_select_cuda_flashinfer_cutlass_backend_with_dp(
    mock_has_flashinfer, mock_is_supported_trtllm, monkeypatch
):
    """Test CUDA backend selection picks FlashInfer CUTLASS when both
    DP and EP are enabled. The runner handles DP communication externally
    so the kernel works correctly with DP+EP."""
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform"
    ) as mock_platform:
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False
        mock_platform.is_cpu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_tpu.return_value = False
        mock_platform.is_out_of_tree.return_value = False
        mock_platform.has_device_capability.return_value = True  # SM90+

        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")

        moe_config = make_dummy_moe_config()

        selected_backend = select_unquantized_moe_backend(
            moe_config=moe_config,
            use_dp=True,
        )

        assert selected_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS


@pytest.mark.parametrize(
    "backend,expected_class_name",
    [
        (UnquantizedMoeBackend.FLASHINFER_CUTLASS, "FlashInferExperts"),
        (UnquantizedMoeBackend.TRITON, "TritonExperts"),
    ],
)
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Only supported on NVIDIA platforms."
)
def test_select_gemm_impl_respects_backend(
    backend,
    expected_class_name,
    default_vllm_config,
):
    """Test that select_gemm_impl returns the correct expert implementation
    matching the selected backend, rather than always falling back to Triton.

    This is needed because select_unquantized_moe_backend (tested above) only
    controls which backend enum is *logged*. In DP+EP, maybe_init_modular_kernel
    replaces the quant method with FusedMoEModularMethod, which rebuilds the
    expert kernel via select_gemm_impl. Without a FlashInfer CUTLASS branch
    there, the runtime silently falls back to TritonExperts even though the logs
    say FlashInfer CUTLASS was selected."""
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    moe_config = make_dummy_moe_config()
    method = UnquantizedFusedMoEMethod(moe_config)
    method.unquantized_backend = backend
    method.moe_quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    mock_pf = MagicMock()
    mock_pf.activation_format = FusedMoEActivationFormat.Standard

    result = method.select_gemm_impl(mock_pf, MagicMock())
    assert type(result).__name__ == expected_class_name
