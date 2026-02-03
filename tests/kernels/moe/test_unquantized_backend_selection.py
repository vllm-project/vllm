# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    select_unquantized_moe_backend,
)


@pytest.mark.parametrize(
    "platform_method,expected_backend",
    [
        ("is_cuda", UnquantizedMoeBackend.TRITON),  # Default CUDA without FlashInfer
        ("is_rocm", UnquantizedMoeBackend.TRITON),
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
def test_select_default_backend_by_platform(
    mock_has_flashinfer,
    monkeypatch,
    platform_method,
    expected_backend,
):
    """Test backend selection for different platforms."""
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
            use_ep=False,
            use_dp=False,
        )

        assert selected_backend == expected_backend


@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer",
    return_value=True,
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.is_supported_config_trtllm_bf16",
    return_value=(True, None),
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
            use_ep=True,
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
            use_ep=True,  # CUTLASS requires EP
            use_dp=False,  # CUTLASS doesn't support DP
        )

        assert selected_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS
