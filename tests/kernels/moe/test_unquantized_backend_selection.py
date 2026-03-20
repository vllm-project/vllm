# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    select_unquantized_moe_backend,
)
from vllm.platforms import current_platform


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
    "vllm.utils.flashinfer.has_flashinfer",
    return_value=False,
)
def test_select_default_backend_by_platform(
    mock_has_flashinfer,
    monkeypatch,
    platform_method,
    expected_backend,
):
    """Test backend selection for different platforms."""

    with (
        patch.object(current_platform, "is_cuda", return_value=False),
        patch.object(current_platform, "is_rocm", return_value=False),
        patch.object(current_platform, "is_cpu", return_value=False),
        patch.object(current_platform, "is_xpu", return_value=False),
        patch.object(current_platform, "is_tpu", return_value=False),
        patch.object(current_platform, "is_out_of_tree", return_value=False),
        patch.object(current_platform, platform_method, return_value=True),
    ):
        moe_config = make_dummy_moe_config()
        selected_backend, expert_cls = select_unquantized_moe_backend(
            moe_config=moe_config
        )

        assert selected_backend == expected_backend
        if expected_backend == UnquantizedMoeBackend.CPU:
            assert expert_cls is None
        else:
            assert expert_cls is not None


@patch(
    "vllm.utils.flashinfer.has_flashinfer",
    return_value=True,
)
@patch(
    "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.TrtLlmBf16Experts.is_supported_config",
    return_value=(True, None),
)
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Only supported on NVIDIA platforms."
)
def test_select_cuda_flashinfer_trtllm_backend(
    mock_has_flashinfer, mock_is_supported_trtllm, monkeypatch
):
    """Test CUDA backend selection when FlashInfer TRTLLM is available and enabled."""
    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(current_platform, "is_rocm", return_value=False),
        patch.object(current_platform, "is_cpu", return_value=False),
        patch.object(current_platform, "is_xpu", return_value=False),
        patch.object(current_platform, "is_tpu", return_value=False),
        patch.object(current_platform, "is_out_of_tree", return_value=False),
        patch.object(current_platform, "has_device_capability", return_value=True),
    ):
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")

        moe_config = make_dummy_moe_config()
        # TRTLLM requires EP and does not support DP
        moe_config.moe_parallel_config.use_ep = True
        moe_config.moe_parallel_config.use_dp = False

        selected_backend, experts_cls = select_unquantized_moe_backend(
            moe_config=moe_config
        )

        assert selected_backend == UnquantizedMoeBackend.FLASHINFER_TRTLLM
        assert experts_cls is not None


@patch(
    "vllm.utils.flashinfer.has_flashinfer",
    return_value=True,
)
@patch(
    "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.TrtLlmBf16Experts.is_supported_config",
    return_value=(False, None),
)
@patch(
    "vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe.FlashInferExperts.is_supported_config",
    return_value=(True, None),
)
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Only supported on NVIDIA platforms."
)
def test_select_cuda_flashinfer_cutlass_backend(
    mock_has_flashinfer,
    mock_is_supported_trtllm,
    mock_is_supported_cutlass,
    monkeypatch,
):
    """Test CUDA backend selection when FlashInfer TRTLLM is not available
    and FlashInfer CUTLASS is available."""
    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(current_platform, "is_rocm", return_value=False),
        patch.object(current_platform, "is_cpu", return_value=False),
        patch.object(current_platform, "is_xpu", return_value=False),
        patch.object(current_platform, "is_tpu", return_value=False),
        patch.object(current_platform, "is_out_of_tree", return_value=False),
        patch.object(current_platform, "has_device_capability", return_value=True),
    ):
        # Enable FlashInfer via env var
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")

        moe_config = make_dummy_moe_config()
        # CUTLASS requires EP and does not support DP
        moe_config.moe_parallel_config.use_ep = True
        moe_config.moe_parallel_config.use_dp = False

        selected_backend, experts_cls = select_unquantized_moe_backend(
            moe_config=moe_config
        )

        assert selected_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS
        assert experts_cls is not None
