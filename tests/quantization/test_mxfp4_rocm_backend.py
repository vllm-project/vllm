# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP4 backend selection on ROCm platforms."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_mxfp4_moe_backend,
)


def _make_lora_config():
    """Create a mock FusedMoEConfig with LoRA enabled."""
    config = MagicMock()
    config.is_lora_enabled = True
    config.moe_backend = "auto"
    config.moe_parallel_config.use_batched_activation_format = False
    return config


def _make_non_lora_config():
    """Create a mock FusedMoEConfig with LoRA disabled."""
    config = MagicMock()
    config.is_lora_enabled = False
    config.moe_backend = "auto"
    config.moe_parallel_config.use_batched_activation_format = False
    return config


class TestMxfp4LoraBackendROCm:
    """Test MXFP4 backend selection for LoRA on ROCm platforms."""

    @patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.current_platform"
    )
    @patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.has_triton_kernels"
    )
    def test_rocm_lora_with_triton_kernels(
        self, mock_has_triton, mock_platform
    ):
        mock_platform.is_rocm.return_value = True
        mock_platform.is_cuda.return_value = False
        mock_platform.get_device_capability.return_value = (9, 5)
        mock_has_triton.return_value = True

        config = _make_lora_config()
        backend, experts_cls = select_mxfp4_moe_backend(config)
        assert backend == Mxfp4MoeBackend.TRITON_UNFUSED

    @patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.current_platform"
    )
    @patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.has_triton_kernels"
    )
    def test_rocm_lora_without_triton_raises(
        self, mock_has_triton, mock_platform
    ):
        mock_platform.is_rocm.return_value = True
        mock_platform.is_cuda.return_value = False
        mock_platform.get_device_capability.return_value = (9, 5)
        mock_has_triton.return_value = False

        config = _make_lora_config()
        with pytest.raises(NotImplementedError, match="triton_kernels"):
            select_mxfp4_moe_backend(config)

    @patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.current_platform"
    )
    @patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.has_triton_kernels"
    )
    def test_non_cuda_non_rocm_lora_raises(
        self, mock_has_triton, mock_platform
    ):
        mock_platform.is_rocm.return_value = False
        mock_platform.is_cuda.return_value = False
        mock_platform.get_device_capability.return_value = (0, 0)
        mock_has_triton.return_value = False

        config = _make_lora_config()
        with pytest.raises(
            NotImplementedError, match="only supported on CUDA and ROCm"
        ):
            select_mxfp4_moe_backend(config)


class TestROCmDeviceIdNameMap:
    """Test that MI350X/MI355X device IDs are properly mapped."""

    def test_mi350x_device_id(self):
        from vllm.platforms.rocm import _ROCM_DEVICE_ID_NAME_MAP

        assert "0x75a0" in _ROCM_DEVICE_ID_NAME_MAP
        assert _ROCM_DEVICE_ID_NAME_MAP["0x75a0"] == "AMD_Instinct_MI350X"

    def test_mi355x_device_id(self):
        from vllm.platforms.rocm import _ROCM_DEVICE_ID_NAME_MAP

        assert "0x75a3" in _ROCM_DEVICE_ID_NAME_MAP
        assert _ROCM_DEVICE_ID_NAME_MAP["0x75a3"] == "AMD_Instinct_MI355X"

    def test_existing_mi300x_unchanged(self):
        from vllm.platforms.rocm import _ROCM_DEVICE_ID_NAME_MAP

        assert _ROCM_DEVICE_ID_NAME_MAP["0x74a1"] == "AMD_Instinct_MI300X"

    def test_existing_mi325x_unchanged(self):
        from vllm.platforms.rocm import _ROCM_DEVICE_ID_NAME_MAP

        assert _ROCM_DEVICE_ID_NAME_MAP["0x74a5"] == "AMD_Instinct_MI325X"


class TestGfx950CapabilityParsing:
    """Test that gfx950 arch string is correctly parsed."""

    def test_gfx950_capability(self):
        from vllm.platforms.rocm import _capability_from_gcn_arch

        result = _capability_from_gcn_arch("gfx950")
        assert result == (9, 5)

    def test_gfx942_capability(self):
        from vllm.platforms.rocm import _capability_from_gcn_arch

        result = _capability_from_gcn_arch("gfx942")
        assert result == (9, 4)
