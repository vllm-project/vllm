# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe
from vllm.platforms.interface import DeviceCapability


@patch("vllm.models.deepseek_v4.nvidia.ops.o_proj.current_platform")
def test_sm12x_uses_hopper_fp32_recipe(mock_platform):
    mock_platform.get_device_capability.return_value = DeviceCapability(
        major=12, minor=1
    )
    recipe, tma_aligned = compute_fp8_einsum_recipe()
    assert recipe == (1, 128, 128)
    assert tma_aligned is False


@patch("vllm.models.deepseek_v4.nvidia.ops.o_proj.current_platform")
def test_sm100_uses_packed_int32_recipe(mock_platform):
    mock_platform.get_device_capability.return_value = DeviceCapability(
        major=10, minor=0
    )
    recipe, tma_aligned = compute_fp8_einsum_recipe()
    assert recipe == (1, 1, 128)
    assert tma_aligned is True


@patch("vllm.models.deepseek_v4.nvidia.ops.o_proj.current_platform")
def test_sm90_uses_hopper_fp32_recipe(mock_platform):
    mock_platform.get_device_capability.return_value = DeviceCapability(
        major=9, minor=0
    )
    recipe, tma_aligned = compute_fp8_einsum_recipe()
    assert recipe == (1, 128, 128)
    assert tma_aligned is False
