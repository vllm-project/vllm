# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test for batched triton kernel fallback behavior when deepgemm is unavailable."""

import os

import pytest

from vllm.model_executor.layers.fused_moe.batched_triton_or_deep_gemm_moe import (
    BatchedTritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig


def test_batched_triton_fallback_disabled_by_default():
    """Test that batched triton fallback is disabled by default when deepgemm is requested."""
    # Create a quant config that doesn't support deepgemm (not fp8_w8a8)
    quant_config = FusedMoEQuantConfig(
        use_fp8_w8a8=False,
        use_int8_w8a16=True,
        use_int4_w4a16=False,
    )

    # Make sure the env variable is not set (or set to 0)
    original_value = os.environ.get("VLLM_ALLOW_BATCHED_TRITON_FALLBACK")
    os.environ["VLLM_ALLOW_BATCHED_TRITON_FALLBACK"] = "0"

    try:
        # This should raise RuntimeError because deepgemm was requested
        # but is unavailable, and fallback is disabled
        with pytest.raises(
            RuntimeError, match="DeepGemm was requested but is not available"
        ):
            BatchedTritonOrDeepGemmExperts(
                max_num_tokens=128,
                num_dispatchers=1,
                quant_config=quant_config,
                allow_deep_gemm=True,  # Request deepgemm
            )
    finally:
        # Restore original value
        if original_value is None:
            os.environ.pop("VLLM_ALLOW_BATCHED_TRITON_FALLBACK", None)
        else:
            os.environ["VLLM_ALLOW_BATCHED_TRITON_FALLBACK"] = original_value


def test_batched_triton_fallback_enabled_with_env_var():
    """Test that batched triton fallback works when env variable is set."""
    # Create a quant config that doesn't support deepgemm (not fp8_w8a8)
    quant_config = FusedMoEQuantConfig(
        use_fp8_w8a8=False,
        use_int8_w8a16=True,
        use_int4_w4a16=False,
    )

    # Set the env variable to allow fallback
    original_value = os.environ.get("VLLM_ALLOW_BATCHED_TRITON_FALLBACK")
    os.environ["VLLM_ALLOW_BATCHED_TRITON_FALLBACK"] = "1"

    try:
        # This should NOT raise an error - it should fall back to batched triton
        experts = BatchedTritonOrDeepGemmExperts(
            max_num_tokens=128,
            num_dispatchers=1,
            quant_config=quant_config,
            allow_deep_gemm=True,  # Request deepgemm
        )

        # Verify that deepgemm is not used and batched triton is used instead
        assert experts.batched_deep_gemm_experts is None
        assert experts.batched_triton_experts is not None
    finally:
        # Restore original value
        if original_value is None:
            os.environ.pop("VLLM_ALLOW_BATCHED_TRITON_FALLBACK", None)
        else:
            os.environ["VLLM_ALLOW_BATCHED_TRITON_FALLBACK"] = original_value


def test_batched_triton_no_error_when_deepgemm_not_requested():
    """Test that no error is raised when deepgemm is not requested."""
    # Create a quant config
    quant_config = FusedMoEQuantConfig(
        use_fp8_w8a8=False,
        use_int8_w8a16=True,
        use_int4_w4a16=False,
    )

    # Make sure the env variable is not set (or set to 0)
    original_value = os.environ.get("VLLM_ALLOW_BATCHED_TRITON_FALLBACK")
    os.environ["VLLM_ALLOW_BATCHED_TRITON_FALLBACK"] = "0"

    try:
        # This should NOT raise an error because deepgemm was not requested
        experts = BatchedTritonOrDeepGemmExperts(
            max_num_tokens=128,
            num_dispatchers=1,
            quant_config=quant_config,
            allow_deep_gemm=False,  # Don't request deepgemm
        )

        # Verify that batched triton is used
        assert experts.batched_deep_gemm_experts is None
        assert experts.batched_triton_experts is not None
    finally:
        # Restore original value
        if original_value is None:
            os.environ.pop("VLLM_ALLOW_BATCHED_TRITON_FALLBACK", None)
        else:
            os.environ["VLLM_ALLOW_BATCHED_TRITON_FALLBACK"] = original_value
