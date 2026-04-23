# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for the RMSNorm IR op, including lowering tests.

These tests verify that:
1. supports_args returns bool (not SymBool) with unbacked SymInts
2. Lowering produces numerically correct results compared to baselines
"""

import pytest
import torch

import vllm.kernels  # noqa: F401 to register kernels
from tests.ir.ir_test_utils import (
    assert_op_e2e_correctness,
    assert_supports_args_returns_bool,
    supported_providers,
)
from vllm.ir.ops.layernorm import rms_norm


class TestRmsNormLowering:
    """Tests for RMSNorm lowering behavior."""

    @pytest.mark.parametrize("provider", supported_providers(rms_norm) + ["native"])
    def test_supports_args_returns_bool(self, provider: str):
        """Verify supports_args returns bool with unbacked SymInts."""
        assert_supports_args_returns_bool(
            rms_norm,
            provider,
            num_tokens=8,
            hidden_size=64,
            dtype=torch.bfloat16,
            epsilon=1e-5,
        )


class TestRmsNormE2E:
    """E2E correctness tests comparing lowering with baselines."""

    @pytest.mark.parametrize("provider", supported_providers(rms_norm) + ["native"])
    def test_e2e_correctness(self, provider: str, default_vllm_config):
        """Compare lowering pipeline output with two baselines."""
        real_args = rms_norm.generate_inputs(
            num_tokens=8, hidden_size=16, dtype=torch.bfloat16, epsilon=1e-5
        )
        assert_op_e2e_correctness(rms_norm, provider, real_args)
