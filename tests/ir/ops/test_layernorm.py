# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for the RMSNorm IR op, including lowering tests.

These tests verify that:
1. supports_args returns bool (not SymBool) with unbacked SymInts
"""

import pytest
import torch

import vllm.kernels  # noqa: F401 to register kernels
from vllm.ir.ops.layernorm import rms_norm

from tests.ir.ir_test_utils import (
    assert_supports_args_returns_bool,
    supported_providers,
)


class TestRmsNormLowering:
    """Tests for RMSNorm lowering behavior."""

    @pytest.mark.parametrize("provider", supported_providers(rms_norm) + ["native"])
    def test_supports_args_returns_bool(self, provider: str):
        """Verify supports_args returns bool with unbacked SymInts."""
        assert_supports_args_returns_bool(rms_norm, provider, hidden_size=64)