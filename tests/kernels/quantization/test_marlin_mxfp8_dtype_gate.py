# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that MarlinMxfp8LinearKernel.can_implement() rejects float16.

The Marlin MXFP8 CUDA templates are compiled only for bfloat16 activations
(see csrc/.../marlin/generate_kernels.py). Passing float16 at runtime causes
a C++ crash ("Invalid thread config"). This test verifies the Python-level
gate catches the mismatch before hitting the kernel.

Run: pytest tests/kernels/quantization/test_marlin_mxfp8_dtype_gate.py
"""

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
from vllm.model_executor.kernels.linear.mxfp8.marlin import (
    MarlinMxfp8LinearKernel,
)

pytestmark = pytest.mark.cpu_test


def test_marlin_mxfp8_rejects_float16():
    config = Mxfp8LinearLayerConfig(input_dtype=torch.float16)
    ok, reason = MarlinMxfp8LinearKernel.can_implement(config)
    assert not ok, "Marlin MXFP8 should reject float16 (no compiled template)"
    assert reason is not None
    assert "bfloat16" in reason.lower()


def test_marlin_mxfp8_accepts_bfloat16():
    config = Mxfp8LinearLayerConfig(input_dtype=torch.bfloat16)
    ok, reason = MarlinMxfp8LinearKernel.can_implement(config)
    assert ok, f"Marlin MXFP8 should accept bfloat16, got reason: {reason}"
