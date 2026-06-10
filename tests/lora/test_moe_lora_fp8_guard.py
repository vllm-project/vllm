# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MoE-LoRA fp8-activation guard (issue #45101).

When an FP8 base MoE is combined with LoRA, the activations reaching the
Triton MoE-LoRA shrink kernel are fp8, which would force an unsupported mixed
``fp8 x bf16`` ``tl.dot`` and crash during the profiling forward with
``AssertionError: Unsupported lhs dtype fp8e4nv``. The guard turns that cryptic
failure into a clear, actionable error. These tests are CPU-only -- they only
exercise the dtype check, not any kernel.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin import (
    _assert_lora_activation_supported,
)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_moe_lora_rejects_fp8_activations(dtype):
    x = torch.empty((4, 8), dtype=dtype)
    with pytest.raises(NotImplementedError, match="FP8-quantized activations"):
        _assert_lora_activation_supported(x)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_moe_lora_allows_non_fp8_activations(dtype):
    # Non-fp8 activations must pass through without raising.
    _assert_lora_activation_supported(torch.empty((4, 8), dtype=dtype))
