# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that TritonAttentionBackend correctly rejects SM < 80 GPUs.

Regression test for https://github.com/vllm-project/vllm/issues/36357
Triton 3.3+ dropped support for SM < 80 (Volta/Turing), causing multimodal
encoder profiling to hang on V100 and RTX 2080Ti.
"""

import pytest

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend


@pytest.mark.parametrize(
    "capability,expected",
    [
        # SM < 80 should NOT be supported (Volta, Turing)
        (DeviceCapability(7, 0), False),  # V100
        (DeviceCapability(7, 5), False),  # RTX 2080Ti
        # SM >= 80 should be supported (Ampere+)
        (DeviceCapability(8, 0), True),  # A100
        (DeviceCapability(8, 6), True),  # RTX 3090
        (DeviceCapability(8, 9), True),  # RTX 4090
        (DeviceCapability(9, 0), True),  # H100
    ],
)
def test_triton_attn_compute_capability(capability: DeviceCapability, expected: bool):
    """Triton attention should only support SM >= 80 GPUs."""
    assert TritonAttentionBackend.supports_compute_capability(capability) == expected
