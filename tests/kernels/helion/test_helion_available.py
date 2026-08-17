# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Helion kernel availability and basic functionality.

This module demonstrates the pattern for testing optional Helion kernels.
Tests in this directory will be skipped if Helion is not installed.
"""

import pytest

from vllm.utils.import_utils import has_helion

# Skip entire module if helion is not available
if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

import helion
import helion.language as hl
import torch


def test_helion_kernel_compilation_smoke():
    """Smoke test: compile and run a simple Helion kernel."""

    @helion.kernel(autotune_effort="none")
    def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.size()):
            out[tile] = x[tile] + y[tile]
        return out

    # Create test tensors
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    y = torch.randn(1024, device="cuda", dtype=torch.float32)

    # Run the helion kernel
    result = add_kernel(x, y)

    # Verify correctness
    expected = x + y
    assert torch.allclose(result, expected), "Helion kernel output mismatch"
