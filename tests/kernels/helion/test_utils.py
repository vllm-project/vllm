# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Helion utility functions."""

import pytest

from vllm.kernels.helion.utils import canonicalize_gpu_name


@pytest.mark.parametrize(
    "driver_reported_name,expected",
    [
        ("NVIDIA H200", "nvidia_h200"),
        ("NVIDIA A100-SXM4-80GB", "nvidia_a100_sxm4_80gb"),
        ("NVIDIA H100 80GB HBM3", "nvidia_h100_80gb_hbm3"),
        ("NVIDIA GeForce RTX 4090", "nvidia_geforce_rtx_4090"),
        ("AMD Instinct MI300X", "amd_instinct_mi300x"),
        ("Tesla V100-SXM2-32GB", "tesla_v100_sxm2_32gb"),
    ],
)
def test_canonicalize_gpu_name(driver_reported_name, expected):
    """Test GPU name canonicalization."""
    assert canonicalize_gpu_name(driver_reported_name) == expected


@pytest.mark.parametrize("invalid_name", ["", "   ", "\t", "\n"])
def test_canonicalize_gpu_name_rejects_empty(invalid_name):
    """Test that empty or whitespace-only names are rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        canonicalize_gpu_name(invalid_name)
