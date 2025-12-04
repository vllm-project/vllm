# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM pooling tests."""

import os
import warnings

from vllm.platforms import current_platform


def pytest_collection_modifyitems(config, items):
    """Set FLEX_ATTENTION backend for SigLIP tests on ROCm."""
    if not current_platform.is_rocm():
        return

    siglip_tests = [item for item in items if "test_siglip" in item.nodeid]

    if siglip_tests:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"
        warnings.warn(
            "ROCm: Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION for SigLIP tests",
            UserWarning,
            stacklevel=1,
        )
