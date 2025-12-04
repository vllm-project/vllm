# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM pooling tests."""

import os
import warnings

from vllm.platforms import current_platform


def pytest_configure(config):
    """Set FLEX_ATTENTION backend for SigLIP tests on ROCm."""
    if not current_platform.is_rocm():
        return

    encoder_sa_patterns = ["test_siglip.py"]
    matched = [
        p for p in encoder_sa_patterns if any(p in str(arg) for arg in config.args)
    ]

    if matched:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"
        warnings.warn(
            f"ROCm: Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION for {', '.join(matched)}",
            UserWarning,
            stacklevel=1,
        )
