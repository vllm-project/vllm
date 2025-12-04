# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM tests."""

import warnings

import torch

from vllm.platforms import current_platform


def pytest_configure(config):
    """Disable Flash/MemEfficient SDP on ROCm to avoid HF
    Transformers accuracy issues.
    """
    if not current_platform.is_rocm():
        return

    skip_patterns = ["test_granite_speech.py"]
    if any(pattern in str(arg) for arg in config.args for pattern in skip_patterns):
        # Skip disabling SDP for Granite Speech tests on ROCm
        return

    # Disable Flash/MemEfficient SDP on ROCm to avoid HF Transformers
    # accuracy issues
    # TODO: Remove once ROCm SDP accuracy issues are resolved on HuggingFace
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    warnings.warn(
        "ROCm: Disabled flash_sdp and mem_efficient_sdp, enabled math_sdp "
        "to avoid HuggingFace Transformers accuracy issues",
        UserWarning,
        stacklevel=1,
    )
