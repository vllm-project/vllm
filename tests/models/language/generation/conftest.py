# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM language generation tests."""

import os
import warnings

import torch

from vllm.platforms import current_platform


def pytest_configure(config):
    """Early ROCm configuration that must happen before test collection."""
    if not current_platform.is_rocm():
        return

    # Disable skinny GEMM on ROCm to avoid non-deterministic results
    # from atomic reductions in wvSplitKrc kernel.
    # See: https://github.com/vllm-project/vllm/pull/33493#issuecomment-3906083975
    os.environ["VLLM_ROCM_USE_SKINNY_GEMM"] = "0"
    warnings.warn(
        "ROCm: Set VLLM_ROCM_USE_SKINNY_GEMM=0 to avoid non-deterministic "
        "results from skinny GEMM atomic reductions",
        UserWarning,
        stacklevel=1,
    )


def pytest_sessionstart(session):
    """Configure ROCm-specific settings before test session starts."""
    if not current_platform.is_rocm():
        return

    # Disable Flash/MemEfficient SDP on ROCm to avoid HF Transformers
    # accuracy issues: https://github.com/vllm-project/vllm/issues/30167
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
