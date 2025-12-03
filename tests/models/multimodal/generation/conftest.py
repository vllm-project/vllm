# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM tests."""

import torch

from vllm.platforms import current_platform


def pytest_configure(config):
    """Disable Flash/MemEfficient SDP on ROCm to avoid HF
    Transformers accuracy issues.
    """
    if not current_platform.is_rocm():
        return

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
