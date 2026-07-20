# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM language generation tests."""

import warnings

import pytest
import torch

from vllm.platforms import current_platform


@pytest.fixture(scope="package", autouse=True)
def rocm_sdp_config():
    if not current_platform.is_rocm():
        yield
        return

    prev_flash = torch.backends.cuda.flash_sdp_enabled()
    prev_mem = torch.backends.cuda.mem_efficient_sdp_enabled()
    prev_math = torch.backends.cuda.math_sdp_enabled()
    prev_matmul = torch.get_float32_matmul_precision()
    # Disable Flash/MemEfficient SDP on ROCm to avoid HF Transformers
    # accuracy issues: https://github.com/vllm-project/vllm/issues/30167
    # TODO: Remove once ROCm SDP accuracy issues are resolved on HuggingFace
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_float32_matmul_precision("high")
    warnings.warn(
        "ROCm: Disabled flash_sdp and mem_efficient_sdp, enabled math_sdp "
        "to avoid HuggingFace Transformers accuracy issues",
        UserWarning,
        stacklevel=1,
    )

    try:
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(prev_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(prev_mem)
        torch.backends.cuda.enable_math_sdp(prev_math)
        torch.set_float32_matmul_precision(prev_matmul)
