# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM language generation tests."""

import torch

from vllm.platforms import current_platform


def pytest_sessionstart(session):
    """Configure ROCm-specific settings before test session starts."""
    if current_platform.is_rocm():
        torch.set_float32_matmul_precision("high")
