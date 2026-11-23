# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM language generation tests."""

import gc

import pytest
import torch

from tests.utils import wait_for_memory_to_settle
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def pytest_sessionstart(session):
    """Configure ROCm-specific settings before test session starts."""
    if current_platform.is_rocm():
        torch.set_float32_matmul_precision("high")


@pytest.fixture(autouse=True)
def release_gpu_memory_between_tests():
    """Reclaim cached GPU memory between tests in this shard.

    These tests start many vLLM/HF engines sequentially in one pytest
    process; without explicit reclamation between tests, VRAM fragmentation
    accumulates and later tests fail at engine startup ("Engine core
    initialization failed").
    """
    yield
    gc.collect()
    if torch.accelerator.is_available():
        torch.accelerator.empty_cache()
    if current_platform.is_rocm():
        return
    try:
        wait_for_memory_to_settle()
    except ValueError as e:
        # Longer-lived (class/module-scoped) engine fixtures may legitimately
        # still hold VRAM; nothing further to reclaim in that case.
        logger.info("Failed to clean GPU memory: %s", e)
