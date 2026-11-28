# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU WNA16 kernel selection guards."""

import pytest
import torch

from vllm.model_executor.kernels.linear import MPLinearLayerConfig
from vllm.model_executor.kernels.linear.mixed_precision import (
    CPUWNA16LinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision import cpu as cpu_kernel
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

pytestmark = pytest.mark.cpu_test

if not current_platform.is_cpu():
    pytest.skip("CPU-only tests", allow_module_level=True)


def _config() -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(4096, 4096),
        partition_weight_shape=(4096, 4096),
        weight_type=scalar_types.uint4b8,
        act_type=torch.bfloat16,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )


def test_cpu_wna16_rejects_unregistered_operator(monkeypatch):
    monkeypatch.setattr(cpu_kernel, "_has_cpu_gemm_wna16", lambda: False)

    can_implement, reason = CPUWNA16LinearKernel.can_implement(_config())

    assert not can_implement
    assert reason == (
        "torch.ops._C.cpu_gemm_wna16 is not registered; CPU WNA16 "
        "requires a build with WNA16 CPU support"
    )


def test_cpu_wna16_accepts_registered_operator(monkeypatch):
    monkeypatch.setattr(cpu_kernel, "_has_cpu_gemm_wna16", lambda: True)

    can_implement, reason = CPUWNA16LinearKernel.can_implement(_config())

    assert can_implement
    assert reason is None
