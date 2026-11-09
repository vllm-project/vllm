# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HummingLinearKernel group-size validation (CPU-only).

The Humming GEMM applies a group's scales once its K tile reaches a group
boundary. Group sizes that do not line up with its 128-row K tiles are computed
with the wrong scales, so the kernel must not be selected for them.

Run `pytest tests/kernels/quantization/test_humming_kernel_selection.py`.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.kernels.linear import MPLinearLayerConfig
from vllm.model_executor.kernels.linear.mixed_precision.humming import (
    HummingLinearKernel,
)
from vllm.scalar_type import scalar_types

pytestmark = pytest.mark.cpu_test

_MODULE = "vllm.model_executor.kernels.linear.mixed_precision.humming"


def _config(group_size: int, k: int = 384) -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(k, 256),
        partition_weight_shape=(k, 256),
        weight_type=scalar_types.uint4b8,
        act_type=torch.bfloat16,
        group_size=group_size,
        zero_points=False,
    )


def _can_implement(config: MPLinearLayerConfig) -> tuple[bool, str | None]:
    platform = MagicMock()
    platform.is_cuda.return_value = True
    with (
        patch(f"{_MODULE}.current_platform", platform),
        patch(f"{_MODULE}.has_humming", return_value=True),
    ):
        return HummingLinearKernel.can_implement(config)


@pytest.mark.parametrize("group_size", [16, 32, 64, 128, 256, 384, -1])
def test_accepts_group_sizes_that_tile_the_k_blocks(group_size: int):
    can_implement, reason = _can_implement(_config(group_size))
    assert can_implement, reason


def test_accepts_a_single_group_spanning_k():
    can_implement, reason = _can_implement(_config(96, k=96))
    assert can_implement, reason


@pytest.mark.parametrize(
    "group_size", [8, 24, 40, 48, 56, 80, 96, 112, 160, 192, 224, 320, 448]
)
def test_rejects_group_sizes_that_end_inside_a_k_block(group_size: int):
    can_implement, reason = _can_implement(_config(group_size))
    assert not can_implement
    assert reason is not None
    assert str(group_size) in reason
