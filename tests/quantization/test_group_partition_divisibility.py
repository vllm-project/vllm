# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the group-size / tensor-parallel divisibility check shared by the
compressed-tensors WNA16 / WNA8A8 / W4A8-FP8 schemes (issue #46230).

Run `pytest tests/quantization/test_group_partition_divisibility.py`.
"""

import pytest

from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    verify_group_size_divides_partition,
)


def test_raises_actionable_error_when_partition_not_divisible():
    # input_size_per_partition = input_size // tensor_parallel_size; when that
    # shard is not a whole number of groups, group-quantized scales cannot be
    # partitioned across ranks. Previously this aborted with a bare
    # `AssertionError` and no guidance (issue #46230).
    with pytest.raises(ValueError) as exc_info:
        verify_group_size_divides_partition(
            input_size_per_partition=320,
            group_size=128,
            layer_name="model.layers.0.mlp.down_proj",
        )
    msg = str(exc_info.value)
    # The error must surface the offending values, the failing layer, and the
    # actionable remedy so the user is not left with a cryptic assertion.
    assert "320" in msg
    assert "128" in msg
    assert "tensor_parallel_size" in msg
    assert "model.layers.0.mlp.down_proj" in msg


def test_no_error_when_partition_is_whole_groups():
    # 256 == 2 * 128, so the shard is a whole number of groups: must not raise.
    verify_group_size_divides_partition(
        input_size_per_partition=256,
        group_size=128,
        layer_name="model.layers.0.mlp.down_proj",
    )
