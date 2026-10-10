# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MergedColumnParallelLinear TP rank synchronization
and scale sharding."""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.parameter import GroupQuantScaleParameter


@pytest.fixture
def mock_distributed_tp_context():
    """Simulate distributed execution under TP=4 at non-zero rank (rank 3)."""
    with (
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
            return_value=3,
        ),
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            return_value=4,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=3,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=4,
        ),
    ):
        yield


@pytest.mark.parametrize(
    "disable_tp, expected_rank, expected_size",
    [
        (True, 0, 1),
        (False, 3, 4),
    ],
)
def test_merged_column_parallel_linear_tp_status(
    mock_distributed_tp_context, disable_tp, expected_rank, expected_size
):
    """Verify disable_tp status synchronizes layer and child parameter
    TP rank and size."""
    layer = MergedColumnParallelLinear(
        input_size=128,
        output_sizes=[64, 64],
        bias=True,
        disable_tp=disable_tp,
    )

    assert layer.tp_rank == expected_rank
    assert layer.tp_size == expected_size
    assert layer.weight.tp_rank == expected_rank
    assert layer.weight.tp_size == expected_size
    if disable_tp:
        assert layer.bias.tp_rank == 0
        assert layer.bias.tp_size == 1


def test_merged_column_parallel_linear_load_weight_replicated(
    mock_distributed_tp_context,
):
    """Verify loading column shards into replicated weight without
    slice bounds errors."""
    layer = MergedColumnParallelLinear(
        input_size=128,
        output_sizes=[64, 64],
        bias=False,
        disable_tp=True,
    )

    shard0 = torch.randn(64, 128)
    shard1 = torch.randn(64, 128)

    layer.weight.load_merged_column_weight(
        shard0, shard_id=0, shard_offset=0, shard_size=64
    )
    layer.weight.load_merged_column_weight(
        shard1, shard_id=1, shard_offset=64, shard_size=64
    )

    assert torch.equal(layer.weight.data[:64], shard0)
    assert torch.equal(layer.weight.data[64:], shard1)


def test_merged_column_parallel_linear_scale_shard_adjustment():
    """Verify GroupQuantScaleParameter uses block-aware shard adjustments
    during weight_loader_v2."""
    with (
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        layer = MergedColumnParallelLinear(
            input_size=128, output_sizes=[64, 64], bias=False
        )
        layer.weight_block_size = [32, 32]

        scale_param = GroupQuantScaleParameter(
            data=torch.zeros(4, 4),
            output_dim=0,
            input_dim=1,
            weight_loader=lambda *args: None,
        )
        layer.weight_scale = scale_param
        loaded_scale = torch.ones(2, 4)

        layer.weight_loader_v2(scale_param, loaded_scale, loaded_shard_id=0)
        assert torch.equal(scale_param.data[:2], loaded_scale)
        assert torch.equal(scale_param.data[2:], torch.zeros(2, 4))
