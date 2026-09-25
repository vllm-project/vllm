# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MergedColumnParallelLinear TP rank synchronization."""

from unittest.mock import patch
import pytest
import torch

from vllm.model_executor.layers.linear import MergedColumnParallelLinear


@pytest.fixture
def mock_distributed_tp_context():
    """Simulate distributed execution under TP=4 at non-zero rank (rank 3)."""
    with patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=3), \
         patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=4), \
         patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=3), \
         patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=4):
        yield


def test_merged_column_parallel_linear_disable_tp_sync(mock_distributed_tp_context):
    """Verify disable_tp=True resets layer and child parameter TP rank and size to 0 and 1."""
    layer = MergedColumnParallelLinear(
        input_size=128,
        output_sizes=[64, 64],
        bias=True,
        disable_tp=True,
    )

    # Layer-level TP status must be single-rank
    assert layer.tp_rank == 0
    assert layer.tp_size == 1

    # Child weight parameter must be synchronized
    assert hasattr(layer, "weight")
    assert layer.weight.tp_rank == 0
    assert layer.weight.tp_size == 1

    # Child bias parameter must be synchronized
    assert hasattr(layer, "bias")
    assert layer.bias.tp_rank == 0
    assert layer.bias.tp_size == 1


def test_merged_column_parallel_linear_enable_tp_retains_rank(mock_distributed_tp_context):
    """Verify disable_tp=False retains the distributed TP rank and size on weight parameter."""
    layer = MergedColumnParallelLinear(
        input_size=128,
        output_sizes=[64, 64],
        bias=True,
        disable_tp=False,
    )

    assert layer.tp_rank == 3
    assert layer.tp_size == 4
    assert layer.weight.tp_rank == 3
    assert layer.weight.tp_size == 4


def test_merged_column_parallel_linear_load_weight_replicated(mock_distributed_tp_context):
    """Verify loading column shards into replicated weight without slice bounds errors."""
    layer = MergedColumnParallelLinear(
        input_size=128,
        output_sizes=[64, 64],
        bias=False,
        disable_tp=True,
    )

    shard0 = torch.randn(64, 128)
    shard1 = torch.randn(64, 128)

    # Load shard 0 (offset 0, size 64)
    layer.weight.load_merged_column_weight(shard0, shard_id=0, shard_offset=0, shard_size=64)
    # Load shard 1 (offset 64, size 64)
    layer.weight.load_merged_column_weight(shard1, shard_id=1, shard_offset=64, shard_size=64)

    assert torch.equal(layer.weight.data[:64], shard0)
    assert torch.equal(layer.weight.data[64:], shard1)
