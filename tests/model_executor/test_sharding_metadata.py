# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for sharding metadata on model parameters."""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.sharding import (
    Sharding,
    ShardingType,
    _attach_sharding,
)


# ── TestShardingDataclass ────────────────────────────────────────────────
class TestShardingDataclass:
    def test_creation(self):
        s = Sharding(
            shape=(1024, 768),
            nd_num_shards=(2, 1),
            sharding_type=ShardingType.COLUMN_WISE,
        )
        assert s.shape == (1024, 768)
        assert s.nd_num_shards == (2, 1)
        assert s.sharding_type == ShardingType.COLUMN_WISE

    def test_frozen(self):
        s = Sharding(
            shape=(1024, 768),
            nd_num_shards=(1, 1),
            sharding_type=ShardingType.REPLICATED,
        )
        with pytest.raises(AttributeError):
            s.shape = (512, 768)  # type: ignore[misc]

    def test_validation_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            Sharding(
                shape=(1024, 768),
                nd_num_shards=(2,),
                sharding_type=ShardingType.COLUMN_WISE,
            )

    def test_validation_shard_lt_one(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            Sharding(
                shape=(1024, 768),
                nd_num_shards=(0, 1),
                sharding_type=ShardingType.COLUMN_WISE,
            )


# ── TestAttachSharding ───────────────────────────────────────────────────
class TestAttachSharding:
    def test_attach_to_parameter(self):
        param = torch.nn.Parameter(torch.empty(4, 4))
        sharding = Sharding(
            shape=(8, 4),
            nd_num_shards=(2, 1),
            sharding_type=ShardingType.COLUMN_WISE,
        )
        _attach_sharding(param, sharding)
        assert hasattr(param, "sharding")
        assert param.sharding is sharding


# ── Helper to mock TP ────────────────────────────────────────────────────
def _mock_tp(tp_rank: int = 0, tp_size: int = 2):
    """Return a context manager that patches TP rank/size for linear layers."""
    return (
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
            return_value=tp_rank,
        ),
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            return_value=tp_size,
        ),
        patch(
            "vllm.model_executor.layers.linear.divide",
            side_effect=lambda a, b: a // b,
        ),
    )


# ── TestLinearSharding ───────────────────────────────────────────────────
class TestLinearSharding:
    def test_replicated_linear(self):
        from vllm.model_executor.layers.linear import ReplicatedLinear

        layer = ReplicatedLinear(768, 1024, bias=False)
        assert hasattr(layer.weight, "sharding")
        s = layer.weight.sharding
        assert s.sharding_type == ShardingType.REPLICATED
        assert s.shape == (1024, 768)
        assert s.nd_num_shards == (1, 1)

    def test_replicated_linear_with_bias(self):
        from vllm.model_executor.layers.linear import ReplicatedLinear

        layer = ReplicatedLinear(768, 1024, bias=True)
        assert layer.bias.sharding.sharding_type == ShardingType.REPLICATED
        assert layer.bias.sharding.shape == (1024,)
        assert layer.bias.sharding.nd_num_shards == (1,)

    def test_column_parallel_linear(self):
        from vllm.model_executor.layers.linear import ColumnParallelLinear

        tp_size = 2
        rank_patch, size_patch, div_patch = _mock_tp(0, tp_size)
        with rank_patch, size_patch, div_patch:
            layer = ColumnParallelLinear(768, 1024, bias=False)
        s = layer.weight.sharding
        assert s.sharding_type == ShardingType.COLUMN_WISE
        assert s.shape == (1024, 768)
        assert s.nd_num_shards == (tp_size, 1)

    def test_column_parallel_linear_bias(self):
        from vllm.model_executor.layers.linear import ColumnParallelLinear

        tp_size = 2
        rank_patch, size_patch, div_patch = _mock_tp(0, tp_size)
        with rank_patch, size_patch, div_patch:
            layer = ColumnParallelLinear(768, 1024, bias=True)
        bs = layer.bias.sharding
        assert bs.sharding_type == ShardingType.COLUMN_WISE
        assert bs.shape == (1024,)
        assert bs.nd_num_shards == (tp_size,)

    def test_row_parallel_linear(self):
        from vllm.model_executor.layers.linear import RowParallelLinear

        tp_size = 4
        rank_patch, size_patch, div_patch = _mock_tp(0, tp_size)
        with rank_patch, size_patch, div_patch:
            layer = RowParallelLinear(1024, 768, bias=False)
        s = layer.weight.sharding
        assert s.sharding_type == ShardingType.ROW_WISE
        assert s.shape == (768, 1024)
        assert s.nd_num_shards == (1, tp_size)

    def test_row_parallel_linear_bias_is_replicated(self):
        from vllm.model_executor.layers.linear import RowParallelLinear

        tp_size = 2
        rank_patch, size_patch, div_patch = _mock_tp(0, tp_size)
        with rank_patch, size_patch, div_patch:
            layer = RowParallelLinear(1024, 768, bias=True)
        bs = layer.bias.sharding
        assert bs.sharding_type == ShardingType.REPLICATED
        assert bs.shape == (768,)
        assert bs.nd_num_shards == (1,)

    def test_qkv_parallel_linear(self):
        from vllm.model_executor.layers.linear import QKVParallelLinear

        tp_size = 2
        rank_patch, size_patch, div_patch = _mock_tp(0, tp_size)
        with rank_patch, size_patch, div_patch:
            layer = QKVParallelLinear(
                hidden_size=768,
                head_size=64,
                total_num_heads=12,
                total_num_kv_heads=4,
                bias=False,
            )
        s = layer.weight.sharding
        assert s.sharding_type == ShardingType.QKV_PARALLEL
        # qk_ratio = 12 // 4 = 3, shape = (3+2, 4, 64, 768)
        assert s.shape == (5, 4, 64, 768)
        assert s.nd_num_shards == (1, tp_size, 1, 1)


# ── TestReplaceParameterPreservation ─────────────────────────────────────
class TestReplaceParameterPreservation:
    def test_utils_replace_parameter_preserves_sharding(self):
        from vllm.model_executor.utils import replace_parameter

        layer = torch.nn.Linear(4, 8, bias=False)
        sharding = Sharding(
            shape=(8, 4),
            nd_num_shards=(2, 1),
            sharding_type=ShardingType.COLUMN_WISE,
        )
        _attach_sharding(layer.weight, sharding)

        new_data = torch.randn(8, 4)
        replace_parameter(layer, "weight", new_data)

        assert hasattr(layer.weight, "sharding")
        assert layer.weight.sharding == sharding

    def test_quant_replace_parameter_preserves_sharding(self):
        from vllm.model_executor.layers.quantization.utils.layer_utils import (
            replace_parameter,
        )

        layer = torch.nn.Module()
        old = torch.nn.Parameter(torch.randn(4, 4))
        sharding = Sharding(
            shape=(8, 4),
            nd_num_shards=(2, 1),
            sharding_type=ShardingType.COLUMN_WISE,
        )
        _attach_sharding(old, sharding)
        layer.register_parameter("w", old)

        # Force the fallback path (different dtype)
        new = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float16))
        replace_parameter(layer, "w", new)

        assert hasattr(layer.w, "sharding")
        assert layer.w.sharding == sharding
