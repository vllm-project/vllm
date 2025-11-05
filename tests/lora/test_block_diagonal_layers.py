# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.config import LoRAConfig
from vllm.lora.layers.block_diagonal_layers import (
    MergedColumnParallelLinearWithBlockDiagonalShardedLoRA,
    RowParallelLinearWithBlockDiagonalShardedLoRA,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)


@pytest.fixture
def bd_lora_config():
    return LoRAConfig(
        max_lora_rank=64,
        max_loras=4,
        block_diagonal_sharded_loras=True,
    )


@pytest.fixture
def regular_lora_config():
    return LoRAConfig(
        max_lora_rank=64,
        max_loras=4,
        block_diagonal_sharded_loras=False,
    )


class TestMergedColumnBDLoRA:
    """Test MergedColumn BD-LoRA: vanilla LoRA A + block-diagonal LoRA B."""

    def test_can_replace_layer_with_bd_lora(self, dist_init, bd_lora_config):
        source_layer = MergedColumnParallelLinear(
            input_size=1024, output_sizes=[2048, 2048], bias=False
        )

        assert MergedColumnParallelLinearWithBlockDiagonalShardedLoRA.can_replace_layer(
            source_layer, bd_lora_config, ["gate_proj", "up_proj"], None
        )

    def test_cannot_replace_layer_without_bd_lora(self, dist_init, regular_lora_config):
        source_layer = MergedColumnParallelLinear(
            input_size=1024, output_sizes=[2048, 2048], bias=False
        )

        assert not MergedColumnParallelLinearWithBlockDiagonalShardedLoRA.can_replace_layer(
            source_layer, regular_lora_config, ["gate_proj", "up_proj"], None
        )

    @pytest.mark.parametrize(
        "tp_rank,tp_size,output_ids,expected_slice",
        [
            (
                0,
                2,
                [0, 0],
                slice(0, 32),
            ),  # TP2 rank 0: shard_size=32, slice rank dim [0:32]
            (
                1,
                2,
                [1, 1],
                slice(32, 64),
            ),  # TP2 rank 1: shard_size=32, slice rank dim [32:64]
            (
                2,
                4,
                [2, 2],
                slice(32, 48),
            ),  # TP4 rank 2: shard_size=16, slice rank dim [32:48]
        ],
    )
    def test_vanilla_lora_a_slicing(
        self, dist_init, bd_lora_config, tp_rank, tp_size, output_ids, expected_slice
    ):
        """Test vanilla LoRA A slicing across TP configurations.

        Slices along rank dimension (rows) of LoRA A tensor (64, 1024).
        Each TP rank gets shard_size = 64 // tp_size rows.
        """
        base_layer = MergedColumnParallelLinear(
            input_size=1024, output_sizes=[2048, 2048], bias=False
        )

        layer = MergedColumnParallelLinearWithBlockDiagonalShardedLoRA(base_layer)
        layer.lora_config = bd_lora_config
        layer.n_slices = 2
        layer.output_ids = output_ids

        shard_size = 64 // tp_size
        layer.lora_a_stacked = [torch.zeros(1, 1, shard_size)]

        lora_a = [torch.randn(64, 1024), torch.randn(64, 1024)]
        sliced = layer.slice_lora_a(lora_a)

        assert sliced[0].shape == (shard_size, 1024)
        assert sliced[1].shape == (shard_size, 1024)
        assert torch.equal(sliced[0], lora_a[0][expected_slice, :])
        assert torch.equal(sliced[1], lora_a[1][expected_slice, :])

    @pytest.mark.parametrize(
        "tp_rank,tp_size,output_ids,expected_slice",
        [
            (
                0,
                2,
                [0, 0],
                slice(0, 1024),
            ),  # TP2 rank 0: shard_size=1024, slice output dim [0:1024]
            (
                1,
                2,
                [1, 1],
                slice(1024, 2048),
            ),  # TP2 rank 1: shard_size=1024, slice output dim [1024:2048]
            (
                2,
                4,
                [2, 2],
                slice(1024, 1536),
            ),  # TP4 rank 2: shard_size=512, slice output dim [1024:1536]
        ],
    )
    def test_block_diagonal_lora_b_slicing(
        self, dist_init, bd_lora_config, tp_rank, tp_size, output_ids, expected_slice
    ):
        """Test block-diagonal LoRA B slicing across TP configurations.

        Slices along output dimension (rows) of block-diagonal LoRA B tensor (2048, 32).
        Each TP rank gets shard_size = 2048 // tp_size rows.
        Uses shard_size * shard_id indexing for slice calculation.
        """
        base_layer = MergedColumnParallelLinear(
            input_size=1024, output_sizes=[2048, 2048], bias=False
        )

        layer = MergedColumnParallelLinearWithBlockDiagonalShardedLoRA(base_layer)
        layer.lora_config = bd_lora_config
        layer.n_slices = 2
        layer.output_ids = output_ids

        shard_size = 2048 // tp_size
        layer.output_slices = [shard_size, shard_size]

        lora_b = [torch.randn(2048, 32), torch.randn(2048, 32)]
        sliced = layer.slice_lora_b(lora_b)

        assert len(sliced) == 2
        assert sliced[0].shape == (shard_size, 32)
        assert sliced[1].shape == (shard_size, 32)
        assert torch.equal(sliced[0], lora_b[0][expected_slice, :])
        assert torch.equal(sliced[1], lora_b[1][expected_slice, :])


class TestRowParallelBDLoRA:
    """Test RowParallel BD-LoRA: block-diagonal LoRA A + vanilla LoRA B."""

    def test_can_replace_layer_with_bd_lora(self, dist_init, bd_lora_config):
        source_layer = RowParallelLinear(input_size=2048, output_size=1024, bias=False)

        assert RowParallelLinearWithBlockDiagonalShardedLoRA.can_replace_layer(
            source_layer, bd_lora_config, [], None
        )

    @pytest.mark.parametrize(
        "tp_rank,tp_size,expected_slice",
        [
            (
                0,
                2,
                slice(0, 1024),
            ),  # TP2 rank 0: input_shard=1024, slice input dim [0:1024]
            (
                1,
                2,
                slice(1024, 2048),
            ),  # TP2 rank 1: input_shard=1024, slice input dim [1024:2048]
            (
                2,
                4,
                slice(1024, 1536),
            ),  # TP4 rank 2: input_shard=512, slice input dim [1024:1536]
        ],
    )
    def test_block_diagonal_lora_a_slicing(
        self, dist_init, bd_lora_config, tp_rank, tp_size, expected_slice
    ):
        """Test block-diagonal LoRA A slicing across TP configurations.

        Slices along input dimension (columns) of block-diagonal LoRA A tensor (32, 2048).
        Each TP rank gets input_shard_size = 2048 // tp_size columns.
        Uses tp_rank * input_shard_size indexing for slice calculation.
        """
        base_layer = RowParallelLinear(input_size=2048, output_size=1024, bias=False)

        layer = RowParallelLinearWithBlockDiagonalShardedLoRA(base_layer)
        layer.lora_config = bd_lora_config
        layer.tp_size = tp_size
        layer.tp_rank = tp_rank

        input_shard_size = 2048 // tp_size
        layer.input_size = input_shard_size

        lora_a = torch.randn(32, 2048)
        sliced = layer.slice_lora_a(lora_a)

        assert sliced.shape == (32, input_shard_size)
        assert torch.equal(sliced, lora_a[:, expected_slice])

    @pytest.mark.parametrize(
        "tp_rank,tp_size,expected_slice",
        [
            (0, 2, slice(0, 32)),  # TP2 rank 0: rank_shard=32, slice rank dim [0:32]
            (1, 2, slice(32, 64)),  # TP2 rank 1: rank_shard=32, slice rank dim [32:64]
            (2, 4, slice(32, 48)),  # TP4 rank 2: rank_shard=16, slice rank dim [32:48]
        ],
    )
    def test_vanilla_lora_b_slicing(
        self, dist_init, bd_lora_config, tp_rank, tp_size, expected_slice
    ):
        """Test vanilla LoRA B slicing across TP configurations.

        Slices along rank dimension (columns) of vanilla LoRA B tensor (1024, 64).
        Each TP rank gets rank_shard_size = 64 // tp_size columns.
        Uses tp_rank * rank_shard_size indexing for slice calculation.
        """
        base_layer = RowParallelLinear(input_size=2048, output_size=1024, bias=False)

        layer = RowParallelLinearWithBlockDiagonalShardedLoRA(base_layer)
        layer.lora_config = bd_lora_config
        layer.tp_size = tp_size
        layer.tp_rank = tp_rank

        lora_b = torch.randn(1024, 64)
        sliced = layer.slice_lora_b(lora_b)

        expected_shard_size = 64 // tp_size
        assert sliced.shape == (1024, expected_shard_size)
        assert torch.equal(sliced, lora_b[:, expected_slice])
