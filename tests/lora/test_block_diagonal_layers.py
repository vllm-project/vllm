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


def make_bd_lora_config(max_lora_rank=64):
    return LoRAConfig(
        max_lora_rank=max_lora_rank,
        max_loras=4,
        block_diagonal_sharded_loras=True,
    )


def make_regular_lora_config(max_lora_rank=64):
    return LoRAConfig(
        max_lora_rank=max_lora_rank,
        max_loras=4,
        block_diagonal_sharded_loras=False,
    )


class TestMergedColumnBDLoRA:
    """Test MergedColumn BD-LoRA: vanilla LoRA A + block-diagonal LoRA B."""

    @pytest.mark.parametrize("lora_rank", [16, 32, 64, 128])
    def test_can_replace_layer_with_bd_lora(self, dist_init, lora_rank):
        bd_lora_config = make_bd_lora_config(lora_rank)
        source_layer = MergedColumnParallelLinear(
            input_size=1024, output_sizes=[2048, 2048], bias=False
        )

        assert MergedColumnParallelLinearWithBlockDiagonalShardedLoRA.can_replace_layer(
            source_layer, bd_lora_config, ["gate_proj", "up_proj"], None
        )

    @pytest.mark.parametrize("lora_rank", [16, 32, 64, 128])
    def test_cannot_replace_layer_without_bd_lora(self, dist_init, lora_rank):
        regular_lora_config = make_regular_lora_config(lora_rank)
        source_layer = MergedColumnParallelLinear(
            input_size=1024, output_sizes=[2048, 2048], bias=False
        )

        assert not (
            MergedColumnParallelLinearWithBlockDiagonalShardedLoRA.can_replace_layer(
                source_layer, regular_lora_config, ["gate_proj", "up_proj"], None
            )
        )

    @pytest.mark.parametrize("lora_rank", [16, 32, 64, 128])
    @pytest.mark.parametrize(
        "tp_rank,tp_size,output_ids",
        [
            (0, 2, [0, 0]),  # TP2 rank 0
            (1, 2, [1, 1]),  # TP2 rank 1
            (2, 4, [2, 2]),  # TP4 rank 2
        ],
    )
    def test_vanilla_lora_a_slicing(
        self, dist_init, lora_rank, tp_rank, tp_size, output_ids
    ):
        """Test vanilla LoRA A slicing across TP configurations.

        Slices along rank dimension (rows) of LoRA A tensor (lora_rank, 1024).
        Each TP rank gets shard_size = lora_rank // tp_size rows.
        """
        bd_lora_config = make_bd_lora_config(lora_rank)
        base_layer = MergedColumnParallelLinear(
            input_size=1024, output_sizes=[2048, 2048], bias=False
        )

        layer = MergedColumnParallelLinearWithBlockDiagonalShardedLoRA(base_layer)
        layer.lora_config = bd_lora_config
        layer.n_slices = 2
        layer.output_ids = output_ids

        shard_size = lora_rank // tp_size
        layer.lora_a_stacked = [torch.zeros(1, 1, shard_size)]

        lora_a = [torch.randn(lora_rank, 1024), torch.randn(lora_rank, 1024)]
        sliced = layer.slice_lora_a(lora_a)

        # Calculate expected slice based on tp_rank and shard_size
        start_idx = tp_rank * shard_size
        end_idx = start_idx + shard_size
        expected_slice = slice(start_idx, end_idx)

        assert sliced[0].shape == (shard_size, 1024)
        assert sliced[1].shape == (shard_size, 1024)
        assert torch.equal(sliced[0], lora_a[0][expected_slice, :])
        assert torch.equal(sliced[1], lora_a[1][expected_slice, :])

    @pytest.mark.parametrize("lora_rank", [16, 32, 64, 128])
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
        self, dist_init, lora_rank, tp_rank, tp_size, output_ids, expected_slice
    ):
        """Test block-diagonal LoRA B slicing across TP configurations.

        Slices along output dimension (rows) of block-diagonal LoRA B tensor
        (2048, block_rank).  Each TP rank gets
        shard_size = 2048 // tp_size rows.  Block rank is
        lora_rank // tp_size for block diagonal.
        """
        bd_lora_config = make_bd_lora_config(lora_rank)
        base_layer = MergedColumnParallelLinear(
            input_size=1024, output_sizes=[2048, 2048], bias=False
        )

        layer = MergedColumnParallelLinearWithBlockDiagonalShardedLoRA(base_layer)
        layer.lora_config = bd_lora_config
        layer.n_slices = 2
        layer.output_ids = output_ids

        shard_size = 2048 // tp_size
        layer.output_slices = [shard_size, shard_size]

        # Block diagonal rank is lora_rank // tp_size
        block_rank = lora_rank // tp_size
        lora_b = [torch.randn(2048, block_rank), torch.randn(2048, block_rank)]
        sliced = layer.slice_lora_b(lora_b)

        assert len(sliced) == 2
        assert sliced[0].shape == (shard_size, block_rank)
        assert sliced[1].shape == (shard_size, block_rank)
        assert torch.equal(sliced[0], lora_b[0][expected_slice, :])
        assert torch.equal(sliced[1], lora_b[1][expected_slice, :])

    @pytest.mark.parametrize("lora_rank", [16, 32, 64, 128])
    def test_lora_b_stacked_allocation_matches_slicing(self, dist_init, lora_rank):
        """Test that lora_b_stacked allocation matches the slicing output size."""
        bd_lora_config = make_bd_lora_config(lora_rank)
        bd_lora_config.lora_dtype = torch.float16
        base_layer = MergedColumnParallelLinear(
            input_size=1024, output_sizes=[2048, 2048], bias=False
        )

        # Create layer with proper initialization
        layer = MergedColumnParallelLinearWithBlockDiagonalShardedLoRA(base_layer)
        layer.create_lora_weights(4, bd_lora_config, None)  # max_loras=4

        # Test that sliced tensor fits in allocated buffer
        block_rank = lora_rank // layer.tp_size
        lora_b = [torch.randn(2048, block_rank), torch.randn(2048, block_rank)]
        sliced = layer.slice_lora_b(lora_b)

        # Check that sliced tensor shape matches allocated buffer shape
        for i in range(layer.n_slices):
            if sliced[i] is not None:
                allocated_shape = layer.lora_b_stacked[i].shape[2:]  # Skip batch dims
                sliced_shape = sliced[i].shape
                assert allocated_shape == sliced_shape, (
                    f"Allocated {allocated_shape} != Sliced {sliced_shape}"
                )


class TestRowParallelBDLoRA:
    """Test RowParallel BD-LoRA: block-diagonal LoRA A + vanilla LoRA B."""

    @pytest.mark.parametrize("lora_rank", [16, 32, 64, 128])
    def test_can_replace_layer_with_bd_lora(self, dist_init, lora_rank):
        bd_lora_config = make_bd_lora_config(lora_rank)
        source_layer = RowParallelLinear(input_size=2048, output_size=1024, bias=False)

        assert RowParallelLinearWithBlockDiagonalShardedLoRA.can_replace_layer(
            source_layer, bd_lora_config, [], None
        )

    @pytest.mark.parametrize("lora_rank", [16, 32, 64, 128])
    @pytest.mark.parametrize(
        "tp_rank,tp_size",
        [
            (0, 2),  # TP2 rank 0
            (1, 2),  # TP2 rank 1
            (2, 4),  # TP4 rank 2
        ],
    )
    def test_block_diagonal_lora_a_slicing(
        self, dist_init, lora_rank, tp_rank, tp_size
    ):
        """Test block-diagonal LoRA A slicing across TP configurations.

        Block-diagonal LoRA A has shape (FULL_RANK, INPUT_SIZE/TP) where TP blocks
        of size (FULL_RANK/TP, INPUT_SIZE/TP) are stacked along the first dimension.
        Each TP rank gets a slice of size (FULL_RANK/TP, INPUT_SIZE/TP) by slicing
        along the rank dimension (first dimension).
        """
        bd_lora_config = make_bd_lora_config(lora_rank)
        base_layer = RowParallelLinear(input_size=2048, output_size=1024, bias=False)

        layer = RowParallelLinearWithBlockDiagonalShardedLoRA(base_layer)
        layer.lora_config = bd_lora_config
        layer.tp_size = tp_size
        layer.tp_rank = tp_rank

        # Block-diagonal LoRA A has shape (FULL_RANK, INPUT_SIZE/TP)
        input_shard_size = 2048 // tp_size
        lora_a = torch.randn(lora_rank, input_shard_size)
        sliced = layer.slice_lora_a(lora_a)

        rank_shard_size = lora_rank // tp_size
        start_idx = tp_rank * rank_shard_size
        end_idx = start_idx + rank_shard_size
        expected_slice = slice(start_idx, end_idx)

        assert sliced.shape == (rank_shard_size, input_shard_size)
        assert torch.equal(sliced, lora_a[expected_slice, :])

    @pytest.mark.parametrize("lora_rank", [16, 32, 64, 128])
    @pytest.mark.parametrize(
        "tp_rank,tp_size",
        [
            (0, 2),  # TP2 rank 0
            (1, 2),  # TP2 rank 1
            (2, 4),  # TP4 rank 2
        ],
    )
    def test_vanilla_lora_b_slicing(self, dist_init, lora_rank, tp_rank, tp_size):
        """Test vanilla LoRA B slicing across TP configurations.

        Slices along rank dimension (columns) of vanilla LoRA B
        tensor (1024, lora_rank).  Each TP rank gets
        rank_shard_size = lora_rank // tp_size columns.
        Uses tp_rank * rank_shard_size indexing for slice calculation.
        """
        bd_lora_config = make_bd_lora_config(lora_rank)
        base_layer = RowParallelLinear(input_size=2048, output_size=1024, bias=False)

        layer = RowParallelLinearWithBlockDiagonalShardedLoRA(base_layer)
        layer.lora_config = bd_lora_config
        layer.tp_size = tp_size
        layer.tp_rank = tp_rank

        lora_b = torch.randn(1024, lora_rank)
        sliced = layer.slice_lora_b(lora_b)

        rank_shard_size = lora_rank // tp_size
        start_idx = tp_rank * rank_shard_size
        end_idx = start_idx + rank_shard_size
        expected_slice = slice(start_idx, end_idx)

        assert sliced.shape == (1024, rank_shard_size)
        assert torch.equal(sliced, lora_b[:, expected_slice])
