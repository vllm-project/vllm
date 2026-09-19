# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.lora.layers import (
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
)
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def local_shard_layer(request):
    layer_cls = request.param
    layer = object.__new__(layer_cls)
    torch.nn.Module.__init__(layer)
    layer.tp_size = 8
    layer.tp_rank = 3
    layer.lora_config = SimpleNamespace(fully_sharded_loras=False)
    layer.fully_sharded = False
    layer.n_slices = 2 if layer_cls is MergedColumnParallelLinearWithLoRA else 1
    layer.input_size = 2 if layer_cls is RowParallelLinearWithLoRA else 16
    layer.output_size = 24 if layer_cls is RowParallelLinearWithLoRA else 3
    layer.is_merged_col_linear = False
    layer.output_slices = (3, 5)
    layer.output_ids = (3, 3)

    def buffers(shapes):
        return tuple(torch.full(shape, -1, dtype=torch.bfloat16) for shape in shapes)

    if isinstance(layer, FusedMoEWithLoRA):
        layer._w13_slices = 1 if layer_cls is FusedMoE3DWithLoRA else 2
        layer._base_model = "GlmMoeDsaForCausalLM"
        layer.moe_config = SimpleNamespace(intermediate_size_per_partition=2)
        layer.w13_lora_a_stacked = buffers([(2, 3, 8, 16)] * layer._w13_slices)
        layer.w13_lora_b_stacked = buffers(
            [(2, 3, 4 if layer._w13_slices == 1 else 2, 8)] * layer._w13_slices
        )
        layer.w2_lora_a_stacked = buffers([(2, 3, 8, 2)])
        layer.w2_lora_b_stacked = buffers([(2, 3, 16, 8)])
        layer.adapter_enabled = torch.zeros(3, dtype=torch.int)
    else:
        layer.lora_a_stacked = buffers([(2, 1, 8, layer.input_size)] * layer.n_slices)
        outputs = layer.output_slices if layer.n_slices == 2 else (layer.output_size,)
        layer.lora_b_stacked = buffers([(2, 1, output, 8) for output in outputs])
    return layer


LOCAL_SHARD_LAYERS = [
    RowParallelLinearWithLoRA,
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    FusedMoEWithLoRA,
    FusedMoE3DWithLoRA,
]


@pytest.mark.parametrize("local_shard_layer", LOCAL_SHARD_LAYERS, indirect=True)
@pytest.mark.parametrize("tp_rank", range(8))
def test_local_shards_match_global_loading_without_reslicing(
    local_shard_layer, tp_rank
):
    layer = local_shard_layer
    layer.tp_rank = tp_rank
    if layer.n_slices == 2:
        layer.output_ids = (tp_rank, tp_rank)

    def values(shape):
        return (torch.arange(math.prod(shape)).reshape(shape) % 113 + 1).to(
            torch.bfloat16
        )

    rank = 4
    if isinstance(layer, FusedMoEWithLoRA):
        a = [values((3, rank, 16)), values((3, rank, 16))]
        b = [
            values((3, 32 if layer._w13_slices == 1 else 16, rank)),
            values((3, 16, rank)),
        ]
        if layer._w13_slices == 2:
            a.append(values((3, rank, 16)) + 1)
            b.append(values((3, 16, rank)) + 1)
        local_a = [layer._slice_w13_a(a[0]), layer._slice_w2_a(a[1])]
        local_b = [layer._slice_w13_b(b[0]), layer._slice_w2_b(b[1])]
        if layer._w13_slices == 2:
            local_a.append(layer._slice_w13_a(a[2]))
            local_b.append(layer._slice_w13_b(b[2]))
    elif layer.n_slices == 2:
        a = [values((rank, 16)), values((rank, 16)) + 1]
        b = [values((24, rank)), values((40, rank))]
        local_a, local_b = layer.slice_lora_a(a), layer.slice_lora_b(b)
    else:
        a, b = values((rank, 16)), values((24, rank))
        local_a, local_b = [layer.slice_lora_a(a)], [layer.slice_lora_b(b)]

    layer.set_lora(0, a, b)
    expected = [(a.clone(), b.clone()) for a, b in layer._get_lora_shard_buffers(0)]
    layer.set_lora_shard(1, rank, local_a, local_b)
    for (actual_a, actual_b), (expected_a, expected_b) in zip(
        layer._get_lora_shard_buffers(1), expected
    ):
        torch.testing.assert_close(actual_a, expected_a, rtol=0, atol=0)
        torch.testing.assert_close(actual_b, expected_b, rtol=0, atol=0)
    for factor in local_a + local_b:
        factor.zero_()
    for (actual_a, actual_b), (expected_a, expected_b) in zip(
        layer._get_lora_shard_buffers(1), expected
    ):
        torch.testing.assert_close(actual_a, expected_a, rtol=0, atol=0)
        torch.testing.assert_close(actual_b, expected_b, rtol=0, atol=0)


@pytest.mark.parametrize("local_shard_layer", LOCAL_SHARD_LAYERS, indirect=True)
@pytest.mark.parametrize(
    "invalid",
    [
        "shape",
        "dtype",
        "count",
        "rank",
        "fully_sharded",
        "meta",
        "sparse",
        "alias",
        "index",
    ],
)
def test_invalid_local_shards_preserve_existing_slot(local_shard_layer, invalid):
    layer = local_shard_layer
    rank = 4
    shapes = layer.get_lora_shard_shapes(rank)
    a = [torch.ones(a_shape, dtype=torch.bfloat16) for a_shape, _ in shapes]
    b = [torch.ones(b_shape, dtype=torch.bfloat16) for _, b_shape in shapes]
    before = [(a.clone(), b.clone()) for a, b in layer._get_lora_shard_buffers(0)]
    index = 0
    if invalid == "shape":
        b[-1] = b[-1][..., :1, :]
    elif invalid == "dtype":
        b[-1] = b[-1].float()
    elif invalid == "count":
        b.pop()
    elif invalid == "rank":
        rank = 0
    elif invalid == "meta":
        b[-1] = b[-1].to("meta")
    elif invalid == "sparse":
        b[-1] = b[-1].to_sparse()
    elif invalid == "alias":
        b[-1] = layer._get_lora_shard_buffers(0)[-1][1][..., :rank]
    elif invalid == "index":
        index = -1
    else:
        layer.lora_config.fully_sharded_loras = True
        layer.fully_sharded = True
    with pytest.raises((ValueError, NotImplementedError)):
        layer.set_lora_shard(index, rank, a, b)
    layer.lora_config.fully_sharded_loras = False
    layer.fully_sharded = False
    for (actual_a, actual_b), (before_a, before_b) in zip(
        layer._get_lora_shard_buffers(0), before
    ):
        assert torch.equal(actual_a, before_a)
        assert torch.equal(actual_b, before_b)
    if isinstance(layer, FusedMoEWithLoRA):
        assert layer.adapter_enabled[0] == 0
