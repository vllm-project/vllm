# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combined TP/PCP shards must load the same heads in every projection."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.models.glm5next.common.kda import (
    _fused_qkv_conv1d,
    _Glm5NextMergedColumnParallelLinear,
)


@pytest.mark.parametrize("rank", range(4))
def test_pcp_merged_projection_preserves_replicated_low_rank_weights(dist_init, rank):
    sizes = [32, 32, 32, 8, 4, 4]
    layer = _Glm5NextMergedColumnParallelLinear(
        8,
        sizes,
        replicated_shard_ids=(4, 5),
        tp_size=4,
        tp_rank=rank,
        bias=False,
        params_dtype=torch.float32,
    )
    weights = [
        torch.arange(size * 8, dtype=torch.float32).view(size, 8) + 1000 * i
        for i, size in enumerate(sizes)
    ]
    for shard_id, weight in enumerate(weights):
        layer.weight.weight_loader(layer.weight, weight, shard_id)
    expected = torch.cat(
        [
            weight if i >= 4 else weight.chunk(4)[rank]
            for i, weight in enumerate(weights)
        ]
    )
    torch.testing.assert_close(layer.weight, expected, rtol=0, atol=0)
    assert layer.quant_method is not None
    layer.quant_method.process_weights_after_loading(layer)
    x = torch.arange(16, dtype=torch.float32).view(2, 8)
    torch.testing.assert_close(layer(x)[0], F.linear(x, expected))


def test_pcp_output_projection_reconstructs_full_projection(dist_init):
    x = torch.arange(64, dtype=torch.float32).view(2, 32) / 64
    weight = torch.arange(256, dtype=torch.float32).view(8, 32) / 256
    outputs = []
    for rank in range(4):
        layer = RowParallelLinear(
            32,
            8,
            bias=False,
            reduce_results=False,
            tp_size=4,
            tp_rank=rank,
            params_dtype=torch.float32,
        )
        layer.weight.weight_loader(layer.weight, weight)
        assert layer.quant_method is not None
        layer.quant_method.process_weights_after_loading(layer)
        outputs.append(layer(x.chunk(4, dim=-1)[rank].contiguous())[0])
    torch.testing.assert_close(sum(outputs), F.linear(x, weight))


@pytest.mark.parametrize("rank", range(4))
def test_pcp_convolution_loads_matching_qkv_head_shards(dist_init, rank):
    conv = _fused_qkv_conv1d(4, 32, 4, rank, prefix="conv1d")
    weights = [
        torch.arange(128, dtype=torch.float32).view(32, 1, 4) + i * 128
        for i in range(3)
    ]
    for shard_id, weight in enumerate(weights):
        conv.weight.weight_loader(conv.weight, weight, shard_id)
    expected = torch.cat([weight.chunk(4)[rank] for weight in weights])
    torch.testing.assert_close(conv.weight, expected, rtol=0, atol=0)
