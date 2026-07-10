# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.linear import ExplicitPaddedMergedColumnParallelLinear
from vllm.model_executor.layers.mamba.gdn.head_partition import (
    explicit_gdn_conv_weight_loader,
    explicit_vector_weight_loader,
    make_gdn_head_partition,
)


def test_gdn_partition_preserves_value_groups_for_qwopus_tp3():
    parts = [
        make_gdn_head_partition(
            num_k_heads=16,
            num_v_heads=48,
            head_k_dim=128,
            head_v_dim=128,
            tp_size=3,
            tp_rank=rank,
        )
        for rank in range(3)
    ]

    assert [(p.k_start, p.k_count) for p in parts] == [(0, 6), (6, 5), (11, 5)]
    assert [(p.v_start, p.v_count) for p in parts] == [(0, 18), (18, 15), (33, 15)]
    assert [p.local_conv_dim for p in parts] == [3840, 3200, 3200]
    assert [p.padded_conv_dim for p in parts] == [3840, 3840, 3840]


def test_explicit_gdn_conv_loader_pads_rank1_without_stealing_next_group():
    partition = make_gdn_head_partition(
        num_k_heads=16,
        num_v_heads=48,
        head_k_dim=2,
        head_v_dim=2,
        tp_size=3,
        tp_rank=1,
    )
    param = torch.nn.Parameter(torch.full((partition.padded_conv_dim, 1, 2), -1.0))
    loaded_rows = partition.num_k_heads * partition.head_k_dim * 2
    loaded_rows += partition.num_v_heads * partition.head_v_dim
    loaded = torch.arange(loaded_rows * 2, dtype=torch.float32).view(-1, 1, 2)

    explicit_gdn_conv_weight_loader(partition)(param, loaded)

    q0 = 0
    k0 = partition.padded_key_dim
    v0 = partition.padded_key_dim * 2
    key_dim = partition.num_k_heads * partition.head_k_dim
    value_offset = key_dim * 2

    torch.testing.assert_close(
        param[q0 : q0 + partition.local_key_dim],
        loaded[partition.k_start * partition.head_k_dim :][: partition.local_key_dim],
    )
    torch.testing.assert_close(
        param[k0 : k0 + partition.local_key_dim],
        loaded[key_dim + partition.k_start * partition.head_k_dim :][
            : partition.local_key_dim
        ],
    )
    torch.testing.assert_close(
        param[v0 : v0 + partition.local_value_dim],
        loaded[value_offset + partition.v_start * partition.head_v_dim :][
            : partition.local_value_dim
        ],
    )
    assert torch.count_nonzero(param[q0 + partition.local_key_dim : k0]) == 0
    assert torch.count_nonzero(param[k0 + partition.local_key_dim : v0]) == 0
    assert torch.count_nonzero(param[v0 + partition.local_value_dim :]) == 0


def test_explicit_vector_loader_pads_tail():
    param = torch.nn.Parameter(torch.full((18,), -1.0))
    loaded = torch.arange(48, dtype=torch.float32)

    explicit_vector_weight_loader(start=18, size=15)(param, loaded)

    torch.testing.assert_close(param[:15], loaded[18:33])
    assert torch.count_nonzero(param[15:]) == 0


def test_explicit_merged_loader_uses_logical_offsets_for_fused_checkpoint(monkeypatch):
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter

    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 3)

    layer = ExplicitPaddedMergedColumnParallelLinear(
        input_size=2,
        output_sizes=[2048, 2048, 6144, 6144],
        padded_output_sizes=[2304, 2304, 6912, 6912],
        local_starts=[1408, 1408, 4224, 4224],
        local_sizes=[640, 640, 1920, 1920],
        bias=False,
        quant_config=None,
        prefix="test.gdn_qkvz",
    )
    loaded = torch.arange((2048 + 2048 + 6144 + 6144) * 2,
                          dtype=torch.float32).view(-1, 2)

    layer.weight_loader(layer.weight, loaded, None)

    offsets = [0, 768, 1536, 3840]
    sizes = [640, 640, 1920, 1920]
    source_offsets = [1408, 2048 + 1408, 4096 + 4224, 10240 + 4224]
    for dest, size, source in zip(offsets, sizes, source_offsets):
        torch.testing.assert_close(layer.weight[dest : dest + size],
                                   loaded[source : source + size])
