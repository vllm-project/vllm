# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.linear import (
    ExplicitPaddedMergedColumnParallelLinear,
    PaddedMergedColumnParallelLinear,
    PaddedRowParallelLinear,
)
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


def test_gdn_partition_matches_regular_qwen35_tp2_split():
    parts = [
        make_gdn_head_partition(
            num_k_heads=16,
            num_v_heads=48,
            head_k_dim=128,
            head_v_dim=128,
            tp_size=2,
            tp_rank=rank,
        )
        for rank in range(2)
    ]

    assert [(p.k_start, p.k_count) for p in parts] == [(0, 8), (8, 8)]
    assert [(p.v_start, p.v_count) for p in parts] == [(0, 24), (24, 24)]
    assert [p.local_key_dim for p in parts] == [1024, 1024]
    assert [p.local_value_dim for p in parts] == [3072, 3072]
    assert [p.padded_qkvz_output_sizes for p in parts] == [
        [2048, 2048, 6144, 6144],
        [2048, 2048, 6144, 6144],
    ]


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


def test_explicit_merged_loader_adjusts_packed_output_fused_checkpoint(monkeypatch):
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter
    from vllm.model_executor.parameter import PackedColumnParameter

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
    packed_factor = 8
    local_padded_rows = sum(layer.output_sizes) // layer.tp_size
    param = PackedColumnParameter(
        data=torch.full((local_padded_rows // packed_factor, 2), -1.0),
        weight_loader=lambda *args, **kwargs: None,
        output_dim=0,
        packed_dim=0,
        packed_factor=packed_factor,
    )
    loaded_rows = sum(layer.logical_output_sizes) // packed_factor
    loaded = torch.arange(loaded_rows * 2, dtype=torch.float32).view(-1, 2)

    layer.weight_loader_v2(param, loaded, None)

    offsets = [0, 768, 1536, 3840]
    sizes = [640, 640, 1920, 1920]
    source_offsets = [1408, 2048 + 1408, 4096 + 4224, 10240 + 4224]
    for dest, size, source in zip(offsets, sizes, source_offsets):
        dest //= packed_factor
        size //= packed_factor
        source //= packed_factor
        torch.testing.assert_close(param[dest : dest + size],
                                   loaded[source : source + size])

    assert torch.count_nonzero(param[80:96]) == 0
    assert torch.count_nonzero(param[176:192]) == 0
    assert torch.count_nonzero(param[432:480]) == 0
    assert torch.count_nonzero(param[720:]) == 0


def test_explicit_merged_loader_reconstructs_wna16_quant_params_tp3(monkeypatch):
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )

    packed_factor = 8
    groups = 160
    logical_sizes = [2048, 2048, 6144, 6144]
    padded_sizes = [2304, 2304, 6912, 6912]
    local_starts_by_rank = [
        [0, 0, 0, 0],
        [768, 768, 2304, 2304],
        [1408, 1408, 4224, 4224],
    ]
    local_sizes_by_rank = [
        [768, 768, 2304, 2304],
        [640, 640, 1920, 1920],
        [640, 640, 1920, 1920],
    ]
    logical_total = sum(logical_sizes)
    padded_local_total = sum(padded_sizes) // 3

    loaded_zp = torch.arange(logical_total // packed_factor * groups,
                             dtype=torch.int32).view(-1, groups)
    loaded_scales = torch.arange(logical_total * groups,
                                 dtype=torch.float32).view(-1, groups)

    for rank in range(3):
        monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
        monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda r=rank: r)
        monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank",
                            lambda r=rank: r)
        monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size",
                            lambda: 3)

        layer = ExplicitPaddedMergedColumnParallelLinear(
            input_size=5120,
            output_sizes=logical_sizes,
            padded_output_sizes=padded_sizes,
            local_starts=local_starts_by_rank[rank],
            local_sizes=local_sizes_by_rank[rank],
            bias=False,
            quant_config=None,
            prefix="test.gdn_qkvz",
        )
        qzeros = PackedvLLMParameter(
            data=torch.full((padded_local_total // packed_factor, groups),
                            -1,
                            dtype=torch.int32),
            weight_loader=lambda *args, **kwargs: None,
            input_dim=1,
            output_dim=0,
            packed_dim=0,
            packed_factor=packed_factor,
        )
        scales = GroupQuantScaleParameter(
            data=torch.full((padded_local_total, groups), -1.0),
            weight_loader=lambda *args, **kwargs: None,
            input_dim=1,
            output_dim=0,
        )

        layer.weight_loader_v2(qzeros, loaded_zp, None)
        layer.weight_loader_v2(scales, loaded_scales, None)

        dst = 0
        src_base = 0
        for shard, logical_size in enumerate(logical_sizes):
            padded = padded_sizes[shard] // 3
            local_start = local_starts_by_rank[rank][shard]
            local_size = local_sizes_by_rank[rank][shard]
            if local_size:
                zp_dst = dst // packed_factor
                zp_src = (src_base + local_start) // packed_factor
                zp_size = local_size // packed_factor
                torch.testing.assert_close(qzeros[zp_dst : zp_dst + zp_size],
                                           loaded_zp[zp_src : zp_src + zp_size])
                torch.testing.assert_close(scales[dst : dst + local_size],
                                           loaded_scales[
                                               src_base
                                               + local_start : src_base
                                               + local_start
                                               + local_size])
            assert torch.count_nonzero(qzeros[
                (dst + local_size) // packed_factor : (dst + padded)
                // packed_factor]) == 0
            assert torch.count_nonzero(scales[dst + local_size : dst + padded]) == 0
            dst += padded
            src_base += logical_size


def test_explicit_merged_loader_handles_qwen35_awq_qkv_tuple_then_z(monkeypatch):
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )

    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 3)

    logical_sizes = [2048, 2048, 6144, 6144]
    padded_sizes = [2304, 2304, 6912, 6912]
    local_starts = [1408, 1408, 4224, 4224]
    local_sizes = [640, 640, 1920, 1920]
    packed_factor = 8
    groups = 160
    layer = ExplicitPaddedMergedColumnParallelLinear(
        input_size=5120,
        output_sizes=logical_sizes,
        padded_output_sizes=padded_sizes,
        local_starts=local_starts,
        local_sizes=local_sizes,
        bias=False,
        quant_config=None,
        prefix="test.gdn_qkvz",
    )
    local_total = sum(layer.output_sizes) // layer.tp_size
    qzeros = PackedvLLMParameter(
        data=torch.full((local_total // packed_factor, groups),
                        -1,
                        dtype=torch.int32),
        weight_loader=lambda *args, **kwargs: None,
        input_dim=1,
        output_dim=0,
        packed_dim=0,
        packed_factor=packed_factor,
    )
    scales = GroupQuantScaleParameter(
        data=torch.full((local_total, groups), -1.0),
        weight_loader=lambda *args, **kwargs: None,
        input_dim=1,
        output_dim=0,
    )

    qkv_logical = sum(logical_sizes[:3])
    z_logical = logical_sizes[3]
    qkv_zp = torch.arange((qkv_logical // packed_factor) * groups,
                          dtype=torch.int32).view(-1, groups)
    qkv_scales = torch.arange(qkv_logical * groups,
                              dtype=torch.float32).view(-1, groups)
    z_zp = torch.arange((z_logical // packed_factor) * groups,
                        dtype=torch.int32).view(-1, groups) + 10_000_000
    z_scales = torch.arange(z_logical * groups,
                            dtype=torch.float32).view(-1, groups) + 10_000_000

    layer.weight_loader_v2(qzeros, qkv_zp, (0, 1, 2))
    layer.weight_loader_v2(scales, qkv_scales, (0, 1, 2))
    layer.weight_loader_v2(qzeros, z_zp, 3)
    layer.weight_loader_v2(scales, z_scales, 3)

    dst_offsets = [0, 768, 1536, 3840]
    qkv_src_bases = [0, 2048, 4096]
    for shard, src_base in enumerate(qkv_src_bases):
        dst = dst_offsets[shard]
        local_start = local_starts[shard]
        local_size = local_sizes[shard]
        torch.testing.assert_close(
            qzeros[dst // packed_factor : (dst + local_size) // packed_factor],
            qkv_zp[(src_base + local_start) // packed_factor : (
                src_base + local_start + local_size
            ) // packed_factor],
        )
        torch.testing.assert_close(
            scales[dst : dst + local_size],
            qkv_scales[src_base + local_start : src_base + local_start + local_size],
        )

    z_dst = dst_offsets[3]
    z_local_start = local_starts[3]
    z_local_size = local_sizes[3]
    torch.testing.assert_close(
        qzeros[z_dst // packed_factor : (z_dst + z_local_size) // packed_factor],
        z_zp[z_local_start // packed_factor : (
            z_local_start + z_local_size
        ) // packed_factor],
    )
    torch.testing.assert_close(
        scales[z_dst : z_dst + z_local_size],
        z_scales[z_local_start : z_local_start + z_local_size],
    )


def test_padded_mlp_loader_reconstructs_wna16_gate_up_tp3_rank2(monkeypatch):
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )

    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 3)

    logical = 17408
    padded = 17472
    packed_factor = 8
    groups = 160
    layer = PaddedMergedColumnParallelLinear(
        input_size=5120,
        output_sizes=[logical, logical],
        padded_output_sizes=[padded, padded],
        bias=False,
        quant_config=None,
        prefix="test.mlp.gate_up_proj",
    )
    local_total = sum(layer.output_sizes) // layer.tp_size
    qzeros = PackedvLLMParameter(
        data=torch.full((local_total // packed_factor, groups),
                        -1,
                        dtype=torch.int32),
        weight_loader=lambda *args, **kwargs: None,
        input_dim=1,
        output_dim=0,
        packed_dim=0,
        packed_factor=packed_factor,
    )
    scales = GroupQuantScaleParameter(
        data=torch.full((local_total, groups), -1.0),
        weight_loader=lambda *args, **kwargs: None,
        input_dim=1,
        output_dim=0,
    )
    loaded_zp = torch.arange((logical * 2 // packed_factor) * groups,
                             dtype=torch.int32).view(-1, groups)
    loaded_scales = torch.arange(logical * 2 * groups,
                                 dtype=torch.float32).view(-1, groups)

    layer.weight_loader_v2(qzeros, loaded_zp, None)
    layer.weight_loader_v2(scales, loaded_scales, None)

    shard = padded // 3
    copy = logical - 2 * shard
    assert copy == 5760
    for idx, src_base in enumerate([0, logical]):
        dst = idx * shard
        torch.testing.assert_close(
            qzeros[dst // packed_factor : (dst + copy) // packed_factor],
            loaded_zp[(src_base + 2 * shard) // packed_factor : (
                src_base + 2 * shard + copy
            ) // packed_factor],
        )
        torch.testing.assert_close(
            scales[dst : dst + copy],
            loaded_scales[src_base + 2 * shard : src_base + 2 * shard + copy],
        )
        assert torch.count_nonzero(qzeros[(dst + copy) // packed_factor : (
            dst + shard
        ) // packed_factor]) == 0
        assert torch.count_nonzero(scales[dst + copy : dst + shard]) == 0


def test_padded_row_loader_reconstructs_wna16_down_proj_tp3_rank2(monkeypatch):
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )

    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 3)

    logical = 17408
    padded = 17472
    output = 5120
    packed_factor = 8
    layer = PaddedRowParallelLinear(
        input_size=logical,
        padded_input_size=padded,
        output_size=output,
        bias=False,
        quant_config=None,
        prefix="test.mlp.down_proj",
    )
    qweight = PackedvLLMParameter(
        data=torch.full((output, layer.input_size_per_partition // packed_factor),
                        -1,
                        dtype=torch.int32),
        weight_loader=lambda *args, **kwargs: None,
        input_dim=1,
        output_dim=0,
        packed_dim=1,
        packed_factor=packed_factor,
    )
    scales = GroupQuantScaleParameter(
        data=torch.full((output, layer.input_size_per_partition // 32), -1.0),
        weight_loader=lambda *args, **kwargs: None,
        input_dim=1,
        output_dim=0,
    )
    qzeros = PackedvLLMParameter(
        data=torch.full((output // packed_factor,
                         layer.input_size_per_partition // 32),
                        -1,
                        dtype=torch.int32),
        weight_loader=lambda *args, **kwargs: None,
        input_dim=1,
        output_dim=0,
        packed_dim=0,
        packed_factor=packed_factor,
    )
    loaded_qweight = torch.arange(output * (logical // packed_factor),
                                  dtype=torch.int32).view(output, -1)
    loaded_scales = torch.arange(output * (logical // 32),
                                 dtype=torch.float32).view(output, -1)
    loaded_qzeros = torch.arange((output // packed_factor) * (logical // 32),
                                 dtype=torch.int32).view(output // packed_factor, -1)

    layer.weight_loader_v2(qweight, loaded_qweight)
    layer.weight_loader_v2(scales, loaded_scales)
    layer.weight_loader_v2(qzeros, loaded_qzeros)

    start = 2 * layer.input_size_per_partition
    copy = logical - start
    assert copy == 5760
    torch.testing.assert_close(qweight[:, : copy // packed_factor],
                               loaded_qweight[:, start // packed_factor : (
                                   start + copy
                               ) // packed_factor])
    torch.testing.assert_close(scales[:, : copy // 32],
                               loaded_scales[:, start // 32 : (
                                   start + copy
                               ) // 32])
    torch.testing.assert_close(qzeros[:, : copy // 32],
                               loaded_qzeros[:, start // 32 : (
                                   start + copy
                               ) // 32])
    assert torch.count_nonzero(qweight[:, copy // packed_factor :]) == 0
    assert torch.count_nonzero(scales[:, copy // 32 :]) == 0
    assert torch.count_nonzero(qzeros[:, copy // 32 :]) == 0
