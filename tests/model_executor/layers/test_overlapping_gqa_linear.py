# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.attention.head_partition import (
    make_attention_head_partition,
)
from vllm.model_executor.layers.linear import QKVParallelLinearOverlappingGQA
from vllm.model_executor.parameter import PerTensorScaleParameter


def test_overlapping_gqa_partition_matches_qwen35_27b_tp3_gate_layout():
    parts = [
        make_attention_head_partition(
            total_num_heads=48,
            total_num_kv_heads=4,
            tp_size=3,
            tp_rank=rank,
        )
        for rank in range(3)
    ]

    assert [p.q_head_indices for p in parts] == [
        tuple(range(0, 16)),
        tuple(range(16, 32)),
        tuple(range(32, 48)),
    ]
    assert [p.kv_head_indices for p in parts] == [
        (0, 0, 0, 1),
        (1, 1, 2, 2),
        (2, 3, 3, 3),
    ]
    assert [p.q_to_local_kv_indices for p in parts] == [
        (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
        (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
        (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
    ]


def test_overlapping_gqa_qkv_loader_duplicates_kv_slots(monkeypatch):
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter

    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 3)

    layer = QKVParallelLinearOverlappingGQA(
        hidden_size=3,
        head_size=2,
        total_num_heads=48,
        total_num_kv_heads=4,
        kv_head_indices=(0, 0, 0, 1),
        bias=False,
        quant_config=None,
        prefix="test.qkv_proj",
    )

    q_weight = torch.arange(48 * 2 * 3, dtype=torch.float32).view(48 * 2, 3)
    k_weight = torch.arange(4 * 2 * 3, dtype=torch.float32).view(4 * 2, 3) + 1000
    v_weight = torch.arange(4 * 2 * 3, dtype=torch.float32).view(4 * 2, 3) + 2000

    layer.weight_loader(layer.weight, q_weight, "q")
    layer.weight_loader(layer.weight, k_weight, "k")
    layer.weight_loader(layer.weight, v_weight, "v")

    q_rows = 16 * 2
    kv_rows = 4 * 2
    actual_q = layer.weight[:q_rows]
    actual_k = layer.weight[q_rows : q_rows + kv_rows]
    actual_v = layer.weight[q_rows + kv_rows :]

    torch.testing.assert_close(actual_q, q_weight[:q_rows])
    torch.testing.assert_close(
        actual_k,
        torch.cat(
            [
                k_weight[0:2],
                k_weight[0:2],
                k_weight[0:2],
                k_weight[2:4],
            ],
            dim=0,
        ),
    )
    torch.testing.assert_close(
        actual_v,
        torch.cat(
            [
                v_weight[0:2],
                v_weight[0:2],
                v_weight[0:2],
                v_weight[2:4],
            ],
            dim=0,
        ),
    )


def test_overlapping_gqa_qkv_loader_adjusts_packed_kv_slots(monkeypatch):
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter
    from vllm.model_executor.parameter import PackedColumnParameter

    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 3)

    layer = QKVParallelLinearOverlappingGQA(
        hidden_size=3,
        head_size=2,
        total_num_heads=48,
        total_num_kv_heads=4,
        kv_head_indices=(0, 0, 0, 1),
        bias=False,
        quant_config=None,
        prefix="test.qkv_proj",
    )
    packed_factor = 2
    param = PackedColumnParameter(
        data=torch.full((layer.output_size_per_partition // packed_factor, 3), -1.0),
        weight_loader=lambda *args, **kwargs: None,
        output_dim=0,
        packed_dim=0,
        packed_factor=packed_factor,
    )
    k_weight = torch.arange(4 * 3, dtype=torch.float32).view(4, 3) + 1000
    v_weight = torch.arange(4 * 3, dtype=torch.float32).view(4, 3) + 2000

    layer.weight_loader_v2(param, k_weight, "k")
    layer.weight_loader_v2(param, v_weight, "v")

    q_rows = 16 * 2 // packed_factor
    kv_rows = 4 * 2 // packed_factor
    actual_k = param[q_rows : q_rows + kv_rows]
    actual_v = param[q_rows + kv_rows : q_rows + kv_rows * 2]

    torch.testing.assert_close(
        actual_k,
        torch.cat(
            [
                k_weight[0:1],
                k_weight[0:1],
                k_weight[0:1],
                k_weight[1:2],
            ],
            dim=0,
        ),
    )
    torch.testing.assert_close(
        actual_v,
        torch.cat(
            [
                v_weight[0:1],
                v_weight[0:1],
                v_weight[0:1],
                v_weight[1:2],
            ],
            dim=0,
        ),
    )


def test_overlapping_gqa_qkv_loader_preserves_scalar_scale_slots(monkeypatch):
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter

    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 3)

    layer = QKVParallelLinearOverlappingGQA(
        hidden_size=3,
        head_size=2,
        total_num_heads=48,
        total_num_kv_heads=4,
        kv_head_indices=(1, 1, 2, 2),
        bias=False,
        quant_config=None,
        prefix="test.qkv_proj",
    )
    param = PerTensorScaleParameter(
        data=torch.full((3,), -1.0),
        weight_loader=layer.weight_loader_v2,
    )

    layer.weight_loader_v2(param, torch.tensor(1.25), "q")
    layer.weight_loader_v2(param, torch.tensor(2.5), "k")
    layer.weight_loader_v2(param, torch.tensor(5.0), "v")

    torch.testing.assert_close(param, torch.tensor([1.25, 2.5, 5.0]))
