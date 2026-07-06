# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.linear import QKVParallelLinearOverlappingGQA


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
