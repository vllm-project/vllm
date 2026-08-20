# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.models.deepseek_v2 as deepseek_v2
import vllm.model_executor.parameter as parameter
import vllm.models.deepseek_v32.attention as attention
from vllm.model_executor.models.deepseek_v2 import DeepSeekV2FusedQkvAProjLinear


def _mock_tp(monkeypatch, rank: int, size: int) -> None:
    monkeypatch.setattr(attention, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(attention, "get_tensor_model_parallel_world_size", lambda: size)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)


def test_fused_q_indexer_projection_loads_mixed_tp_weights(monkeypatch) -> None:
    _mock_tp(monkeypatch, rank=2, size=4)
    layer = attention.DeepseekV32FusedQIndexerProjLinear(
        input_size=8,
        q_output_size=12,
        indexer_output_size=4,
        prefix="self_attn.q_b_proj",
    )
    params = {"self_attn.q_b_proj.weight": layer.weight}
    q_weight = torch.arange(96, dtype=torch.float32).view(12, 8)
    indexer_weight = torch.arange(32, dtype=torch.float32).view(4, 8) + 1000

    attention.try_load_fused_indexer_projection(
        "self_attn.q_b_proj.weight", q_weight, params
    )
    attention.try_load_fused_indexer_projection(
        "self_attn.indexer.wq_b.weight", indexer_weight, params
    )

    expected_weight = torch.cat((q_weight[6:9], indexer_weight))
    torch.testing.assert_close(layer.weight, expected_weight)
    inputs = torch.randn(2, 8)
    torch.testing.assert_close(layer(inputs)[0], inputs @ expected_weight.T)


def test_fused_qkv_a_projection_loads_indexer_weights(monkeypatch) -> None:
    _mock_tp(monkeypatch, rank=0, size=1)
    layer = DeepSeekV2FusedQkvAProjLinear(8, [4, 3, 2, 1])
    params = {"self_attn.fused_qkv_a_proj.weight": layer.weight}
    q_weight = torch.randn(4, 8)
    kv_weight = torch.randn(3, 8)
    wk_weight = torch.randn(2, 8)
    score_weight = torch.randn(1, 8)

    layer.weight.weight_loader(layer.weight, q_weight, 0)
    layer.weight.weight_loader(layer.weight, kv_weight, 1)

    attention.try_load_fused_indexer_projection(
        "self_attn.indexer.wk.weight", wk_weight, params
    )
    attention.try_load_fused_indexer_projection(
        "self_attn.indexer.weights_proj.weight", score_weight, params
    )

    expected_weight = torch.cat((q_weight, kv_weight, wk_weight, score_weight))
    torch.testing.assert_close(layer.weight, expected_weight)
    inputs = torch.randn(2, 8)
    torch.testing.assert_close(layer(inputs)[0], inputs @ expected_weight.T)


def test_fp8_indexer_wk_loader_targets_fused_qkv_a(monkeypatch) -> None:
    _mock_tp(monkeypatch, rank=0, size=1)
    layer = DeepSeekV2FusedQkvAProjLinear(8, [4, 3, 2, 1])
    target_name = "self_attn.fused_qkv_a_proj.weight"
    params = {target_name: layer.weight}
    loaded_params: set[str] = set()
    pending: dict = {}
    weight = torch.randn(2, 8).to(torch.float8_e4m3fn)
    scale = torch.ones(2)
    monkeypatch.setattr(
        deepseek_v2,
        "scaled_dequantize",
        lambda weight, *args, **kwargs: weight.to(torch.bfloat16),
    )

    for name, tensor in (
        ("self_attn.indexer.wk.weight", weight),
        ("self_attn.indexer.wk.weight_scale_inv", scale),
    ):
        assert deepseek_v2._try_load_fp8_indexer_wk(
            name, tensor, pending, params, loaded_params, set()
        )

    torch.testing.assert_close(layer.weight[7:9], weight.float())
    assert loaded_params == {target_name}
