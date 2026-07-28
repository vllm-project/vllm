# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.lora.utils import get_supported_lora_modules
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.models.inkling.nvidia import moe
from vllm.models.inkling.nvidia.model import _TmlForCausalLMBase
from vllm.platforms import current_platform


def test_gate_loads_directly_into_padded_runtime_weight() -> None:
    gate = moe.InklingGate(
        d_model=4,
        n_routed_experts=5,
        n_shared_experts=2,
        experts_per_token=2,
        route_scale=1.0,
    )
    loaded = torch.arange(28, dtype=gate.weight.dtype).reshape(7, 4)

    gate.weight.weight_loader(gate.weight, loaded)

    assert gate.weight.shape == (8, 4)
    torch.testing.assert_close(gate.weight[:7], loaded)
    torch.testing.assert_close(gate.weight[7], torch.zeros(4))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize(("num_tokens", "expected_calls"), [(1, 1), (64, 1), (65, 0)])
def test_gate_uses_ll_bf16_gemm_through_token_limit(
    monkeypatch, num_tokens, expected_calls
) -> None:
    gate = moe.InklingGate(
        d_model=8,
        n_routed_experts=5,
        n_shared_experts=2,
        experts_per_token=2,
        route_scale=1.0,
    ).to(device="cuda", dtype=torch.bfloat16)
    hidden_states = torch.randn(num_tokens, 8, device="cuda", dtype=torch.bfloat16)
    calls = []

    def fake_ll_bf16_gemm(x, weight):
        calls.append((x, weight))
        return torch.ones(num_tokens, 8, device="cuda", dtype=torch.float32)

    monkeypatch.setattr(
        moe.current_platform, "has_device_capability", lambda capability: True
    )
    monkeypatch.setattr(moe.ll_bf16, "is_available", lambda: True)
    monkeypatch.setattr(moe.ll_bf16, "ll_bf16_gemm", fake_ll_bf16_gemm)

    logits = gate.compute_logits(hidden_states)

    assert len(calls) == expected_calls
    if calls:
        assert calls[0][0] is hidden_states
        assert calls[0][1] is gate.weight
    assert logits.shape == (num_tokens, 8)
    assert logits.dtype == torch.float32


def test_gate_is_not_a_lora_target() -> None:
    model = torch.nn.Module()
    model.gate = moe.InklingGate(
        d_model=4,
        n_routed_experts=5,
        n_shared_experts=2,
        experts_per_token=2,
        route_scale=1.0,
    )

    assert "gate" not in get_supported_lora_modules(model)


def test_custom_embedding_is_not_a_lora_target() -> None:
    model = torch.nn.Module()
    model.embedding_modules = _TmlForCausalLMBase.embedding_modules

    supported = get_supported_lora_modules(model)

    assert "embed_tokens" not in supported
    assert "lm_head" in supported


def test_inkling_mapper_maps_modelopt_exclusions() -> None:
    quant_config = ModelOptNvFp4Config.from_config(
        {
            "quantization": {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "kv_cache_quant_algo": None,
                "exclude_modules": [
                    "model.llm.layers.2.mlp.experts",
                    "model.llm.layers.2.mlp.shared_experts",
                ],
            }
        }
    )

    quant_config.apply_vllm_mapper(
        _TmlForCausalLMBase.hf_to_vllm_mapper.get_unstacked_mapper()
    )

    assert quant_config.is_layer_excluded("model.layers.2.mlp.experts")
    assert quant_config.is_layer_excluded("model.layers.2.mlp.shared_experts")
    assert not quant_config.is_layer_excluded("model.layers.3.mlp.experts")


@pytest.mark.parametrize(("projection", "amax"), [("w13", 4.375), ("w2", 2960.0)])
def test_moe_loads_calibrated_input_scale(projection: str, amax: float) -> None:
    experts = SimpleNamespace(
        w13_input_scale=torch.nn.Parameter(torch.empty(3, 2)),
        w2_input_scale=torch.nn.Parameter(torch.empty(3)),
    )
    layer = SimpleNamespace(experts=SimpleNamespace(routed_experts=experts))

    loaded = moe.InklingMoE.load_expert_weight(
        layer,
        f"experts.{projection}_weight.input_amax",
        torch.tensor([amax]),
    )

    scale = getattr(experts, f"{projection}_input_scale")
    expected = torch.full_like(scale, amax / (448.0 * 6.0))
    torch.testing.assert_close(scale, expected)
    assert loaded == [f"experts.routed_experts.{projection}_input_scale"]


def test_sink_down_projection_is_packed_during_load(monkeypatch) -> None:
    monkeypatch.setattr(moe, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(moe, "get_tensor_model_parallel_rank", lambda: 1)
    sink = moe.InklingSinkExperts(n_experts=2, d_model=3, d_mlp=8)
    loaded = torch.arange(48, dtype=sink.w2_weight.dtype).reshape(2, 3, 8)

    sink.load_weight("w2_weight", loaded)

    expected = loaded[:, :, 4:].permute(1, 0, 2).reshape(3, 8)
    torch.testing.assert_close(sink.w2_weight, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_sink_packed_weight_forward_matches_expert_sum(monkeypatch) -> None:
    monkeypatch.setattr(moe, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(moe, "get_tensor_model_parallel_rank", lambda: 1)
    sink = moe.InklingSinkExperts(n_experts=2, d_model=3, d_mlp=8).to(
        device="cuda", dtype=torch.bfloat16
    )
    torch.manual_seed(1)
    w13 = torch.randn(2, 16, 3, dtype=torch.bfloat16)
    w2 = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    sink.load_weight("w13_weight", w13)
    sink.load_weight("w2_weight", w2)
    x = torch.randn(5, 3, device="cuda", dtype=torch.bfloat16)
    gammas = torch.randn(5, 2, device="cuda")

    output = sink(x, gammas)

    raw = torch.einsum("td,efd->tef", x, w13[:, 8:].to("cuda"))
    hidden = torch.nn.functional.silu(raw[:, :, 0::2].float())
    hidden = (hidden * raw[:, :, 1::2] * gammas[:, :, None]).to(torch.bfloat16)
    expected = torch.einsum("tef,edf->td", hidden, w2[:, :, 4:].to("cuda"))
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
