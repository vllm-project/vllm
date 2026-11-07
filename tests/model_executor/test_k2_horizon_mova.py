# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.model_executor.models.k2_horizon as k2_horizon
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.models.k2_horizon import (
    K2HorizonModel,
    K2HorizonMoVAAttention,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed

DEVICE = current_platform.device_type
HIDDEN_SIZE = 256
NUM_KV_HEADS = 2
HEAD_DIM = 64
NUM_EXPERTS = 16
PREFIX = "layers.0.self_attn"


class _DummyAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


def _build_mova_model(monkeypatch) -> tuple[K2HorizonModel, K2HorizonMoVAAttention]:
    monkeypatch.setattr(k2_horizon, "Attention", _DummyAttention)
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device(DEVICE),
    ):
        attn = K2HorizonMoVAAttention(
            vllm_config=vllm_config,
            hidden_size=HIDDEN_SIZE,
            num_heads=4,
            num_kv_heads=NUM_KV_HEADS,
            rope_parameters={"rope_theta": 10000.0},
            rope_head_dim=HEAD_DIM,
            query_key_norm=False,
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=4,
            moe_gate_bias=False,
            router_score_func="sigmoid",
            router_scaling_factor=2.5,
            head_dim=HEAD_DIM,
            prefix=PREFIX,
        )

    model = K2HorizonModel.__new__(K2HorizonModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        head_dim=HEAD_DIM,
        rope_head_dim=HEAD_DIM,
        num_key_value_heads=NUM_KV_HEADS,
        num_experts=0,
    )
    model.num_redundant_experts = 0
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].self_attn = attn
    return model, attn


def _v_expert_weights() -> list[tuple[str, torch.Tensor]]:
    shape = (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE)
    return [
        (f"{PREFIX}.v_experts.{i}.weight", torch.randn(shape, dtype=torch.bfloat16))
        for i in range(NUM_EXPERTS)
    ]


def _stacked(weights: list[tuple[str, torch.Tensor]]) -> torch.Tensor:
    return torch.stack([w for _, w in weights]).to(DEVICE)


@pytest.mark.usefixtures("dist_init")
def test_reloaded_v_experts_update_stacked_weight_in_place(monkeypatch):
    """Weight reloads must reach the stacked experts without moving them, or
    captured CUDA graphs would read stale or freed memory."""
    set_random_seed(0)
    model, attn = _build_mova_model(monkeypatch)

    weights = _v_expert_weights()
    model.load_weights(iter(weights))
    model.process_weights_after_loading()
    stacked = attn.v_experts_weight
    assert stacked is not None
    assert torch.equal(stacked, _stacked(weights))
    data_ptr = stacked.data_ptr()

    reloaded = _v_expert_weights()
    model.load_weights(iter(reloaded))
    model.process_weights_after_loading()
    stacked = attn.v_experts_weight
    assert stacked is not None
    assert stacked.data_ptr() == data_ptr
    assert torch.equal(stacked, _stacked(reloaded))


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="fused_mova_impl needs Triton"
)
@pytest.mark.usefixtures("dist_init")
def test_mova_v_sparse_does_not_copy_v_experts(monkeypatch):
    """compute_mova_v_sparse must read the stacked experts instead of
    re-stacking every expert weight on each forward."""
    set_random_seed(0)
    model, attn = _build_mova_model(monkeypatch)
    router_weight = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, dtype=torch.bfloat16)
    v_experts = _v_expert_weights()
    model.load_weights(iter([(f"{PREFIX}.v_router.weight", router_weight), *v_experts]))
    hidden_states = torch.randn(4, HIDDEN_SIZE, dtype=torch.bfloat16, device=DEVICE)
    expected = attn.compute_mova_v_sparse(hidden_states)

    model.process_weights_after_loading()
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    allocated = torch.accelerator.memory_allocated()
    actual = attn.compute_mova_v_sparse(hidden_states)
    peak = torch.accelerator.max_memory_allocated() - allocated

    torch.testing.assert_close(actual, expected)
    assert peak < sum(w.nbytes for _, w in v_experts)
