# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`BeamModel.hf_to_vllm_mapper` must fuse only the dense projections.

Beam's checkpoints carry two `gate_proj` names that are *not* the SwiGLU
gate -- the per-expert `mlp.experts.N.gate_proj` (owned by
`MoERunner.load_weights`) and the attention output gate `self_attn.gate_proj`.
Folding either into `mlp.gate_up_proj` silently corrupts the weights, so pin the
mapping. The attention gate is additionally renamed to `output_gate_proj` so
that `packed_modules_mapping` consumers (LoRA) cannot match it as a
`gate_up_proj` shard either.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.beam.nvidia import model as beam
from vllm.models.beam.nvidia.model import BeamModel
from vllm.transformers_utils.configs.beam import BeamConfig

MAPPER = BeamModel.hf_to_vllm_mapper


def _map(name: str) -> tuple[str, int | str | None]:
    weight = torch.empty(0)
    ((mapped, out),) = MAPPER.apply([(name, weight)])
    return mapped, getattr(out, "shard_id", None)


@pytest.mark.parametrize(
    ("checkpoint_name", "expected_name", "expected_shard"),
    [
        (
            "model.layers.3.self_attn.q_proj.weight",
            "model.layers.3.self_attn.qkv_proj.weight",
            "q",
        ),
        (
            "model.layers.3.self_attn.k_proj.weight",
            "model.layers.3.self_attn.qkv_proj.weight",
            "k",
        ),
        (
            "model.layers.3.self_attn.v_proj.weight",
            "model.layers.3.self_attn.qkv_proj.weight",
            "v",
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
    ],
)
def test_dense_projections_are_fused(checkpoint_name, expected_name, expected_shard):
    assert _map(checkpoint_name) == (expected_name, expected_shard)


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        # Owned by MoERunner.load_weights, which is also what keeps expert
        # weights correct under EP and redundant experts.
        "model.layers.1.mlp.experts.7.gate_proj.weight",
        "model.layers.1.mlp.experts.7.up_proj.weight",
        "model.layers.1.mlp.experts.7.down_proj.weight",
        "model.layers.1.self_attn.gate_bias",
        # The MoE router.
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.gate.e_score_correction_bias",
    ],
)
def test_gates_and_experts_pass_through(checkpoint_name):
    assert _map(checkpoint_name) == (checkpoint_name, None)


def test_attention_output_gate_is_renamed_not_fused():
    """The attention gate must be renamed, and must not pick up a shard id."""
    assert _map("model.layers.1.self_attn.gate_proj.weight") == (
        "model.layers.1.self_attn.output_gate_proj.weight",
        None,
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "module_path,name",
    [("self_attn", "gate_bias"), ("mlp.gate", "e_score_correction_bias")],
)
def test_optional_bias_loads_only_when_present(module_path, name, enabled):
    """Unknown Beam biases must fail instead of silently changing model output."""
    model = BeamModel.__new__(BeamModel)
    nn.Module.__init__(model)
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.gate = nn.Module()
    model.layers = nn.ModuleList([layer])
    target = layer.get_submodule(module_path)
    target.register_parameter(name, nn.Parameter(torch.zeros(2)) if enabled else None)
    weight_name = f"layers.0.{module_path}.{name}"
    weight = torch.tensor([1.0, 2.0])
    weights = [(weight_name, weight)]
    if enabled:
        assert model.load_weights(weights) == {weight_name}
        torch.testing.assert_close(getattr(target, name), weight)
    else:
        with pytest.raises(ValueError, match=name):
            model.load_weights(weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_checkpoint_norm_names_match_sublayer_order(default_vllm_config, monkeypatch):
    """Load reference names directly and use each weight on its intended branch."""

    class Attention(nn.Module):
        def forward(self, positions, hidden_states):
            return hidden_states

    monkeypatch.setattr(beam, "BeamAttention", lambda *args, **kwargs: Attention())
    monkeypatch.setattr(beam, "BeamMLP", lambda **kwargs: nn.Identity())
    config = BeamConfig(
        hidden_size=8,
        mlp_only_layers=[0],
        enable_sandwich_norm=True,
        self_attn_depth_scaled_residual=False,
        mlp_depth_scaled_residual=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        cache_config=None,
        quant_config=None,
    )
    model = BeamModel.__new__(BeamModel)
    nn.Module.__init__(model)
    model.config = config
    layer = beam.BeamDecoderLayer(vllm_config, prefix="layers.0")
    model.layers = nn.ModuleList([layer])
    model.cuda()
    names = [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ]
    weights = {
        f"layers.0.{name}.weight": torch.arange(1, 9, device="cuda").float().roll(i) / 8
        for i, name in enumerate(names)
    }
    assert model.load_weights(weights.items()) == set(weights)
    x = torch.arange(1, 17, device="cuda").float().view(2, 8) / 16

    def norm(value, name):
        return (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            * weights[f"layers.0.{name}.weight"]
        )

    expected = x + norm(norm(x, names[0]), names[1])
    expected = expected + norm(norm(expected, names[2]), names[3])
    actual, _ = layer(torch.arange(2, device="cuda"), x, None)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tp_size", [4, 8])
@torch.inference_mode()
def test_checkpoint_loads_into_merged_qkv_gate(
    default_vllm_config, monkeypatch, tp_size
):
    """All four checkpoint projections load directly into the allocated TP shard."""
    from vllm.models.beam.nvidia import attention
    from vllm.models.beam.nvidia.projection import packed_projection

    for module in (
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(
            f"{module}.get_tensor_model_parallel_rank", lambda: tp_size - 1
        )
        monkeypatch.setattr(
            f"{module}.get_tensor_model_parallel_world_size", lambda: tp_size
        )
    monkeypatch.setattr(
        attention, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    attention_stub = nn.Identity()
    attention_stub.attn_backend = SimpleNamespace(get_name=lambda: "TEST")
    monkeypatch.setattr(attention, "Attention", lambda *args, **kwargs: attention_stub)
    config = BeamConfig(
        hidden_size=64, num_attention_heads=48, num_key_value_heads=8, head_dim=8
    )
    attn = (
        attention.BeamAttention(config, None, None, prefix="layers.0.self_attn")
        .bfloat16()
        .cuda()
    )
    model = BeamModel.__new__(BeamModel)
    nn.Module.__init__(model)
    model.config = config
    model.hf_to_vllm_mapper = model.get_weights_mapper(config, tp_size)
    layer = nn.Module()
    layer.self_attn = attn
    model.layers = nn.ModuleList([layer])
    param = attn.qkv_gate_proj.weight
    initial_pointer = param.data_ptr()
    for _ in range(2):
        weights = [
            torch.randn(n, 64, device="cuda", dtype=torch.bfloat16)
            for n in (384, 64, 64, 384)
        ]
        loaded = model.load_weights(
            (f"layers.0.self_attn.{name}.weight", w)
            for name, w in zip(("q_proj", "k_proj", "v_proj", "gate_proj"), weights)
        )
        assert loaded == {"layers.0.self_attn.qkv_gate_proj.weight"}
        assert param.data_ptr() == initial_pointer
        expected_weight = torch.cat([w.chunk(tp_size)[-1] for w in weights])
        torch.testing.assert_close(param, expected_weight, rtol=0, atol=0)
        x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(
            packed_projection(x, param, False),
            torch.nn.functional.linear(x, expected_weight),
            rtol=0,
            atol=0,
        )
