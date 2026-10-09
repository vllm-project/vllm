# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models import llama
from vllm.v1.worker.tpsp_profile import SPProfile, TPSPProjection, TPSPShape


def make_projections(enabled):
    projections = nn.ModuleDict(
        {
            name: TPSPProjection(TPSPShape(2, 2, 1e-5, True), 2, "test")
            for name in ("o", "down")
        }
    )
    for name, is_enabled in zip(projections, enabled, strict=True):
        projections[name].profile = SPProfile(
            2, 2, 8, "enabled" if is_enabled else "disabled", ""
        )
    return projections


@pytest.mark.parametrize("unsupported", ("pp", "lora", "quant"))
def test_tpsp_incompatible_config_preserves_regular_model(unsupported):
    model = llama.LlamaForCausalLM.__new__(llama.LlamaForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.tpsp_projections = None
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2 if unsupported == "pp" else 1
        ),
        lora_config=object() if unsupported == "lora" else None,
        quant_config=object() if unsupported == "quant" else None,
    )
    model._init_tpsp(config)
    assert model.model.tpsp_projections is None


def test_tpsp_capable_llama_registers_profile_on_existing_model(monkeypatch):
    group = SimpleNamespace(world_size=2, device_group=SimpleNamespace(group_name="tp"))
    monkeypatch.setattr(llama, "get_tp_group", lambda: group)
    model = llama.LlamaForCausalLM.__new__(llama.LlamaForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=4)
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=SimpleNamespace(input_size_per_partition=2)),
        mlp=SimpleNamespace(down_proj=SimpleNamespace(input_size_per_partition=3)),
        post_attention_layernorm=SimpleNamespace(variance_epsilon=1e-5),
        input_layernorm=SimpleNamespace(variance_epsilon=1e-5),
    )
    model.model = llama.LlamaModel.__new__(llama.LlamaModel)
    nn.Module.__init__(model.model)
    model.model.layers = [layer]
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        quant_config=None,
        lora_config=None,
        device_config=SimpleNamespace(device_type="test"),
    )
    model._init_tpsp(config)
    assert model.model.tpsp_projections["o"].shape.input_width == 2
    assert model.model.tpsp_projections["down"].shape.input_width == 3
    assert not model.model.tpsp_active


def test_tpsp_attention_reuses_qkv_and_attention():
    attention = llama.LlamaAttention.__new__(llama.LlamaAttention)
    nn.Module.__init__(attention)
    attention.q_size = attention.kv_size = 2
    calls = []

    def qkv_proj(x):
        calls.append("qkv")
        return torch.cat((x, x, x), dim=-1), None

    def attend(q, k, v):
        calls.append("attention")
        return q + k + v

    def o_proj(x):
        calls.append("o_proj")
        return x * 2, None

    attention.qkv_proj = qkv_proj
    attention.rotary_emb = lambda positions, q, k: (q, k)
    attention.attn = attend
    attention.o_proj = o_proj
    inputs = torch.ones(2, 2)

    torch.testing.assert_close(attention(None, inputs), 6 * inputs)
    assert calls == ["qkv", "attention", "o_proj"]
    calls.clear()
    torch.testing.assert_close(attention.compute_attention(None, inputs), 3 * inputs)
    assert calls == ["qkv", "attention"]


def test_tpsp_mlp_reuses_gate_up_and_activation(monkeypatch):
    mlp = llama.LlamaMLP.__new__(llama.LlamaMLP)
    nn.Module.__init__(mlp)
    calls = []

    def gate_up_proj(x):
        calls.append("gate_up")
        return x + 1, None

    def down_proj(x):
        calls.append("down")
        return x + 2, None

    mlp.gate_up_proj = gate_up_proj
    mlp.act_fn = lambda x: x * 3
    mlp.down_proj = down_proj
    monkeypatch.setattr(
        llama, "maybe_fused_act_quant", lambda act_fn, x, down: act_fn(x)
    )
    inputs = torch.ones(2, 2)

    torch.testing.assert_close(mlp(inputs), torch.full((2, 2), 8.0))
    assert calls == ["gate_up", "down"]
    calls.clear()
    torch.testing.assert_close(
        mlp.compute_down_proj_input(inputs), torch.full((2, 2), 6.0)
    )
    assert calls == ["gate_up"]


def test_tpsp_projection_does_not_activate_layer_by_itself():
    layer = llama.LlamaDecoderLayer.__new__(llama.LlamaDecoderLayer)
    nn.Module.__init__(layer)
    layer.input_layernorm = nn.Identity()
    layer.self_attn = lambda positions, hidden_states: hidden_states + 2
    layer.post_attention_layernorm = lambda hidden_states, residual: (
        hidden_states + 1,
        residual,
    )
    layer.mlp = lambda hidden_states: hidden_states + 3
    inputs = torch.ones(2, 2)

    output, residual = layer(None, inputs, None, tpsp_o=object())
    torch.testing.assert_close(output, torch.full((2, 2), 7.0))
    torch.testing.assert_close(residual, inputs)


def test_tpsp_biased_projection_uses_supported_fused_path(monkeypatch):
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(llama, "get_tp_group", lambda: group)
    monkeypatch.setattr(llama, "select_sp_config", lambda profile, tokens: True)
    layer = llama.LlamaDecoderLayer.__new__(llama.LlamaDecoderLayer)
    nn.Module.__init__(layer)
    layer.hidden_size = 2

    class Projection(nn.Module):
        input_size_per_partition = 2

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.eye(2, dtype=torch.bfloat16))
            self.bias = nn.Parameter(torch.tensor([2.0, 3.0], dtype=torch.bfloat16))

        def forward(self, x):
            return x + self.bias, None

    class Norm:
        weight = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        variance_epsilon = 1e-5

        def __call__(self, x, residual):
            return x + residual, x + residual

    class Backend:
        supports_projection_bias = True

        def fused(
            self, x, weight, norm_weight, residual, eps, config, *, projection_bias
        ):
            torch.testing.assert_close(projection_bias, projection.bias)
            output = x + projection_bias + residual
            return output, None, output

    x = torch.ones((2, 2), dtype=torch.bfloat16)
    projection = Projection()
    profile = SimpleNamespace(tp_size=1, hidden_size=2, input_width=2, config=object())
    output, residual = layer._project_and_normalize(
        x,
        projection,
        x,
        Norm(),
        SimpleNamespace(profile=profile, backend=Backend()),
        False,
    )
    expected = x + torch.tensor([2.0, 3.0], dtype=x.dtype) + x
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(residual, expected)


def test_disabled_down_projection_reconstructs_sharded_residual(monkeypatch):
    group = SimpleNamespace(world_size=2, rank_in_group=0, device_group=object())
    monkeypatch.setattr(llama, "get_tp_group", lambda: group)

    def gather(output, local, *, group):
        output[:2] = local
        output[2:] = local + 10

    monkeypatch.setattr(llama.dist, "all_gather_into_tensor", gather)
    layer = llama.LlamaDecoderLayer.__new__(llama.LlamaDecoderLayer)
    nn.Module.__init__(layer)
    layer.hidden_size = 2

    class Projection(nn.Module):
        input_size_per_partition = 2
        bias = None

        def forward(self, x):
            return torch.full_like(x, 2), None

    class Norm:
        def __call__(self, x, residual):
            return x + residual, x + residual

    class Backend:
        def fused(self, *args, **kwargs):
            raise AssertionError("disabled projection must not use the fused op")

    profile = SimpleNamespace(
        tp_size=2,
        hidden_size=2,
        input_width=2,
        max_batched_tokens=3,
        enabled=False,
        threshold_tokens=None,
    )
    output, residual = layer._project_and_normalize(
        torch.ones(3, 2),
        Projection(),
        torch.ones(2, 2),
        Norm(),
        SimpleNamespace(profile=profile, backend=Backend()),
        True,
    )
    torch.testing.assert_close(output[:, 0], torch.tensor([3.0, 3.0, 13.0]))
    torch.testing.assert_close(residual, torch.full((2, 2), 3.0))


class _TPSPLayer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.input_layernorm = nn.Identity()

    def forward(
        self,
        positions,
        hidden_states,
        residual,
        *,
        tpsp_active,
        tpsp_o,
        tpsp_down,
        next_norm,
    ):
        assert tpsp_active
        assert tpsp_o.active or tpsp_down.active
        local_residual = torch.full(
            (2, hidden_states.size(-1)),
            float(self.index + 1),
            dtype=hidden_states.dtype,
        )
        return hidden_states + 1, local_residual


@pytest.mark.parametrize("taps", [(), (0, 1, 2), (2,)])
@pytest.mark.parametrize("enabled", [(True, True), (True, False), (False, True)])
def test_tpsp_aux_states_gather_only_requested_layers(monkeypatch, taps, enabled):
    group = SimpleNamespace(world_size=2, device_group=object())
    monkeypatch.setattr(
        llama,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1, is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(llama, "get_tp_group", lambda: group)
    gathered = []

    def all_gather(output, local, *, group):
        gathered.append(local.clone())
        output[: local.size(0)] = local
        output[local.size(0) :] = local + 10

    monkeypatch.setattr(llama.dist, "all_gather_into_tensor", all_gather)
    model = llama.LlamaModel.__new__(llama.LlamaModel)
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 2
    model._aux_upstream_total_cached = 0
    model.config = SimpleNamespace(hidden_size=2)
    model.embed_tokens = nn.Identity()
    model.layers = nn.ModuleList([_TPSPLayer(0), _TPSPLayer(1)])
    model.norm = nn.Identity()
    model.tpsp_projections = make_projections(enabled)
    model.aux_hidden_state_layers = taps

    inputs = torch.zeros((3, 2))
    result = model.forward(None, None, None, inputs_embeds=inputs)
    if not taps:
        torch.testing.assert_close(result, inputs + 2)
    else:
        output, aux = result
        torch.testing.assert_close(output, inputs + 2)
        expected = {
            0: inputs,
            1: torch.tensor([[1.0, 1.0], [1.0, 1.0], [11.0, 11.0]]),
            2: torch.tensor([[2.0, 2.0], [2.0, 2.0], [12.0, 12.0]]),
        }
        for state, tap in zip(aux, taps, strict=True):
            torch.testing.assert_close(state, expected[tap])
    assert len(gathered) == sum(tap > 0 for tap in taps)


def test_tpsp_disabled_profiles_use_regular_llama_forward(monkeypatch):
    monkeypatch.setattr(
        llama,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    class RegularLayer(nn.Module):
        def forward(self, positions, hidden_states, residual):
            return hidden_states + 1, residual

    class RegularNorm(nn.Module):
        def forward(self, hidden_states, residual):
            return hidden_states, residual

    model = llama.LlamaModel.__new__(llama.LlamaModel)
    nn.Module.__init__(model)
    model.embed_tokens = nn.Identity()
    model.layers = nn.ModuleList([RegularLayer()])
    model.norm = RegularNorm()
    model.start_layer, model.end_layer = 0, 1
    model.aux_hidden_state_layers = ()
    model._aux_upstream_total_cached = 0
    model.tpsp_projections = make_projections((False, False))
    assert not model.tpsp_active
    result = model.forward(None, None, None, inputs_embeds=torch.zeros(2, 2))
    torch.testing.assert_close(result, torch.ones(2, 2))


def test_tpsp_requested_forward_requires_startup_profile():
    model = llama.LlamaModel.__new__(llama.LlamaModel)
    nn.Module.__init__(model)
    model.tpsp_projections = nn.ModuleDict(
        {"o": TPSPProjection(TPSPShape(2, 2, 1e-5, True), 2, "test")}
    )
    with pytest.raises(RuntimeError, match="requires worker startup profiling"):
        model.forward(None, None, None)
