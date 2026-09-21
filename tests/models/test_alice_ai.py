# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.models import alice_ai as alice_ai_module
from vllm.model_executor.models import qwen3_next as qwen3_next_module
from vllm.model_executor.models.alice_ai import (
    AliceAIDecoderBlock,
    AliceAIForCausalLM,
    AliceAIModel,
    _AttentionResidualFinalMixer,
    _load_split_kda_conv_weights,
)
from vllm.model_executor.offloader import NoopOffloader, PrefetchOffloader, UVAOffloader


def test_lora_configuration_reaches_parent_model_initialization(monkeypatch):
    monkeypatch.setattr(alice_ai_module, "get_offloader", lambda: NoopOffloader())
    parent_init = Mock(return_value=None)
    monkeypatch.setattr(qwen3_next_module.Qwen3NextForCausalLM, "__init__", parent_init)
    config = SimpleNamespace(lora_config=object())

    AliceAIForCausalLM(vllm_config=config, prefix="target")

    parent_init.assert_called_once_with(
        vllm_config=config, prefix="target", model_cls=AliceAIModel
    )


@pytest.mark.parametrize(
    "backend,uva_available,offload_bytes,error",
    [
        ("none", False, 0, None),
        ("uva", True, 1024, None),
        ("uva", False, 0, None),
        ("uva", False, 1024, "requires UVA"),
        ("prefetch", True, 1024, "does not support prefetch"),
    ],
)
def test_offload_modes_are_checked_before_model_initialization(
    monkeypatch, backend, uva_available, offload_bytes, error
):
    from vllm.model_executor.offloader import uva

    monkeypatch.setattr(uva, "is_uva_available", lambda: uva_available)
    monkeypatch.setenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", "0")
    if backend == "uva":
        offloader = UVAOffloader(cpu_offload_max_bytes=offload_bytes)
    elif backend == "prefetch":
        offloader = PrefetchOffloader.__new__(PrefetchOffloader)
    else:
        offloader = NoopOffloader()
    monkeypatch.setattr(alice_ai_module, "get_offloader", lambda: offloader)
    parent_init = Mock(return_value=None)
    monkeypatch.setattr(qwen3_next_module.Qwen3NextForCausalLM, "__init__", parent_init)
    config = SimpleNamespace(lora_config=None)

    if error is not None:
        with pytest.raises(NotImplementedError, match=error):
            AliceAIForCausalLM(vllm_config=config)
        parent_init.assert_not_called()
    else:
        AliceAIForCausalLM(vllm_config=config, prefix="target")
        parent_init.assert_called_once_with(
            vllm_config=config, prefix="target", model_cls=AliceAIModel
        )


@pytest.mark.parametrize(
    "source,target,shard_id",
    [
        ("linear_attn.q_proj.weight", "linear_attn.in_proj_qkvgfab.weight", 0),
        ("linear_attn.k_proj.weight", "linear_attn.in_proj_qkvgfab.weight", 1),
        ("linear_attn.v_proj.weight", "linear_attn.in_proj_qkvgfab.weight", 2),
        ("linear_attn.b_proj.weight", "linear_attn.in_proj_qkvgfab.weight", 3),
        ("linear_attn.f_a_proj.weight", "linear_attn.in_proj_qkvgfab.weight", 4),
        ("self_attn.q_proj.weight", "self_attn.qkv_proj.weight", "q"),
        ("linear_attn.a_log_bias", "linear_attn.A_log", None),
    ],
)
def test_weight_mapping_preserves_shard_ids(source, target, shard_id):
    tensor = torch.empty(1)
    [(mapped, mapped_tensor)] = list(
        AliceAIModel.hf_to_vllm_mapper.apply([(f"model.layers.0.{source}", tensor)])
    )
    assert mapped == f"model.layers.0.{target}"
    assert getattr(mapped_tensor, "shard_id", None) == shard_id


def test_split_kda_conv_weights_use_fused_loader_shard_ids() -> None:
    calls: list[tuple[torch.Tensor, int]] = []
    param = torch.nn.Parameter(torch.empty(3, 1, 1))

    def weight_loader(
        target: torch.Tensor,
        weight: torch.Tensor,
        shard_id: int,
    ) -> None:
        assert target is param
        calls.append((weight, shard_id))

    param.weight_loader = weight_loader
    target_name = "model.layers.0.linear_attn.conv1d.weight"
    weights = [
        ("model.layers.0.linear_attn.q_conv1d.weight", torch.full((1, 1), 1.0)),
        ("model.layers.0.linear_attn.k_conv1d.weight", torch.full((1, 1), 2.0)),
        ("model.layers.0.linear_attn.v_conv1d.weight", torch.full((1, 1), 3.0)),
        ("model.embed_tokens.weight", torch.full((1, 1), 4.0)),
    ]
    loaded: set[str] = set()

    remaining = list(
        _load_split_kda_conv_weights(weights, {target_name: param}, loaded)
    )

    assert loaded == {target_name}
    assert [shard_id for _, shard_id in calls] == [0, 1, 2]
    assert [weight.item() for weight, _ in calls] == [1.0, 2.0, 3.0]
    assert remaining == [weights[-1]]


def test_kda_cache_contract_is_fp32_recurrent() -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                linear_num_key_heads=32,
                linear_key_head_dim=128,
                linear_conv_kernel_dim=4,
            ),
        ),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        speculative_config=None,
    )

    assert AliceAIForCausalLM.get_mamba_state_dtype_from_config(vllm_config) == (
        torch.bfloat16,
        torch.float32,
    )
    conv_shape = (12288, 3) if is_conv_state_dim_first() else (3, 12288)
    assert AliceAIForCausalLM.get_mamba_state_shape_from_config(vllm_config) == (
        conv_shape,
        (32, 128, 128),
    )


class _FixedOutput(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.inputs: list[torch.Tensor] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.inputs.append(hidden_states.detach().clone())
        return torch.full_like(hidden_states, self.value)


class _FixedLinearAttention(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.inputs: list[torch.Tensor] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del positions
        self.inputs.append(hidden_states.detach().clone())
        output.fill_(self.value)


def _make_state_layer(
    layer_idx: int,
    block_size: int,
    attention_value: float,
    mlp_value: float,
) -> AliceAIDecoderBlock:
    layer = AliceAIDecoderBlock.__new__(AliceAIDecoderBlock)
    torch.nn.Module.__init__(layer)
    layer.layer_idx = layer_idx
    layer.layer_type = "linear_attention"
    layer.layer_scale = False
    layer.attn_res_block_size = block_size
    layer.attn_res_rms_norm_eps = 1e-6
    layer.input_layernorm = torch.nn.Identity()
    layer.post_attention_layernorm = torch.nn.Identity()
    layer.linear_attn = _FixedLinearAttention(attention_value)
    layer.mlp = _FixedOutput(mlp_value)
    if layer_idx > 0:
        layer.attn_res_proj = torch.nn.Linear(1, 1, bias=False)
        layer.attn_res_norm_weight = torch.nn.Parameter(torch.ones(1))
    layer.mlp_res_proj = torch.nn.Linear(1, 1, bias=False)
    layer.mlp_res_norm_weight = torch.nn.Parameter(torch.ones(1))
    return layer


def _make_final_mixer() -> _AttentionResidualFinalMixer:
    final = _AttentionResidualFinalMixer.__new__(_AttentionResidualFinalMixer)
    torch.nn.Module.__init__(final)
    final.rms_norm_eps = 1e-6
    final.res_proj = torch.nn.Linear(1, 1, bias=False)
    final.res_norm_weight = torch.nn.Parameter(torch.ones(1))
    return final


def test_attn_res_block_grouping_and_pre_norm_mtp_hidden(
    monkeypatch,
) -> None:
    calls: list[list[torch.Tensor]] = []

    def reference_attn_res(
        prefix,
        residual_bank,
        norm_weight,
        query_weight,
        *,
        num_blocks,
        eps,
    ):
        del norm_weight, query_weight, eps
        sources = [residual_bank[:, index] for index in range(num_blocks)] + [prefix]
        calls.append([source.detach().clone() for source in sources])
        return torch.stack(sources).sum(dim=0)

    monkeypatch.setattr(alice_ai_module, "_mix_attn_res", reference_attn_res)
    monkeypatch.setattr(
        alice_ai_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    raw_outputs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
    model = AliceAIModel.__new__(AliceAIModel)
    torch.nn.Module.__init__(model)
    model.attn_res_block_size = 4
    model.start_layer = 0
    model.end_layer = len(raw_outputs)
    model.layers = torch.nn.ModuleList(
        [
            _make_state_layer(index, 4, attention, mlp)
            for index, (attention, mlp) in enumerate(raw_outputs)
        ]
    )
    model.config = SimpleNamespace(num_hidden_layers=len(raw_outputs))
    model.attnres_final = _make_final_mixer()
    model._mtp_hidden_buffer = torch.empty(2, 1)
    model.norm = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.norm.weight.fill_(2)

    output = AliceAIModel.forward(
        model,
        input_ids=None,
        positions=torch.arange(2),
        inputs_embeds=torch.full((2, 1), 10.0),
    )

    torch.testing.assert_close(model._mtp_hidden_buffer, torch.full((2, 1), 65.0))
    torch.testing.assert_close(output, torch.full((2, 1), 130.0))
    torch.testing.assert_close(
        model.layers[0].linear_attn.inputs[0], torch.full((2, 1), 10.0)
    )
    assert [len(sources) for sources in calls] == [2] * 8 + [3, 3]
    assert [source[0, 0].item() for source in calls[-1]] == [10, 36, 19]
