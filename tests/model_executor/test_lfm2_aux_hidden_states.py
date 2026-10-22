# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture contract for LFM2 DSpark target features.

The backbone must return the full pre-normalization stream
(hidden_states + residual) at the indices
``get_eagle3_aux_layers_from_config`` already computed. A second offset, or
a capture of ``hidden_states`` alone, leaves shapes valid and drops the
trained features.
"""

from math import prod
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.model_executor.models import lfm2 as lfm2_mod
from vllm.model_executor.models import lfm2_vl as lfm2_vl_mod
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import get_target_lm_head


class _SplitLayer(nn.Module):
    """Stand-in for an attention or ShortConv block.

    Real LFM2 layers return the feed-forward output and the residual
    separately. The full stream at the next layer's entry is their sum.
    """

    def __init__(self, hidden_delta: float, residual_delta: float) -> None:
        super().__init__()
        self.hidden_delta = hidden_delta
        self.residual_delta = residual_delta

    def forward(self, positions, hidden_states, residual):
        del positions
        full = hidden_states if residual is None else hidden_states + residual
        return full + self.hidden_delta, full + self.residual_delta


class _AddNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual, None


def _reconstruct(inputs: torch.Tensor, specs: list[tuple[float, float]]):
    """Replay the stub layers without calling LFM2."""
    hidden: torch.Tensor | None = inputs
    residual = None
    boundaries: dict[int, torch.Tensor] = {}
    assert hidden is not None
    for idx, (hidden_delta, residual_delta) in enumerate(specs):
        full = hidden if residual is None else hidden + residual
        hidden = full + hidden_delta
        residual = full + residual_delta
        boundaries[idx + 1] = hidden + residual
    assert residual is not None
    return boundaries, hidden + residual


def _make_target(specs: list[tuple[float, float]]):
    backbone = lfm2_mod.Lfm2Model.__new__(lfm2_mod.Lfm2Model)
    nn.Module.__init__(backbone)
    backbone.start_layer = 0
    backbone.end_layer = len(specs)
    backbone.layers = nn.ModuleList(
        [
            _SplitLayer(hidden_delta, residual_delta)
            for hidden_delta, residual_delta in specs
        ]
    )
    backbone.embedding_norm = _AddNorm()
    backbone.do_not_compile = True

    target = lfm2_mod.Lfm2ForCausalLM.__new__(lfm2_mod.Lfm2ForCausalLM)
    nn.Module.__init__(target)
    target.model = backbone
    return target, backbone


# Layer 0 is attention-like, layer 1 is ShortConv-like, layer 2 is attention.
_LAYER_SPECS = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]


@pytest.fixture
def single_rank(monkeypatch):
    monkeypatch.setattr(
        lfm2_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )


@pytest.mark.cpu_test
def test_lfm2_captures_full_streams_at_helper_indices(single_rank):
    target, backbone = _make_target(_LAYER_SPECS)
    spec_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"target_layer_ids": [0, 1]})
        )
    )

    assert supports_eagle3(target)
    assert not backbone.supports_aux_hidden_states_over_pp
    set_eagle3_aux_hidden_state_layers(target, spec_config)
    # The helper adds the only +1. Published ids 0 and 1 become entries 1 and 2.
    assert backbone.aux_hidden_state_layers == (1, 2)

    inputs = torch.tensor([[1.0, 2.0]])
    output, aux_hidden_states = target(
        input_ids=None,
        positions=torch.tensor([0]),
        inputs_embeds=inputs,
    )
    boundaries, final = _reconstruct(inputs, _LAYER_SPECS)

    torch.testing.assert_close(output, final)
    assert len(aux_hidden_states) == 2
    # Boundary 1 follows the attention block. Boundary 2 follows ShortConv.
    torch.testing.assert_close(aux_hidden_states[0], boundaries[1])
    torch.testing.assert_close(aux_hidden_states[1], boundaries[2])


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_spec", [0, 9])
@pytest.mark.parametrize(
    "model_class",
    [lfm2_mod.Lfm2ForCausalLM, lfm2_vl_mod.Lfm2VLForConditionalGeneration],
)
def test_lfm2_cache_planning_includes_speculative_conv_slots(
    monkeypatch, model_class, num_spec
):
    """Planned state must fit the runtime convolution's speculative history."""
    language_config = SimpleNamespace(conv_dim=2048, hidden_size=2048, conv_L_cache=3)
    hf_config = (
        SimpleNamespace(text_config=language_config)
        if model_class is lfm2_vl_mod.Lfm2VLForConditionalGeneration
        else language_config
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        num_speculative_tokens=num_spec,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.short_conv."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    conv = ShortConv.__new__(ShortConv)
    nn.Module.__init__(conv)
    conv.conv_dim = 2048
    conv.L_cache = 3
    conv.num_spec = num_spec

    planned = model_class.get_mamba_state_shape_from_config(config)
    assert planned == conv.get_state_shape()
    assert prod(planned[0]) == 2048 * (2 + num_spec)


@pytest.mark.cpu_test
def test_lfm2_without_capture_returns_final_hidden_state(single_rank):
    target, _backbone = _make_target(_LAYER_SPECS)
    inputs = torch.tensor([[1.0, 2.0]])
    output = target(
        input_ids=None,
        positions=torch.tensor([0]),
        inputs_embeds=inputs,
    )
    _boundaries, final = _reconstruct(inputs, _LAYER_SPECS)

    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output, final)


@pytest.mark.cpu_test
def test_lfm2_vl_forwards_capture_and_shared_weights(single_rank):
    target, backbone = _make_target(_LAYER_SPECS)
    embed_tokens = nn.Embedding(8, 2)
    lm_head = nn.Linear(2, 8, bias=False)
    backbone.embed_tokens = embed_tokens
    target.lm_head = lm_head

    wrapper = lfm2_vl_mod.Lfm2VLForConditionalGeneration.__new__(
        lfm2_vl_mod.Lfm2VLForConditionalGeneration
    )
    nn.Module.__init__(wrapper)
    wrapper.language_model = target
    wrapper._language_model_names = ["language_model"]

    spec_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"target_layer_ids": [0, 1]})
        )
    )
    assert supports_eagle3(wrapper)
    set_eagle3_aux_hidden_state_layers(wrapper, spec_config)
    assert backbone.aux_hidden_state_layers == (1, 2)

    language_model = wrapper.get_language_model()
    assert language_model is target
    assert language_model.model.embed_tokens is embed_tokens
    assert get_target_lm_head(wrapper, language_model) is lm_head

    inputs = torch.tensor([[1.0, 2.0]])
    output, aux_hidden_states = wrapper(
        input_ids=None,
        positions=torch.tensor([0]),
        inputs_embeds=inputs,
    )
    boundaries, final = _reconstruct(inputs, _LAYER_SPECS)
    torch.testing.assert_close(output, final)
    torch.testing.assert_close(aux_hidden_states[0], boundaries[1])
    torch.testing.assert_close(aux_hidden_states[1], boundaries[2])
