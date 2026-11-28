# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EAGLE-3 / DFlash / DSpark auxiliary hidden-state layer bounds."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from vllm.model_executor.models import nemotron_h as nemotron_h_mod
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsEagle3,
)
from vllm.model_executor.models.nano_nemotron_vl import NemotronH_Nano_VL_V2
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
    set_eagle3_aux_hidden_state_layers,
)


class _DummyDecoder(nn.Module, EagleModelMixin):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])


class _DummyCausalLM(nn.Module, SupportsEagle3):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.model = _DummyDecoder(num_layers)


class _DummyMultimodalLM(nn.Module, SupportsEagle3):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.language_model = _DummyCausalLM(num_layers)

    def get_language_model(self) -> nn.Module:
        return self.language_model


class _DummySizedLayerContainer(nn.Module, EagleModelMixin):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self._inner = [nn.Identity() for _ in range(num_layers)]

    def __len__(self) -> int:
        return len(self._inner)


class _DummyPlamoStyleCausalLM(nn.Module, SupportsEagle3):
    """Mimics Plamo3Model where `model.layers` is a Sized decoder module."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        decoder = _DummySizedLayerContainer(num_layers)
        model = nn.Module()
        model.layers = decoder
        self.model = model
        self._decoder = decoder

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self._decoder._set_aux_hidden_state_layers(layers)


class _DummyParseStyleModel(nn.Module, SupportsEagle3):
    """Mimics RadioMBartPrefixForCausalLM holding `self.decoder.layers`."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        decoder = _DummyDecoder(num_layers)
        self.decoder = decoder

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.decoder._set_aux_hidden_state_layers(layers)


class _DummyTransformersStyleModel(nn.Module, SupportsEagle3):
    """Mimics TransformersBase storing `_layer_names` instead of `layers`."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self._layer_names = {i: f"layers.{i}" for i in range(num_layers)}
        self.text_config = SimpleNamespace(num_hidden_layers=num_layers)
        self.aux_layers: tuple[int, ...] = ()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.aux_layers = tuple(layers)


class _DummyInklingLoopInner(nn.Module):
    def __init__(self, physical_layers: int, num_layer_visits: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(physical_layers)])
        self.config = SimpleNamespace(num_layer_visits=num_layer_visits)
        self.aux_hidden_state_layers: tuple[int, ...] = ()


class _DummyInklingLoopModel(nn.Module, SupportsEagle3):
    def __init__(self, physical_layers: int, num_layer_visits: int) -> None:
        super().__init__()
        self.model = _DummyInklingLoopInner(physical_layers, num_layer_visits)

    def get_language_model(self) -> nn.Module:
        return self

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = tuple(layers)


class _DummyNemotronBlock(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions
        out = hidden_states + 1.0
        return (out, out if residual is None else residual + out)


class _IdentityNorm(nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if residual is None:
            return hidden_states
        return hidden_states + residual, residual


def _spec_config(**hf_fields: object) -> SimpleNamespace:
    return SimpleNamespace(
        method="dflash",
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(**hf_fields)),
    )


@pytest.mark.parametrize(
    "model_factory,get_saved_layers",
    [
        (lambda: _DummyCausalLM(4), lambda m: m.model.aux_hidden_state_layers),
        (
            lambda: _DummyMultimodalLM(4),
            lambda m: m.language_model.model.aux_hidden_state_layers,
        ),
        (
            lambda: _DummyPlamoStyleCausalLM(4),
            lambda m: m._decoder.aux_hidden_state_layers,
        ),
        (
            lambda: _DummyParseStyleModel(4),
            lambda m: m.decoder.aux_hidden_state_layers,
        ),
        (lambda: _DummyTransformersStyleModel(4), lambda m: m.aux_layers),
    ],
)
@pytest.mark.parametrize("bad_target_layer_ids", [[4], [0, 4], [-1], [False]])
def test_out_of_range_dflash_target_layer_ids_rejected(
    model_factory, get_saved_layers, bad_target_layer_ids
):
    model = model_factory()
    spec_config = _spec_config(dflash_config={"target_layer_ids": bad_target_layer_ids})

    with pytest.raises(ValueError, match="dflash_config.target_layer_ids"):
        set_eagle3_aux_hidden_state_layers(model, spec_config)

    assert get_saved_layers(model) == ()


@pytest.mark.parametrize(
    "hf_fields,expected_aux_layers",
    [
        ({"dflash_config": {"target_layer_ids": [0, 3]}}, (1, 4)),
        ({"target_layer_ids": [0, 3]}, (1, 4)),
        ({"dspark_target_layer_ids": [0, 3]}, (1, 4)),
        (
            {"model_type": "deepseek_v41", "dspark_target_layer_ids": [0, 4]},
            (0, 4),
        ),
        ({"eagle_aux_hidden_state_layer_ids": [0, 2, 4]}, (0, 2, 4)),
        ({"dflash_config": {"layer_ids": [0, 4]}}, (0, 4)),
        ({"eagle_config": {"layer_ids": [1, 4]}}, (1, 4)),
    ],
)
def test_valid_aux_layer_sources_accepted(hf_fields, expected_aux_layers):
    model = _DummyCausalLM(4)
    spec_config = _spec_config(**hf_fields)

    assert get_eagle3_aux_layers_from_config(spec_config) == expected_aux_layers
    set_eagle3_aux_hidden_state_layers(model, spec_config)
    assert model.model.aux_hidden_state_layers == expected_aux_layers


@pytest.mark.parametrize(
    "hf_fields",
    [
        {"target_layer_ids": [4]},
        {"target_layer_ids": [-1]},
        {"dspark_target_layer_ids": [4]},
        {"dspark_target_layer_ids": [-1]},
        {"model_type": "deepseek_v41", "dspark_target_layer_ids": [5]},
        {"model_type": "deepseek_v41", "dspark_target_layer_ids": [-1]},
        {"eagle_aux_hidden_state_layer_ids": [5]},
        {"eagle_aux_hidden_state_layer_ids": [-1]},
        {"dflash_config": {"layer_ids": [5]}},
        {"eagle_config": {"layer_ids": [-1]}},
    ],
)
def test_out_of_range_aux_layer_sources_rejected(hf_fields):
    model = _DummyCausalLM(4)
    spec_config = _spec_config(**hf_fields)

    with pytest.raises(ValueError, match="out of range"):
        set_eagle3_aux_hidden_state_layers(model, spec_config)

    assert model.model.aux_hidden_state_layers == ()


def test_default_aux_layers_are_validated():
    tiny_model = _DummyCausalLM(2)
    spec_config = SimpleNamespace(method="eagle3", draft_model_config=None)

    with pytest.raises(ValueError, match="out of range"):
        set_eagle3_aux_hidden_state_layers(tiny_model, spec_config)

    ok_model = _DummyCausalLM(8)
    set_eagle3_aux_hidden_state_layers(ok_model, spec_config)
    assert ok_model.model.aux_hidden_state_layers == (2, 4, 5)


def test_inkling2_loop_uses_num_layer_visits_bound():
    loop_model = _DummyInklingLoopModel(physical_layers=4, num_layer_visits=8)

    valid_spec = _spec_config(eagle_aux_hidden_state_layer_ids=[0, 5, 8])
    set_eagle3_aux_hidden_state_layers(loop_model, valid_spec)
    assert loop_model.model.aux_hidden_state_layers == (0, 5, 8)

    invalid_spec = _spec_config(eagle_aux_hidden_state_layer_ids=[9])
    with pytest.raises(ValueError, match="out of range"):
        set_eagle3_aux_hidden_state_layers(loop_model, invalid_spec)


def test_nemotron_h_and_nano_vl_dflash2_aux_layer_validation(monkeypatch):
    monkeypatch.setattr(
        nemotron_h_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    inner = object.__new__(nemotron_h_mod.NemotronHModel)
    nn.Module.__init__(inner)
    inner.do_not_compile = True
    inner.start_layer = 0
    inner.end_layer = 4
    inner.layers = nn.ModuleList([_DummyNemotronBlock() for _ in range(4)])
    inner.norm_f = _IdentityNorm()

    causal_lm = object.__new__(nemotron_h_mod.NemotronHForCausalLM)
    nn.Module.__init__(causal_lm)
    causal_lm.model = inner

    nano_vl = object.__new__(cast(Any, NemotronH_Nano_VL_V2))
    nn.Module.__init__(nano_vl)
    nano_vl.language_model = causal_lm

    bad_spec = _spec_config(dflash_config={"target_layer_ids": [4]})
    with pytest.raises(
        ValueError,
        match=r"NemotronH_Nano_VL_V2 has 4 decoder layers.*\[0, 3\].*\[4\]",
    ):
        set_eagle3_aux_hidden_state_layers(nano_vl, bad_spec)
    assert inner.aux_hidden_state_layers == ()

    good_spec = _spec_config(dflash_config={"target_layer_ids": [0, 3]})
    set_eagle3_aux_hidden_state_layers(nano_vl, good_spec)
    assert inner.aux_hidden_state_layers == (1, 4)

    hidden_states, aux_hidden_states = inner(
        input_ids=None,
        positions=torch.tensor([0]),
        inputs_embeds=torch.zeros((1, 4)),
    )
    assert hidden_states.shape == (1, 4)
    assert len(aux_hidden_states) == 2


def test_unknown_layer_count_still_rejects_negative_indices():
    class _OpaqueEagle3Model(nn.Module, SupportsEagle3):
        def __init__(self) -> None:
            super().__init__()
            self.aux_layers: tuple[int, ...] = ()

        def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
            self.aux_layers = tuple(layers)

    model = _OpaqueEagle3Model()

    with pytest.raises(ValueError, match="non-negative"):
        set_eagle3_aux_hidden_state_layers(
            model, _spec_config(dflash_config={"target_layer_ids": [-1]})
        )
    with pytest.raises(ValueError, match="out of range"):
        set_eagle3_aux_hidden_state_layers(
            model, _spec_config(eagle_aux_hidden_state_layer_ids=[-1])
        )

    set_eagle3_aux_hidden_state_layers(
        model, _spec_config(dflash_config={"target_layer_ids": [0, 1]})
    )
    assert model.aux_layers == (1, 2)


def test_zero_layer_model_rejects_target_layer_ids():
    model = _DummyCausalLM(0)
    with pytest.raises(ValueError, match="0 layers"):
        set_eagle3_aux_hidden_state_layers(
            model, _spec_config(dflash_config={"target_layer_ids": [0]})
        )
