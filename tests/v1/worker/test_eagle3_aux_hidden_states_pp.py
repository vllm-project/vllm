# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed import parallel_state
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.model_executor.models.mimo import MiMoModel
from vllm.models.minimax_m3.nvidia import model as minimax_m3
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.pp_utils import PPHandler
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    reserve_aux_intermediate_tensor_slots,
    verify_supports_aux_hidden_states_over_pp,
)


def test_aux_layers_are_sorted_and_deduplicated():
    model = EagleModelMixin()
    model._set_aux_hidden_state_layers((48, 3, 90, 24, 48))
    assert model.aux_hidden_state_layers == (3, 24, 48, 90)


def test_mimo_does_not_inherit_aux_hidden_state_pp_support():
    inner = MiMoModel.__new__(MiMoModel)
    target = SimpleNamespace(model=inner)

    assert not inner.supports_aux_hidden_states_over_pp
    with pytest.raises(ValueError, match="does not support eagle3"):
        verify_supports_aux_hidden_states_over_pp(target, "eagle3")


class _PPAuxLayer(torch.nn.Module):
    ffn_all_reduce_deferred = False

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, positions, hidden_states, residual):
        total = hidden_states if residual is None else hidden_states + residual
        residual = hidden_states if residual is None else residual
        residual.copy_(total + self.index + 1)
        return torch.full_like(hidden_states, 10 * (self.index + 1)), residual


class _PPAuxNorm(torch.nn.Module):
    def forward(self, hidden_states, residual):
        output = hidden_states + residual
        hidden_states.fill_(-101)
        residual.fill_(-103)
        return output, residual


@pytest.fixture
def minimax_pp_stage(monkeypatch):
    """Use the real MiniMax constructor/forward and aux helpers on CPU tensors."""
    group = SimpleNamespace(world_size=2, is_first_rank=True, is_last_rank=False)
    bounds = [0, 3]
    monkeypatch.setattr(minimax_m3, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        minimax_m3,
        "VocabParallelEmbedding",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        minimax_m3, "MiniMAXGemmaRMSNorm", lambda *args, **kwargs: _PPAuxNorm()
    )
    monkeypatch.setattr(
        minimax_m3,
        "MiniMaxM3DecoderLayer",
        lambda **kwargs: _PPAuxLayer(int(kwargs["prefix"].rsplit(".", 1)[1])),
    )

    def make_layers(num_layers, build, prefix):
        return (
            *bounds,
            torch.nn.ModuleList(
                build(f"{prefix}.{index}")
                if bounds[0] <= index < bounds[1]
                else torch.nn.Identity()
                for index in range(num_layers)
            ),
        )

    monkeypatch.setattr(minimax_m3, "make_layers", make_layers)

    def make_stage(start, end, taps, world_size=2):
        bounds[:] = [start, end]
        group.world_size = world_size
        group.is_first_rank = start == 0
        group.is_last_rank = end == 6
        config = SimpleNamespace(
            vocab_size=16, hidden_size=6144, num_hidden_layers=6, rms_norm_eps=1e-6
        )
        model = minimax_m3.MiniMaxM3Model(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(hf_text_config=config), quant_config=None
            )
        )
        model._set_aux_hidden_state_layers(taps)
        target = SimpleNamespace(
            model=model,
            make_empty_intermediate_tensors=model.make_empty_intermediate_tensors,
        )
        reserve_aux_intermediate_tensor_slots(target)
        handler = PPHandler.__new__(PPHandler)
        handler.configure_aux_hidden_state_relay(target)
        return model, target, handler

    return make_stage


def _expected_aux_value(tap):
    return 7 + 11 * tap * (tap + 1) // 2


def _pp_aux_state(value):
    return torch.full((2, 6144), float(value), device="cpu")


def test_minimax_satisfies_existing_eagle3_pp_capability_guard(minimax_pp_stage):
    for start, end in ((0, 3), (3, 6)):
        _, target, _ = minimax_pp_stage(start, end, (0, 3, 6))
        verify_supports_aux_hidden_states_over_pp(target, "eagle3")


@pytest.mark.parametrize(
    "taps", [(0, 3, 6), (1, 4, 5), (2, 3, 4), (0, 1, 3), (4, 5, 6), (6, 0, 3, 3)]
)
def test_minimax_pp2_preserves_ordered_global_aux_taps(minimax_pp_stage, taps):
    first, _, relay = minimax_pp_stage(0, 3, taps)
    sent = first(None, None, None, _pp_aux_state(7))
    assert relay.relay_aux_hidden_states(None, sent) is sent
    last, _, _ = minimax_pp_stage(3, 6, taps)
    result = last(None, None, sent)
    assert isinstance(result, tuple)
    output, aux = result
    torch.testing.assert_close(output, _pp_aux_state(_expected_aux_value(6)))
    for hidden, tap in zip(aux, sorted(set(taps)), strict=True):
        torch.testing.assert_close(hidden, _pp_aux_state(_expected_aux_value(tap)))


def test_minimax_boundary_and_tap_zero_are_packed_upstream_once(minimax_pp_stage):
    first, _, _ = minimax_pp_stage(0, 3, (0, 3, 6))
    sent = first(None, None, None, _pp_aux_state(7))
    assert set(sent.tensors) == {
        "hidden_states",
        "residual",
        "aux_hidden_states_0",
        "aux_hidden_states_1",
    }
    torch.testing.assert_close(sent["aux_hidden_states_0"], _pp_aux_state(7))
    torch.testing.assert_close(
        sent["aux_hidden_states_1"], _pp_aux_state(_expected_aux_value(3))
    )


def test_minimax_existing_reservation_adds_static_upstream_slots(minimax_pp_stage):
    _, target, handler = minimax_pp_stage(3, 6, (6, 0, 3, 3))
    buffers = target.make_empty_intermediate_tensors(12, torch.bfloat16, "cpu")
    assert set(buffers.tensors) == {
        "hidden_states",
        "residual",
        "aux_hidden_states_0",
        "aux_hidden_states_1",
    }
    assert all(tensor.shape == (12, 6144) for tensor in buffers.tensors.values())
    assert all(
        tensor.dtype == torch.bfloat16 and tensor.device == torch.device("cpu")
        for tensor in buffers.tensors.values()
    )
    assert handler.aux_hidden_state_relay_keys == ()


def test_minimax_missing_upstream_aux_slot_is_not_silently_dropped(minimax_pp_stage):
    model, _, _ = minimax_pp_stage(3, 6, (0, 3, 6))
    incoming = IntermediateTensors(
        {
            "hidden_states": _pp_aux_state(30),
            "residual": _pp_aux_state(43),
            "aux_hidden_states_0": _pp_aux_state(7),
        }
    )
    with pytest.raises(RuntimeError, match="Missing aux_hidden_states_1"):
        model(None, None, incoming)


@pytest.mark.parametrize("taps", [(), (1, 3, 6)])
def test_minimax_pp1_output_and_existing_nonzero_taps_are_unchanged(
    minimax_pp_stage, taps
):
    model, target, _ = minimax_pp_stage(0, 6, taps, world_size=1)
    result = model(None, None, None, _pp_aux_state(7))
    assert set(
        target.make_empty_intermediate_tensors(12, torch.bfloat16, "cpu").tensors
    ) == {"hidden_states", "residual"}
    if taps:
        output, aux = result
        for hidden, tap in zip(aux, taps, strict=True):
            torch.testing.assert_close(hidden, _pp_aux_state(_expected_aux_value(tap)))
    else:
        output = result
    torch.testing.assert_close(output, _pp_aux_state(_expected_aux_value(6)))


def test_minimax_tap_zero_is_a_snapshot_before_in_place_residual_updates(
    minimax_pp_stage,
):
    model, _, _ = minimax_pp_stage(0, 6, (0, 3, 6), world_size=1)
    _, aux = model(None, None, None, _pp_aux_state(7))
    for hidden, tap in zip(aux, (0, 3, 6), strict=True):
        torch.testing.assert_close(hidden, _pp_aux_state(_expected_aux_value(tap)))


def test_minimax_pp2_without_aux_keeps_hidden_and_residual_transport(minimax_pp_stage):
    first, _, _ = minimax_pp_stage(0, 3, ())
    sent = first(None, None, None, _pp_aux_state(7))
    assert set(sent.tensors) == {"hidden_states", "residual"}
    torch.testing.assert_close(sent["hidden_states"], _pp_aux_state(30))
    torch.testing.assert_close(sent["residual"], _pp_aux_state(43))
    last, target, _ = minimax_pp_stage(3, 6, ())
    assert set(
        target.make_empty_intermediate_tensors(12, torch.bfloat16, "cpu").tensors
    ) == {"hidden_states", "residual"}
    output = last(None, None, sent)
    torch.testing.assert_close(output, _pp_aux_state(_expected_aux_value(6)))
