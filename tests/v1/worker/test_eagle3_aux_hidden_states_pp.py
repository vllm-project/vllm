# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed import parallel_state
from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.mimo import MiMoModel
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.models.minimax_m3.nvidia import model as minimax_m3
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    requires_aux_hidden_states,
    reserve_aux_intermediate_tensor_slots,
    verify_supports_aux_hidden_states_over_pp,
)


@pytest.mark.parametrize(
    ("method", "hf_config", "expected"),
    [
        ("eagle3", {"eagle_config": {"use_aux_hidden_state": False}}, False),
        ("eagle3", {"eagle_config": {"use_aux_hidden_state": True}}, True),
        ("eagle3", {"eagle_config": {}}, True),
        ("eagle3", {"eagle_config": None}, True),
        ("eagle3", {}, True),
        ("dflash", {"eagle_config": {"use_aux_hidden_state": False}}, True),
        ("dspark", {"eagle_config": {"use_aux_hidden_state": False}}, True),
        ("extract_hidden_states", {}, True),
        ("mtp", {}, False),
    ],
)
def test_aux_hidden_state_requirement_respects_eagle3_config(
    method, hf_config, expected
):
    spec_config = SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(**hf_config)),
    )
    assert requires_aux_hidden_states(spec_config) is expected


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
    """Aliases and mutates residual in place like MiniMaxM3DecoderLayer."""

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
    """Clobbers its inputs so aux taps that alias them would fail."""

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
                model_config=SimpleNamespace(hf_text_config=config),
                quant_config=None,
                speculative_config=None,
                use_v2_model_runner=True,
            )
        )
        model._set_aux_hidden_state_layers(taps)
        target = SimpleNamespace(
            model=model,
            make_empty_intermediate_tensors=model.make_empty_intermediate_tensors,
        )
        reserve_aux_intermediate_tensor_slots(target)
        return model, target

    return make_stage


# Input 7; _PPAuxLayer i adds 11 * (i + 1) to hidden + residual.
def _expected_aux_value(tap):
    return 7 + 11 * tap * (tap + 1) // 2


def _pp_aux_state(value):
    return torch.full((2, 6144), float(value), device="cpu")


@pytest.mark.parametrize("taps", [(0, 3, 6), (0, 1, 3), (4, 5, 6)])
def test_minimax_pp2_preserves_ordered_global_aux_taps(minimax_pp_stage, taps):
    first, _ = minimax_pp_stage(0, 3, taps)
    sent = first(None, None, None, _pp_aux_state(7))
    last, target = minimax_pp_stage(3, 6, taps)
    reserved = target.make_empty_intermediate_tensors(2, torch.float32, "cpu")
    assert set(sent.tensors) == set(reserved.tensors)
    result = last(None, None, sent)
    assert isinstance(result, tuple)
    output, aux = result
    torch.testing.assert_close(output, _pp_aux_state(_expected_aux_value(6)))
    for hidden, tap in zip(aux, taps, strict=True):
        torch.testing.assert_close(hidden, _pp_aux_state(_expected_aux_value(tap)))


def test_minimax_missing_upstream_aux_slot_is_not_silently_dropped(minimax_pp_stage):
    model, _ = minimax_pp_stage(3, 6, (0, 3, 6))
    incoming = IntermediateTensors(
        {
            "hidden_states": _pp_aux_state(30),
            "residual": _pp_aux_state(43),
            "aux_hidden_states_0": _pp_aux_state(7),
        }
    )
    with pytest.raises(RuntimeError, match="Missing aux_hidden_states_1"):
        model(None, None, incoming)


@pytest.mark.parametrize("taps", [(), (0, 3, 6)])
def test_minimax_pp1_output_and_aux_taps(minimax_pp_stage, taps):
    model, _ = minimax_pp_stage(0, 6, taps, world_size=1)
    output = model(None, None, None, _pp_aux_state(7))
    if taps:
        output, aux = output
        for hidden, tap in zip(aux, taps, strict=True):
            torch.testing.assert_close(hidden, _pp_aux_state(_expected_aux_value(tap)))
    torch.testing.assert_close(output, _pp_aux_state(_expected_aux_value(6)))


def test_minimax_pp2_without_aux_keeps_hidden_and_residual_transport(minimax_pp_stage):
    first, _ = minimax_pp_stage(0, 3, ())
    sent = first(None, None, None, _pp_aux_state(7))
    assert set(sent.tensors) == {"hidden_states", "residual"}
    torch.testing.assert_close(sent["hidden_states"], _pp_aux_state(30))
    torch.testing.assert_close(sent["residual"], _pp_aux_state(43))
    last, target = minimax_pp_stage(3, 6, ())
    assert set(
        target.make_empty_intermediate_tensors(12, torch.bfloat16, "cpu").tensors
    ) == {"hidden_states", "residual"}
    output = last(None, None, sent)
    torch.testing.assert_close(output, _pp_aux_state(_expected_aux_value(6)))


# Kimi-K3 / DSpark: 93 layers, target_layer_ids [2, 23, 47, 71, 89].
# get_eagle3_aux_layers_from_config maps target_layer_ids -> +1.
NUM_LAYERS = 93
AUX_IDS = (3, 24, 48, 72, 90)


def _local_aux_layer_ids(start_layer, end_layer, aux_ids, is_first_rank):
    return list(
        EagleModelMixin.local_aux_layer_ids(
            start_layer, end_layer, aux_ids, is_first_rank
        )
    )


def _simulate(num_layers, pp_size, aux_ids):
    gathered: list[int] = []
    for rank in range(pp_size):
        start, end = get_pp_indices(num_layers, rank, pp_size)
        gathered.extend(_local_aux_layer_ids(start, end, aux_ids, rank == 0))
    return gathered


@pytest.mark.parametrize("pp_size", [1, 2, 3, 4, 6, 8])
def test_drafter_sees_all_aux_layers_in_order(pp_size):
    assert _simulate(NUM_LAYERS, pp_size, AUX_IDS) == list(AUX_IDS)


def test_pp2_split_matches_expected_stages():
    start0, end0 = get_pp_indices(NUM_LAYERS, 0, 2)
    start1, end1 = get_pp_indices(NUM_LAYERS, 1, 2)
    assert _local_aux_layer_ids(start0, end0, AUX_IDS, True) == [3, 24]
    assert _local_aux_layer_ids(start1, end1, AUX_IDS, False) == [48, 72, 90]


@pytest.mark.parametrize("pp_size", [1, 2, 4])
def test_no_drafter_adds_no_payload_keys(pp_size):
    assert _simulate(NUM_LAYERS, pp_size, ()) == []


def test_aux_layer_on_stage_boundary_is_not_double_counted():
    _, end0 = get_pp_indices(NUM_LAYERS, 0, 2)
    aux_ids = tuple(sorted(AUX_IDS + (end0,)))
    assert _simulate(NUM_LAYERS, 2, aux_ids).count(end0) == 1


def test_middle_stage_produces_only_local_aux_states():
    start, end = get_pp_indices(NUM_LAYERS, 1, 4)
    local = _local_aux_layer_ids(start, end, AUX_IDS, False)
    upstream = [a for a in AUX_IDS if a <= start]
    assert not set(local) & set(upstream)
    assert local == [a for a in AUX_IDS if start < a <= end]


@pytest.mark.parametrize(
    "model_cls", [pytest.param(LlamaModel), pytest.param(Qwen2Model)]
)
def test_forward_does_not_name_update(model_cls):
    """Packing auxiliary states must not go through dict.update.

    TorchDynamoWrapper.bytecode_hook refuses any compiled forward whose
    bytecode names `update`, so a model that packs its states that way cannot
    start under cudagraphs. The same holds for the other opted-in models.
    """
    assert "update" not in model_cls.forward.__code__.co_names
