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
    """Aliases and mutates residual in place like MiniMaxM3DecoderLayer.

    With ``deferred``, the FFN output is this rank's half of a 2-way TP sum, and
    the next input norm completes the all-reduce.
    """

    def __init__(self, index, deferred):
        super().__init__()
        self.index = index
        self.ffn_all_reduce_deferred = deferred
        self.fuse_input_allreduce = False

    def forward_input_norm(self, hidden_states, residual):
        if residual is None:
            return hidden_states, hidden_states
        if self.fuse_input_allreduce:
            hidden_states = hidden_states * 2
        residual.add_(hidden_states)
        return hidden_states, residual

    def forward_after_input_norm(self, positions, hidden_states, residual):
        residual.add_(self.index + 1)
        output = torch.full_like(hidden_states, 10 * (self.index + 1))
        return output / 2 if self.ffn_all_reduce_deferred else output, residual


class _PPAuxNorm(torch.nn.Module):
    """Fused add + norm (identity scaling) that overwrites its inputs in place."""

    def forward(self, hidden_states, residual):
        residual.add_(hidden_states)
        hidden_states.copy_(residual)
        return hidden_states, residual


@pytest.fixture
def minimax_pp_stage(monkeypatch):
    """Use the real MiniMax constructor/forward and aux helpers on CPU tensors."""
    group = SimpleNamespace(world_size=2, is_first_rank=True, is_last_rank=False)
    bounds = [0, 3]
    monkeypatch.setattr(minimax_m3, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(minimax_m3, "supports_fused_moe_finalize", lambda _: False)
    monkeypatch.setattr(
        minimax_m3,
        "VocabParallelEmbedding",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        minimax_m3, "MiniMAXGemmaRMSNorm", lambda *args, **kwargs: _PPAuxNorm()
    )
    deferred = [False]
    monkeypatch.setattr(
        minimax_m3,
        "MiniMaxM3DecoderLayer",
        lambda **kwargs: _PPAuxLayer(
            int(kwargs["prefix"].rsplit(".", 1)[1]), deferred[0]
        ),
    )
    monkeypatch.setattr(
        minimax_m3,
        "fused_allreduce_gemma_rms_norm",
        lambda hidden_states, residual, norm: norm(hidden_states * 2, residual),
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

    def make_stage(start, end, taps, world_size=2, deferred_all_reduce=False):
        bounds[:] = [start, end]
        deferred[0] = deferred_all_reduce
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
                kernel_config=SimpleNamespace(moe_backend="auto"),
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


@pytest.mark.parametrize("taps", [(0, 3, 6), (1, 2, 5)])
def test_minimax_aux_taps_follow_deferred_tp_all_reduce(minimax_pp_stage, taps):
    """Under TP each FFN output stays a partial sum until the next input norm or
    the final norm all-reduces it, so aux taps must read the reduced stream."""
    model, _ = minimax_pp_stage(0, 6, taps, world_size=1, deferred_all_reduce=True)
    assert model.fuse_final_norm_allreduce
    output, aux = model(None, None, None, _pp_aux_state(7))
    for hidden, tap in zip(aux, taps, strict=True):
        torch.testing.assert_close(hidden, _pp_aux_state(_expected_aux_value(tap)))
    torch.testing.assert_close(output, _pp_aux_state(_expected_aux_value(6)))
