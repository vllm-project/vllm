# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.models.kimi_k3.amd import linear as amd_linear
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia.model import (
    KimiK3ForConditionalGeneration,
    KimiLinearForCausalLM,
    KimiLinearModel,
)


def _make_kimi_linear_model(cls: type = KimiLinearModel) -> Any:
    model: Any = object.__new__(cls)
    object.__setattr__(model, "aux_hidden_state_layers", (2,))
    object.__setattr__(model, "use_sequence_parallel", False)
    object.__setattr__(model, "use_attn_res", False)
    return model


def test_kimi_k3_advertises_eagle3_support():
    assert supports_eagle3(KimiK3ForConditionalGeneration)


def test_kimi_linear_advertises_eagle3_support():
    # The text-only architecture serves the same inner KimiLinearModel, which
    # already carries the EagleModelMixin tap machinery - only the interface
    # declaration was missing, so EAGLE3-family speculative decoding (dspark)
    # was rejected at startup with "Model does not support EAGLE3 interface".
    assert supports_eagle3(KimiLinearForCausalLM)


def test_kimi_k3_uses_shared_eagle3_layer_configuration():
    target = object.__new__(KimiK3ForConditionalGeneration)
    torch.nn.Module.__init__(target)
    model = _make_kimi_linear_model()
    object.__setattr__(model, "layers", [None] * 93)
    language_model = SimpleNamespace(
        embed_input_ids=lambda _: None,
        forward=lambda input_ids, positions: None,
        model=model,
    )
    object.__setattr__(target, "language_model", language_model)
    object.__setattr__(target, "_language_model_names", ["language_model"])

    target.set_aux_hidden_state_layers((2, 46, 90))

    assert model.aux_hidden_state_layers == (2, 46, 90)
    assert target.get_eagle3_default_aux_hidden_state_layers() == (
        2,
        46,
        90,
    )


def test_kimi_linear_forward_extracts_standard_aux_hidden_states(monkeypatch):
    model = _make_kimi_linear_model()
    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    layer_hidden_states = torch.tensor([[3.0, 4.0]])
    layer_residual = torch.tensor([[5.0, 6.0]])

    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", 1)
    object.__setattr__(
        model,
        "layers",
        [Mock(return_value=(layer_hidden_states, None, layer_residual))],
    )
    object.__setattr__(model, "aux_hidden_state_layers", (0, 1))
    object.__setattr__(model, "use_attn_res", False)
    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    expected_layer_output = layer_hidden_states + layer_residual
    torch.testing.assert_close(output, expected_layer_output)
    torch.testing.assert_close(aux_hidden_states[0], initial_hidden_states)
    torch.testing.assert_close(aux_hidden_states[1], expected_layer_output)


def test_kimi_linear_forward_extracts_attn_res_aux_hidden_states(monkeypatch):
    model = _make_kimi_linear_model()
    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    layer_hidden_states = torch.tensor([[3.0, 4.0]])
    prefix_sum = torch.tensor([[5.0, 6.0]])
    block_residual = torch.tensor([[[7.0, 8.0]]])
    final_hidden_states = torch.tensor([[9.0, 10.0]])

    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", 1)
    object.__setattr__(
        model,
        "layers",
        [Mock(return_value=(layer_hidden_states, prefix_sum, block_residual))],
    )
    object.__setattr__(model, "aux_hidden_state_layers", (0, 1))
    object.__setattr__(model, "use_attn_res", True)
    object.__setattr__(model, "num_attn_res_blocks", 1)
    object.__setattr__(
        model,
        "output_attn_res_norm",
        SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-5),
    )
    object.__setattr__(
        model,
        "output_attn_res_proj",
        SimpleNamespace(weight=torch.ones(1, 2)),
    )
    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    final_attn_res = Mock(return_value=final_hidden_states)
    monkeypatch.setattr(kimi_model, "attn_res", final_attn_res)

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    torch.testing.assert_close(output, final_hidden_states)
    torch.testing.assert_close(aux_hidden_states[0], initial_hidden_states)
    torch.testing.assert_close(aux_hidden_states[1], prefix_sum + layer_hidden_states)
    assert final_attn_res.call_args.args[2] is block_residual


def test_attn_res_stream_capture_receives_the_layer_outputs_in_order(monkeypatch):
    """Pin the argument mapping at the call site.

    The capture helper's own tests invoke it directly by keyword, so they
    cannot catch a swap where `forward` hands it the residual as the pending
    MLP output. Both are tensors of the same shape, so a swap is silent: it
    feeds the drafter a wrong but well-formed tensor.
    """
    model = _make_kimi_linear_model()
    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    layer_hidden_states = torch.tensor([[3.0, 4.0]])
    prefix_sum = torch.tensor([[5.0, 6.0]])
    block_residual = torch.tensor([[[7.0, 8.0]]])
    captured = torch.tensor([[11.0, 12.0]])

    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", 1)
    object.__setattr__(
        model,
        "layers",
        [Mock(return_value=(layer_hidden_states, prefix_sum, block_residual))],
    )
    object.__setattr__(model, "aux_hidden_state_layers", (1,))
    object.__setattr__(model, "use_attn_res", True)
    object.__setattr__(model, "num_attn_res_blocks", 1)
    object.__setattr__(
        model,
        "output_attn_res_norm",
        SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-5),
    )
    object.__setattr__(
        model,
        "output_attn_res_proj",
        SimpleNamespace(weight=torch.ones(1, 2)),
    )
    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(kimi_model, "attn_res", Mock(return_value=torch.zeros(1, 2)))
    monkeypatch.setenv("VLLM_KIMI_K3_AUX_ATTN_RES_STREAM", "1")

    capture = Mock(return_value=captured)
    monkeypatch.setattr(KimiLinearModel, "_capture_aux_hidden_stream", capture)

    _, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    layer_idx, got_prefix, got_pending, got_residual = capture.call_args.args
    assert layer_idx == 0
    assert got_prefix is prefix_sum
    assert got_pending is layer_hidden_states
    assert got_residual is block_residual
    torch.testing.assert_close(aux_hidden_states[0], captured)


class _AmdAttnResLayer:
    """Stand in for an AMD layer whose AttnRes fills the snapshot it is given."""

    def __init__(self, prefix_sum: torch.Tensor, mlp_output: torch.Tensor) -> None:
        self.prefix_sum = prefix_sum
        self.mlp_output = mlp_output
        self.prefix_snapshot: torch.Tensor | None | str = "not called"

    def __call__(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        prefix_delta: torch.Tensor | None,
        prefix_snapshot: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.prefix_snapshot = prefix_snapshot
        if prefix_snapshot is not None:
            # The fused kernel writes this sum while it updates the prefix.
            torch.add(hidden_states, prefix_delta, out=prefix_snapshot)
        return self.prefix_sum, residual, self.mlp_output


@pytest.mark.parametrize("is_last_rank", [True, False])
def test_amd_kimi_linear_forward_fills_aux_hidden_states_in_attn_res(
    monkeypatch, is_last_rank: bool
):
    """The next AttnRes fills the aux buffer of a target layer. If no AttnRes
    follows on this rank, one add fills it."""
    model = _make_kimi_linear_model(amd_linear.KimiLinearModel)
    object.__setattr__(model, "config", SimpleNamespace(attn_res_block_size=2))
    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", 3)
    object.__setattr__(model, "aux_hidden_state_layers", (1, 3))
    layers = [
        _AmdAttnResLayer(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])),
        _AmdAttnResLayer(torch.tensor([[5.0, 6.0]]), torch.tensor([[7.0, 8.0]])),
        _AmdAttnResLayer(torch.tensor([[9.0, 10.0]]), torch.tensor([[11.0, 12.0]])),
    ]
    object.__setattr__(model, "layers", layers)
    object.__setattr__(model, "output_attn_res_norm", Mock())
    object.__setattr__(model, "output_attn_res_proj", Mock())
    monkeypatch.setattr(
        amd_linear,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=is_last_rank),
    )
    final_hidden_states = torch.tensor([[13.0, 14.0]])

    def fake_apply_attn_res(prefix_sum, block_residual, proj, norm, num_blocks, **kw):
        if kw["prefix_snapshot"] is not None:
            torch.add(prefix_sum, kw["delta"], out=kw["prefix_snapshot"])
        return final_hidden_states

    monkeypatch.setattr(amd_linear, "_apply_attn_res", fake_apply_attn_res)

    result = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.tensor([[0.5, 1.5]]),
    )

    # Only the layer after a target layer receives a buffer.
    assert layers[0].prefix_snapshot is None
    assert layers[2].prefix_snapshot is None
    first_snapshot = layers[1].prefix_snapshot
    assert first_snapshot is not None
    torch.testing.assert_close(
        first_snapshot, layers[0].prefix_sum + layers[0].mlp_output
    )

    last_sum = layers[2].prefix_sum + layers[2].mlp_output
    if is_last_rank:
        output, aux_hidden_states = result
        assert output is final_hidden_states
        # The drafter reads the buffers the kernel wrote, not copies of them.
        assert aux_hidden_states[0] is first_snapshot
        torch.testing.assert_close(aux_hidden_states[1], last_sum)
    else:
        # No AttnRes follows the last layer on this rank, so one add remains.
        torch.testing.assert_close(result["hidden_states"], last_sum)


class _StopAfterAttnRes(Exception):
    pass


@pytest.mark.parametrize("with_snapshot", [True, False])
def test_amd_decoder_layer_hands_prefix_snapshot_to_attn_res(
    monkeypatch, with_snapshot: bool
):
    """The pre-attention AttnRes receives the snapshot that forward() gets."""
    layer = object.__new__(amd_linear.KimiDecoderLayer)
    for name, value in {
        "use_attn_residuals": True,
        "self_attn": SimpleNamespace(),
        "self_attention_res_proj": Mock(),
        "self_attention_res_norm": Mock(),
        "input_layernorm": Mock(),
        "prev_valid_blocks": 2,
        "block_write_idx": 1,
        "is_block_write_layer": False,
    }.items():
        object.__setattr__(layer, name, value)
    calls = []

    def fake_apply_attn_res(prefix_sum, block_residual, proj, norm, num_blocks, **kw):
        calls.append((prefix_sum, block_residual, kw))
        raise _StopAfterAttnRes

    monkeypatch.setattr(amd_linear, "_apply_attn_res", fake_apply_attn_res)
    hidden_states = torch.tensor([[1.0, 2.0]])
    residual = torch.zeros(1, 3, 2)
    prefix_delta = torch.tensor([[3.0, 4.0]])
    prefix_snapshot = torch.empty(1, 2) if with_snapshot else None

    with pytest.raises(_StopAfterAttnRes):
        layer.forward(
            torch.tensor([0]),
            hidden_states,
            residual,
            prefix_delta=prefix_delta,
            prefix_snapshot=prefix_snapshot,
        )

    ((got_prefix, got_residual, kw),) = calls
    assert got_prefix is hidden_states
    assert got_residual is residual
    assert kw["delta"] is prefix_delta
    assert kw["prefix_snapshot"] is prefix_snapshot
    assert kw["block_write_idx"] == -1
