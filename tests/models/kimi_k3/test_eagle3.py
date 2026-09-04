# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia.model import (
    KimiK3ForConditionalGeneration,
    KimiLinearForCausalLM,
    KimiLinearModel,
)


def _make_kimi_linear_model() -> KimiLinearModel:
    model = object.__new__(KimiLinearModel)
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


def _make_stage(
    *,
    start_layer: int,
    taps: tuple[int, ...],
    layer_outputs: list[tuple[torch.Tensor, None, torch.Tensor]],
) -> KimiLinearModel:
    model = _make_kimi_linear_model()
    end_layer = start_layer + len(layer_outputs)
    object.__setattr__(model, "start_layer", start_layer)
    object.__setattr__(model, "end_layer", end_layer)
    # The real model keeps the global layer list and slices [start:end].
    layers = [Mock() for _ in range(end_layer)]
    for i, out in enumerate(layer_outputs):
        layers[start_layer + i] = Mock(return_value=out)
    object.__setattr__(model, "layers", layers)
    object.__setattr__(model, "aux_hidden_state_layers", taps)
    object.__setattr__(model, "config", SimpleNamespace(hidden_size=2))
    return model


def test_kimi_linear_aux_hidden_states_flow_across_pp_stages(monkeypatch):
    """A tap owned by an earlier PP stage must reach the last stage intact.

    The drafter's taps can reference layers outside the last stage (K3 taps
    [24, 48, 72, 88, 92]); each stage packs the taps it owns into
    IntermediateTensors and the last stage returns the full ordered set.
    """
    stage0_hidden = torch.tensor([[1.0, 2.0]])
    stage0_residual = torch.tensor([[3.0, 4.0]])
    stage1_hidden = torch.tensor([[5.0, 6.0]])
    stage1_residual = torch.tensor([[7.0, 8.0]])

    stage0 = _make_stage(
        start_layer=0,
        taps=(1, 2),
        layer_outputs=[(stage0_hidden, None, stage0_residual)],
    )
    stage1 = _make_stage(
        start_layer=1,
        taps=(1, 2),
        layer_outputs=[(stage1_hidden, None, stage1_residual)],
    )

    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )
    stage0_out = stage0.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.zeros(1, 2),
    )

    # Stage 0 owns the post-layer-1 tap; it is packed for the wire.
    stage0_aux = stage0_hidden + stage0_residual
    torch.testing.assert_close(stage0_out.tensors["aux_hidden_states"], stage0_aux)

    # The receiving buffer on stage 1 must be sized for exactly that one tap.
    stage1_buffers = stage1.make_empty_intermediate_tensors(
        batch_size=1, dtype=torch.bfloat16, device=torch.device("cpu")
    )
    assert stage1_buffers.tensors["aux_hidden_states"].shape == (1, 2)

    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    output, aux_hidden_states = stage1.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=stage0_out,
    )

    # The boundary tap (position 1 == stage1's start_layer) must not be
    # duplicated by the stage-entry capture: two taps, in ascending order.
    assert len(aux_hidden_states) == 2
    torch.testing.assert_close(aux_hidden_states[0], stage0_aux)
    torch.testing.assert_close(aux_hidden_states[1], stage1_hidden + stage1_residual)
    torch.testing.assert_close(output, stage1_hidden + stage1_residual)


def test_kimi_linear_first_stage_without_taps_sends_no_aux_buffer(monkeypatch):
    """No taps at or below the stage boundary -> no aux key on the wire."""
    stage0 = _make_stage(
        start_layer=0,
        taps=(2,),
        layer_outputs=[(torch.ones(1, 2), None, torch.zeros(1, 2))],
    )
    assert (
        "aux_hidden_states"
        not in stage0.make_empty_intermediate_tensors(
            batch_size=1, dtype=torch.bfloat16, device=torch.device("cpu")
        ).tensors
    )

    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )
    out = stage0.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.zeros(1, 2),
    )
    assert "aux_hidden_states" not in out.tensors
