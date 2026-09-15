# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.amd import linear as kimi_linear


def _layer(scale: float) -> SimpleNamespace:
    layer = SimpleNamespace(
        self_attention_res_proj=object(),
        self_attention_res_norm=object(),
        input_layernorm=object(),
        post_attention_layernorm=object(),
        mlp_res_proj=object(),
        mlp_res_norm=object(),
        prev_valid_blocks=0,
        is_block_write_layer=False,
        block_write_idx=0,
        _run_self_attn=lambda _positions, value: value * scale,
        mlp=lambda value: value * (scale + 1),
    )
    layer.forward_attn_residual_deferred = lambda **kwargs: (
        kimi_linear.KimiDecoderLayer.forward_attn_residual_deferred(layer, **kwargs)
    )
    return layer


@pytest.mark.cpu_test
def test_deferred_attn_res_matches_two_eager_layer_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deferring an MLP add preserves its position before the next AttnRes."""
    deltas: list[torch.Tensor | None] = []

    def fake_apply(prefix, _blocks, *_args, delta=None, **_kwargs):
        deltas.append(None if delta is None else delta.clone())
        if delta is not None:
            prefix.add_(delta)
        return prefix.clone()

    monkeypatch.setattr(kimi_linear, "_apply_attn_res", fake_apply)
    positions = torch.zeros(1, dtype=torch.int64)
    blocks = torch.zeros(1, 1, 2)
    initial = torch.tensor([[1.0, -2.0]])
    layer1 = _layer(2.0)
    layer2 = _layer(0.5)

    eager_prefix, eager_pending, _ = (
        kimi_linear.KimiDecoderLayer.forward_attn_residual_deferred(
            layer1, positions, initial.clone(), blocks.clone()
        )
    )
    eager1 = eager_prefix + eager_pending
    eager1_saved = eager1.clone()
    eager_prefix, eager_pending, _ = (
        kimi_linear.KimiDecoderLayer.forward_attn_residual_deferred(
            layer2, positions, eager1.clone(), blocks.clone()
        )
    )
    eager2 = eager_prefix + eager_pending

    deltas.clear()
    prefix, pending, _ = kimi_linear.KimiDecoderLayer.forward_attn_residual_deferred(
        layer1, positions, initial.clone(), blocks.clone()
    )
    prefix, pending, _ = kimi_linear.KimiDecoderLayer.forward_attn_residual_deferred(
        layer2, positions, prefix, blocks.clone(), pending
    )

    torch.testing.assert_close(prefix + pending, eager2, atol=0, rtol=0)
    assert deltas[0] is None
    torch.testing.assert_close(deltas[1], initial * 2.0)
    # The first layer's MLP output is consumed by layer two's first AttnRes.
    torch.testing.assert_close(deltas[2], eager1_saved - (initial + initial * 2.0))


@pytest.mark.cpu_test
def test_eager_attn_res_entrypoint_materializes_deferred_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = torch.tensor([[3.0]])
    pending = torch.tensor([[4.0]])
    blocks = torch.zeros(1, 1, 1)
    layer = SimpleNamespace(
        forward_attn_residual_deferred=lambda *_args: (prefix, pending, blocks)
    )

    actual, actual_blocks = kimi_linear.KimiDecoderLayer.forward_attn_residual(
        layer, torch.zeros(1, dtype=torch.int64), prefix, blocks
    )

    torch.testing.assert_close(actual, torch.tensor([[7.0]]), atol=0, rtol=0)
    assert actual_blocks is blocks


@pytest.mark.cpu_test
def test_model_preserves_tapped_state_and_folds_final_pending_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aux taps materialize once; the output AttnRes consumes the last delta."""
    calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def fake_apply(prefix, _blocks, *_args, delta=None, **_kwargs):
        calls.append((prefix.clone(), None if delta is None else delta.clone()))
        if delta is not None:
            prefix.add_(delta)
        return prefix.clone()

    monkeypatch.setattr(kimi_linear, "_apply_attn_res", fake_apply)
    monkeypatch.setattr(
        kimi_linear,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(kimi_linear.envs, "VLLM_KIMI_K3_DEFER_ATTN_RES_MLP", True)

    aux_layers = {1}
    captured: list[torch.Tensor] = []

    def maybe_capture(outputs, layer_idx, hidden_states, _residual):
        if layer_idx in aux_layers:
            outputs.append(hidden_states)
            captured.append(hidden_states.clone())
        return outputs

    model = SimpleNamespace(
        config=SimpleNamespace(attn_res_block_size=1),
        start_layer=0,
        end_layer=2,
        layers=[_layer(2.0), _layer(0.5)],
        aux_hidden_state_layers=aux_layers,
        _maybe_add_hidden_state=maybe_capture,
        output_attn_res_proj=object(),
        output_attn_res_norm=object(),
    )
    initial = torch.tensor([[1.0, -2.0]])
    output, aux = kimi_linear.KimiLinearModel.forward(
        model,
        input_ids=None,
        positions=torch.zeros(1, dtype=torch.int64),
        intermediate_tensors=None,
        inputs_embeds=initial.clone(),
    )

    # Layer one's exact post-MLP state is exposed to DSpark and remains stable
    # after layer two consumes its separate pending delta in place.
    torch.testing.assert_close(aux[0], initial * 12.0, atol=0, rtol=0)
    torch.testing.assert_close(captured[0], initial * 12.0, atol=0, rtol=0)
    # The output AttnRes receives layer two's pending MLP result as delta.
    final_prefix, final_delta = calls[-1]
    assert final_delta is not None
    torch.testing.assert_close(final_prefix, initial * 18.0, atol=0, rtol=0)
    torch.testing.assert_close(final_delta, initial * 27.0, atol=0, rtol=0)
    torch.testing.assert_close(output, final_prefix + final_delta, atol=0, rtol=0)
