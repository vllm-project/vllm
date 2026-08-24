# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which value the DFlash drafter is fed under AttnRes.

`_capture_aux_hidden_stream` picks the weights it mixes against from one of
three places depending on where the tapped layer sits, and returns the plain
running prefix when the feature is off. The mixture itself is the kernel's
job and is covered by ``test_attn_res.py``; what is asserted here is the
selection, which is the part that can silently feed the drafter the wrong
tensor.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia import model as k3_model

END_LAYER = 4


def _weights(tag: float) -> SimpleNamespace:
    """A norm/projection pair that is identifiable by value."""
    return SimpleNamespace(
        weight=torch.full((2,), tag),
        variance_epsilon=tag,
    )


def _stub_model(*, enabled: bool, use_attn_res: bool = True) -> SimpleNamespace:
    """A stand-in carrying only what the tap reads.

    Constructing the real model needs a distributed init and weights, and none
    of it participates in the selection under test.
    """
    consumers = []
    for i in range(END_LAYER):
        consumers.append(
            SimpleNamespace(
                self_attention_res_norm=_weights(float(i)),
                self_attention_res_proj=SimpleNamespace(
                    weight=torch.full((1, 2), float(i))
                ),
                prev_valid_blocks=i,
            )
        )
    return SimpleNamespace(
        _aux_attn_res_stream=enabled,
        use_attn_res=use_attn_res,
        end_layer=END_LAYER,
        layers=consumers,
        output_attn_res_norm=_weights(99.0),
        output_attn_res_proj=SimpleNamespace(weight=torch.full((1, 2), 99.0)),
        num_attn_res_blocks=99,
    )


@pytest.fixture
def recorder(monkeypatch):
    """Replace the kernel so the call it would have made is inspectable."""
    calls = []

    def _fake_attn_res(
        prefix,
        delta,
        block_residual,
        norm_weight,
        proj_weight,
        output_norm_weight,
        **kwargs,
    ):
        calls.append(
            SimpleNamespace(
                prefix=prefix,
                delta=delta,
                block_residual=block_residual,
                norm_weight=norm_weight,
                proj_weight=proj_weight,
                kwargs=kwargs,
            )
        )
        return torch.full_like(prefix, -1.0)

    monkeypatch.setattr(k3_model, "attn_res", _fake_attn_res)
    return calls


def _set_last_rank(monkeypatch, is_last: bool):
    monkeypatch.setattr(
        k3_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=is_last),
    )


def _call(stub, layer_idx, prefix_sum, pending_mlp_out, block_residual):
    return k3_model.KimiLinearModel._capture_aux_hidden_stream(
        stub, layer_idx, prefix_sum, pending_mlp_out, block_residual
    )


@pytest.mark.parametrize(
    "enabled,use_attn_res", [(False, True), (True, False), (False, False)]
)
def test_disabled_reproduces_the_plain_residual_sum(
    recorder, monkeypatch, enabled, use_attn_res
):
    """Off, the tap must be exactly the sum it replaced.

    Both conditions matter. `use_attn_res` is what constructs the norm and
    projection weights, so without it the lookups below would raise rather
    than fall back.
    """
    _set_last_rank(monkeypatch, True)
    prefix_sum = torch.tensor([1.0, 2.0])
    pending = torch.tensor([0.5, 0.25])

    got = _call(
        _stub_model(enabled=enabled, use_attn_res=use_attn_res),
        0,
        prefix_sum,
        pending,
        torch.zeros(2),
    )

    torch.testing.assert_close(got, prefix_sum + pending)
    assert not recorder, "the kernel must not run when the tap is off"


def test_taps_the_consumer_layer_when_one_follows(recorder, monkeypatch):
    """The value the next layer reads is the mixture against *its* weights,
    so the tap has to reach forward rather than use the current layer's."""
    _set_last_rank(monkeypatch, True)

    _call(_stub_model(enabled=True), 1, torch.zeros(2), None, torch.zeros(2))

    assert len(recorder) == 1
    call = recorder[0]
    # Layer 2's weights, not layer 1's.
    torch.testing.assert_close(call.norm_weight, torch.full((2,), 2.0))
    assert call.kwargs["num_blocks"] == 2


def test_last_layer_on_the_final_rank_uses_the_output_aggregation(
    recorder, monkeypatch
):
    """Nothing downstream but the model's own output-side mixture."""
    _set_last_rank(monkeypatch, True)

    _call(
        _stub_model(enabled=True), END_LAYER - 1, torch.zeros(2), None, torch.zeros(2)
    )

    assert len(recorder) == 1
    torch.testing.assert_close(recorder[0].norm_weight, torch.full((2,), 99.0))
    assert recorder[0].kwargs["num_blocks"] == 99


def test_last_layer_of_a_non_final_stage_falls_back(recorder, monkeypatch):
    """The consumer lives on the next rank and the output aggregation only
    exists on the last one, so there is nothing here to mix against.

    This is the case that would otherwise reach for weights this rank never
    constructs. The forward guard is `layer_idx + 1 < end_layer`, where
    `end_layer` is the rank's own exclusive bound from `get_pp_indices`, so a
    `PPMissingLayer` is unreachable by construction -- the fallback below is
    what makes that true rather than merely likely.
    """
    _set_last_rank(monkeypatch, False)
    prefix_sum = torch.tensor([3.0, 4.0])

    got = _call(
        _stub_model(enabled=True), END_LAYER - 1, prefix_sum, None, torch.zeros(2)
    )

    torch.testing.assert_close(got, prefix_sum)
    assert not recorder, "no weights exist on this rank to mix against"


def test_pending_mlp_output_is_folded_in_rather_than_passed_as_delta(
    recorder, monkeypatch
):
    """The kernel writes an applied delta back into the prefix in place, which
    would double-add it into the live residual stream, so the pending output
    has to arrive already summed into the prefix with `delta` left None."""
    _set_last_rank(monkeypatch, True)
    prefix_sum = torch.tensor([1.0, 2.0])
    pending = torch.tensor([0.5, 0.25])

    _call(_stub_model(enabled=True), 0, prefix_sum, pending, torch.zeros(2))

    assert len(recorder) == 1
    assert recorder[0].delta is None
    torch.testing.assert_close(recorder[0].prefix, prefix_sum + pending)
    # And the caller's tensor is not mutated on the way.
    torch.testing.assert_close(prefix_sum, torch.tensor([1.0, 2.0]))
