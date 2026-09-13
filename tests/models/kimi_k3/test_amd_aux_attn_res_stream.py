# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which value the DFlash drafter is fed under AttnRes, on the AMD path.

The AMD twin of ``test_aux_attn_res_stream.py``. A layer returns the running
prefix sum, but its consumer reads a learned mixture over the per-block
residual bank, so tapping the prefix hands the drafter the right shape with
the wrong value: it costs acceptance and raises nothing. The mixture itself is
covered by ``test_amd_attn_res.py``; what is asserted here is the selection.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.amd import linear as k3_amd

END_LAYER = 4
BLOCK_SIZE = 2


def _weights(tag: float) -> SimpleNamespace:
    """A norm/projection pair that is identifiable by value."""
    return SimpleNamespace(weight=torch.full((2,), tag), variance_epsilon=tag)


def _stub_model(*, enabled: bool = True, use_attn_res: bool = True) -> SimpleNamespace:
    """A stand-in carrying only what the tap reads: the real model needs a
    distributed init and weights, none of which shapes the selection."""
    return SimpleNamespace(
        _aux_attn_res_stream=enabled,
        use_attn_res=use_attn_res,
        config=SimpleNamespace(attn_res_block_size=BLOCK_SIZE),
        end_layer=END_LAYER,
        layers=[
            SimpleNamespace(
                self_attention_res_norm=_weights(float(i)),
                self_attention_res_proj=SimpleNamespace(
                    weight=torch.full((1, 2), float(i))
                ),
                prev_valid_blocks=i,
            )
            for i in range(END_LAYER)
        ],
        output_attn_res_norm=_weights(99.0),
        output_attn_res_proj=SimpleNamespace(weight=torch.full((1, 2), 99.0)),
    )


@pytest.fixture
def recorder(monkeypatch):
    """Replace the kernel so the call it would have made is inspectable."""
    calls = []

    def _fake_apply(prefix, block_residual, proj, norm, num_valid_blocks, **kwargs):
        calls.append(
            SimpleNamespace(
                prefix=prefix,
                block_residual=block_residual,
                proj=proj,
                norm=norm,
                num_valid_blocks=num_valid_blocks,
            )
        )
        return torch.full_like(prefix, -1.0)

    monkeypatch.setattr(k3_amd, "_apply_attn_res", _fake_apply)
    return calls


def _set_last_rank(monkeypatch, is_last: bool):
    monkeypatch.setattr(
        k3_amd,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=is_last),
    )


def _call(stub, layer_idx, prefix, block_residual):
    return k3_amd.KimiLinearModel._capture_aux_hidden_stream(
        stub, layer_idx, prefix, block_residual
    )


@pytest.mark.parametrize("enabled,use_attn_res", [(False, True), (True, False)])
def test_off_reproduces_the_plain_prefix(recorder, enabled, use_attn_res):
    """Off, the tap must be exactly the tensor it replaced. Both conditions
    matter: the block size is what CONSTRUCTS the weights the lookups read, so
    without it they would raise instead of falling back."""
    prefix = torch.tensor([1.0, 2.0])

    stub = _stub_model(enabled=enabled, use_attn_res=use_attn_res)

    got = _call(stub, 0, prefix, torch.zeros(2))

    torch.testing.assert_close(got, prefix)
    assert not recorder, "the kernel must not run when the tap is off"


def test_taps_the_consumer_layer_when_one_follows(recorder, monkeypatch):
    """The value the next layer reads is the mixture against *its* weights, so
    the tap has to reach forward rather than use the current layer's."""
    _set_last_rank(monkeypatch, True)

    _call(_stub_model(), 1, torch.zeros(2), torch.zeros(2))

    assert len(recorder) == 1
    # Layer 2's weights, not layer 1's.
    torch.testing.assert_close(recorder[0].norm.weight, torch.full((2,), 2.0))
    assert recorder[0].num_valid_blocks == 2


def test_last_layer_on_the_final_rank_uses_the_output_aggregation(
    recorder, monkeypatch
):
    """Nothing downstream but the model's own output-side mixture, counted the
    way ``forward`` counts it: blocks over the whole stack."""
    _set_last_rank(monkeypatch, True)

    _call(_stub_model(), END_LAYER - 1, torch.zeros(2), torch.zeros(2))

    assert len(recorder) == 1
    torch.testing.assert_close(recorder[0].norm.weight, torch.full((2,), 99.0))
    assert recorder[0].num_valid_blocks == k3_amd.cdiv(END_LAYER, BLOCK_SIZE)


def test_last_layer_off_the_final_rank_falls_back_to_the_prefix(recorder, monkeypatch):
    """The consumer lives on the next rank and the output-side aggregation only
    exists on the last one, so there is nothing here to mix against."""
    _set_last_rank(monkeypatch, False)
    prefix = torch.tensor([1.0, 2.0])

    got = _call(_stub_model(), END_LAYER - 1, prefix, torch.zeros(2))

    torch.testing.assert_close(got, prefix)
    assert not recorder, "no weights on this rank to mix against"


def test_aux_layer_at_a_non_final_pp_boundary_is_rejected(monkeypatch):
    """Rejected once, at configuration time, rather than silently degrading."""
    model = k3_amd.KimiLinearModel.__new__(k3_amd.KimiLinearModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(attn_res_block_size=BLOCK_SIZE)
    model.end_layer = 72
    monkeypatch.setattr(
        k3_amd.envs, "VLLM_KIMI_K3_AUX_ATTN_RES_STREAM", True, raising=False
    )
    # The mixin caches a PP layout first; keep it out of the distributed state.
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.model_parallel_is_initialized", lambda: False
    )
    _set_last_rank(monkeypatch, False)

    with pytest.raises(ValueError, match="Auxiliary layer 72"):
        model._set_aux_hidden_state_layers((3, 24, 48, 72, 90))


def test_forward_taps_the_capture_and_not_the_layer_output(monkeypatch):
    """Pin the argument mapping at the call site.

    The tests above invoke the capture directly, so they pass even if `forward`
    never calls it, or hands it the wrong tensor: the prefix and the bank are
    both tensors and a swap is silent.
    """
    model = k3_amd.KimiLinearModel.__new__(k3_amd.KimiLinearModel)
    torch.nn.Module.__init__(model)
    hidden = torch.tensor([[1.0, 2.0]])
    bank = torch.tensor([[[3.0, 4.0]]])
    captured = torch.tensor([[9.0, 9.0]])

    model.config = SimpleNamespace(attn_res_block_size=BLOCK_SIZE, hidden_size=2)
    model.start_layer, model.end_layer = 0, 1
    model.layers = [lambda **kw: (hidden, bank)]
    model.aux_hidden_state_layers = (1,)
    model.embed_tokens = lambda ids: hidden
    model.output_attn_res_norm = _weights(1.0)
    model.output_attn_res_proj = SimpleNamespace(weight=torch.ones(1, 2))

    seen = []
    monkeypatch.setattr(
        k3_amd.KimiLinearModel,
        "_capture_aux_hidden_stream",
        lambda self, *a: (seen.append(a), captured)[1],
    )
    monkeypatch.setattr(k3_amd, "_apply_attn_res", lambda *a, **k: hidden)
    _set_last_rank(monkeypatch, True)

    _, aux = model.forward(
        input_ids=torch.zeros(1, dtype=torch.long),
        positions=torch.zeros(1, dtype=torch.long),
        intermediate_tensors=None,
        inputs_embeds=None,
    )

    assert seen == [(0, hidden, bank)], "capture must get (layer_idx, prefix, bank)"
    torch.testing.assert_close(aux[0], captured)
