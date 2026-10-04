# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash pipeline-parallel stage boundary.

The mHC configs carry ``mhc_num_residual_streams`` residual streams per token
and defer each layer's ``hc_post`` mix into the next layer's fused post+pre
kernel. At a PP boundary there is no next layer, so ``Glm5NextModel.forward``
materializes the pending ``hc_post`` on the sending rank and ships only the
residual streams; the receiving rank's first layer then takes the standalone
``hc_pre`` path. These tests pin that wiring (and the shapes/keys of
``make_empty_intermediate_tensors``) without weights, a process group or a GPU.
"""

import types
from unittest import mock

import pytest
import torch
from torch import nn

import vllm.models.glm5next.common.model as glm_model
from vllm.model_executor.models.interfaces import supports_pp
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory
from vllm.sequence import IntermediateTensors

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

HIDDEN = 128
N_STREAMS = 4
NUM_TOKENS = 6
LAST_LAYER = 44


def _make_empty_mhc(batch_size: int, dtype: torch.dtype) -> IntermediateTensors:
    """Call the mHC factory against a stand-in carrying the attrs it reads."""
    stub = types.SimpleNamespace(
        mhc_num_residual_streams=N_STREAMS,
        config=types.SimpleNamespace(hidden_size=HIDDEN),
    )
    return glm_model.Glm5NextModel._make_empty_mhc_intermediate_tensors(
        stub, batch_size=batch_size, dtype=dtype, device=torch.device("cpu")
    )


def test_make_empty_intermediate_tensors_mhc():
    out = _make_empty_mhc(17, torch.bfloat16)
    assert isinstance(out, IntermediateTensors)
    # A single key: the deferred hc_post state is materialized into the
    # streams before the send, so no separate "residual" copy is needed.
    assert list(out.tensors) == ["hidden_states"]
    hidden_states = out["hidden_states"]
    assert hidden_states.shape == (17, N_STREAMS, HIDDEN)
    assert hidden_states.dtype == torch.bfloat16
    assert hidden_states.device == torch.device("cpu")
    # The runner slices the buffer per batch, so tokens must lead.
    assert out[:5]["hidden_states"].shape == (5, N_STREAMS, HIDDEN)


def test_make_empty_intermediate_tensors_non_mhc():
    factory = make_empty_intermediate_tensors_factory(
        ["hidden_states", "residual"], HIDDEN
    )
    out = factory(batch_size=9, dtype=torch.float16, device=torch.device("cpu"))
    assert list(out.tensors) == ["hidden_states", "residual"]
    for key in ("hidden_states", "residual"):
        assert out[key].shape == (9, HIDDEN)
        assert out[key].dtype == torch.float16


class _FakeLayer(nn.Module):
    """Records what it receives; mimics the decoder layer's return contract."""

    def __init__(self, layer_idx: int, is_last_layer: bool, mhc: bool):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_last_layer = is_last_layer
        self.mhc = mhc
        self.seen: tuple | None = None
        self.hc_post_calls = 0

    def forward(self, positions, hidden_states, residual, post, comb):
        self.seen = (hidden_states, residual, post, comb)
        if not self.mhc:
            return hidden_states + 1, hidden_states, None, None
        if post is None:
            # Standalone hc_pre: expand only at layer 0, the streams become
            # the residual and the layer input is derived from them.
            x = hidden_states
            if self.layer_idx == 0:
                x = x.unsqueeze(1).expand(-1, N_STREAMS, -1).contiguous()
            residual = x
            x = residual[:, 0, :]
        else:
            # Fused post+pre: a 2-D layer input plus the 3-D streams.
            assert hidden_states.dim() == 2
            assert residual.dim() == 3
            x = hidden_states
        if self.is_last_layer:
            return residual.mean(dim=1), None, None, None
        num_tokens = residual.shape[0]
        idx = float(self.layer_idx)
        return (
            x * 2,
            residual,
            torch.full((num_tokens, N_STREAMS, 1), idx),
            torch.full((num_tokens, N_STREAMS, N_STREAMS), idx),
        )

    def hc_post(self, x, residual, post, comb):
        self.hc_post_calls += 1
        return residual + x.unsqueeze(1)


def _build_model(mhc: bool, layer_ids: list[int]) -> glm_model.Glm5NextModel:
    model = glm_model.Glm5NextModel.__new__(glm_model.Glm5NextModel)
    nn.Module.__init__(model)
    model.mhc = mhc
    model.mhc_num_residual_streams = N_STREAMS
    model.config = types.SimpleNamespace(hidden_size=HIDDEN)
    model.is_sequence_parallel = False
    model._active_layers = [_FakeLayer(i, i == LAST_LAYER, mhc) for i in layer_ids]
    model.norm = lambda hidden_states: hidden_states * 10
    model.embed_tokens = lambda input_ids: torch.zeros(
        input_ids.shape[0], HIDDEN, dtype=torch.bfloat16
    )
    return model


def _run_rank(model, *, is_first: bool, is_last: bool, intermediate_tensors):
    pp_group = types.SimpleNamespace(is_first_rank=is_first, is_last_rank=is_last)
    input_ids = torch.zeros(NUM_TOKENS, dtype=torch.long) if is_first else None
    with mock.patch.object(glm_model, "get_pp_group", return_value=pp_group):
        return model(input_ids, torch.arange(NUM_TOKENS), intermediate_tensors)


def _streams() -> torch.Tensor:
    return torch.randn(NUM_TOKENS, N_STREAMS, HIDDEN, dtype=torch.bfloat16)


def test_middle_rank_materializes_pending_hc_post():
    model = _build_model(True, [10, 11, 12])
    streams = _streams()
    out = _run_rank(
        model,
        is_first=False,
        is_last=False,
        intermediate_tensors=IntermediateTensors({"hidden_states": streams}),
    )

    first = model._active_layers[0]
    # The received streams go in as-is and the first layer takes the
    # standalone hc_pre path (no deferred state to fuse).
    assert first.seen[0] is streams
    assert first.seen[1:] == (None, None, None)

    assert list(out.tensors) == ["hidden_states"]
    assert out["hidden_states"].shape == (NUM_TOKENS, N_STREAMS, HIDDEN)
    assert out["hidden_states"].dtype == streams.dtype
    # Exactly the last active layer materializes its deferred hc_post.
    assert model._active_layers[-1].hc_post_calls == 1
    assert all(layer.hc_post_calls == 0 for layer in model._active_layers[:-1])
    x_last = streams[:, 0, :] * 2 ** len(model._active_layers)
    assert torch.equal(out["hidden_states"], streams + x_last.unsqueeze(1))


def test_first_rank_embeds_and_sends_streams():
    model = _build_model(True, [0, 1])
    out = _run_rank(model, is_first=True, is_last=False, intermediate_tensors=None)

    first = model._active_layers[0]
    assert first.seen[0].dim() == 2  # embeddings, expanded inside layer 0
    assert first.seen[1:] == (None, None, None)
    assert list(out.tensors) == ["hidden_states"]
    assert out["hidden_states"].shape == (NUM_TOKENS, N_STREAMS, HIDDEN)
    assert model._active_layers[-1].hc_post_calls == 1


def test_last_rank_returns_hidden_states():
    model = _build_model(True, [43, LAST_LAYER])
    out = _run_rank(
        model,
        is_first=False,
        is_last=True,
        intermediate_tensors=IntermediateTensors({"hidden_states": _streams()}),
    )
    assert isinstance(out, torch.Tensor)
    assert out.shape == (NUM_TOKENS, HIDDEN)
    # The final layer already contracts; no boundary materialization here.
    assert all(layer.hc_post_calls == 0 for layer in model._active_layers)


def test_mhc_rank_rejects_two_dimensional_input():
    model = _build_model(True, [10])
    flat = torch.zeros(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        _run_rank(
            model,
            is_first=False,
            is_last=False,
            intermediate_tensors=IntermediateTensors({"hidden_states": flat}),
        )


def test_non_mhc_rank_round_trips_hidden_and_residual():
    model = _build_model(False, [10, 11])
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    residual = torch.randn(NUM_TOKENS, HIDDEN)
    out = _run_rank(
        model,
        is_first=False,
        is_last=False,
        intermediate_tensors=IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        ),
    )

    first = model._active_layers[0]
    assert first.seen[0] is hidden_states
    assert first.seen[1] is residual
    assert sorted(out.tensors) == ["hidden_states", "residual"]
    assert out["hidden_states"].shape == (NUM_TOKENS, HIDDEN)
    assert out["residual"].shape == (NUM_TOKENS, HIDDEN)
    assert all(layer.hc_post_calls == 0 for layer in model._active_layers)


@pytest.mark.parametrize(
    "model_cls",
    [glm_model.Glm5NextForCausalLM, glm_model.Glm5NextForConditionalGeneration],
)
def test_supports_pp(model_cls):
    # The gate the runner uses to decide PP eligibility.
    assert supports_pp(model_cls)
