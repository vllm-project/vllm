# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The receiving PP stage's aux hidden-state buffers must match what the sending
stage produces. Aux hidden states are read from the residual stream, so they take
the dtype of the model's hidden_states buffer, which can be wider than the model
dtype (e.g. an fp32 residual stream in a bf16 model)."""

from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.parallel_state as parallel_state
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.spec_decode.eagle import eagle3_utils

HIDDEN = 16
AUX_KEYS = ("aux_hidden_states_0", "aux_hidden_states_1")


class _Decoder:
    supports_aux_hidden_states_over_pp = True
    config = SimpleNamespace(hidden_size=HIDDEN)

    def aux_hidden_state_slot_keys(self):
        return AUX_KEYS, ()


def _model(hidden_states_dtype: torch.dtype | None):
    def make_empty(batch_size, dtype, device):
        tensors = {}
        if hidden_states_dtype is not None:
            tensors["hidden_states"] = torch.zeros(
                (batch_size, HIDDEN), dtype=hidden_states_dtype, device=device
            )
        return IntermediateTensors(tensors)

    return SimpleNamespace(make_empty_intermediate_tensors=make_empty)


@pytest.mark.parametrize(
    "hidden_states_dtype,expected",
    [
        (torch.float32, torch.float32),  # residual stream wider than the model
        (torch.bfloat16, torch.bfloat16),
        (None, torch.bfloat16),  # no hidden_states buffer: model dtype
    ],
)
def test_aux_buffers_follow_hidden_states_dtype(
    monkeypatch, hidden_states_dtype, expected
):
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(eagle3_utils, "get_eagle_decoder", lambda model: _Decoder())
    model = _model(hidden_states_dtype)

    eagle3_utils.reserve_aux_intermediate_tensor_slots(model)
    tensors = model.make_empty_intermediate_tensors(4, torch.bfloat16, "cpu")

    for key in AUX_KEYS:
        assert tensors[key].dtype == expected
        assert tensors[key].shape == (4, HIDDEN)
