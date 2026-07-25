# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode import extract_hidden_states as spec_module
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
    ExtractHiddenStatesSpeculator,
)


class _RecordingModel(torch.nn.Module):
    def forward(self, *, hidden_states: torch.Tensor) -> None:
        self.hidden_states = hidden_states.clone()


def test_init_speculator_dispatches_extract_hidden_states(monkeypatch):
    vllm_config = cast(
        Any,
        SimpleNamespace(
            speculative_config=SimpleNamespace(method="extract_hidden_states")
        ),
    )
    device = torch.device("cpu")

    def fake_speculator(config, target_device):
        return config, target_device

    monkeypatch.setattr(spec_module, "ExtractHiddenStatesSpeculator", fake_speculator)

    assert init_speculator(vllm_config, device) == (vllm_config, device)


def test_propose_caches_hidden_states_and_returns_sampled_tokens(monkeypatch):
    contexts = []

    def fake_set_forward_context(*args, **kwargs):
        contexts.append((args, kwargs))
        return nullcontext()

    monkeypatch.setattr(spec_module, "set_forward_context", fake_set_forward_context)

    layer_name = "cache_only_layers.2"
    speculator = object.__new__(ExtractHiddenStatesSpeculator)
    speculator.vllm_config = cast(Any, SimpleNamespace())
    speculator.num_hidden_states = 2
    speculator.hidden_states = torch.zeros(4, 2, 3)
    speculator.draft_attn_layer_names = {layer_name}
    speculator.model = _RecordingModel()

    input_batch = cast(
        Any,
        SimpleNamespace(
            idx_mapping=torch.tensor([2, 0], dtype=torch.int32),
            is_padding=torch.zeros(4, dtype=torch.bool),
        ),
    )
    aux_hidden_states = [
        torch.full((4, 3), 1.0),
        torch.full((4, 3), 2.0),
    ]
    attn_metadata = {layer_name: object(), "target_layer": object()}
    slot_mappings = {
        layer_name: torch.arange(4),
        "target_layer": torch.arange(4),
    }
    last_sampled = torch.tensor([[10], [11], [12]], dtype=torch.int64)

    draft_tokens = speculator.propose(
        input_batch=input_batch,
        attn_metadata=attn_metadata,
        slot_mappings=slot_mappings,
        last_hidden_states=torch.empty(0),
        aux_hidden_states=aux_hidden_states,
        num_sampled=torch.empty(0),
        num_rejected=torch.empty(0),
        last_sampled=last_sampled,
        next_prefill_tokens=torch.empty(0),
        temperature=torch.empty(0),
        seeds=torch.empty(0),
    )

    expected_hidden_states = torch.stack(aux_hidden_states, dim=1)
    assert torch.equal(speculator.model.hidden_states, expected_hidden_states)
    assert torch.equal(draft_tokens, torch.tensor([[12], [10]]))

    assert len(contexts) == 1
    args, kwargs = contexts[0]
    assert args[0] == {layer_name: attn_metadata[layer_name]}
    assert kwargs["num_tokens"] == 4
    assert set(kwargs["slot_mapping"]) == {layer_name}
    assert torch.equal(kwargs["slot_mapping"][layer_name], slot_mappings[layer_name])


def test_propose_requires_aux_hidden_states():
    speculator = object.__new__(ExtractHiddenStatesSpeculator)
    speculator.num_hidden_states = 2
    input_batch = cast(
        Any, SimpleNamespace(idx_mapping=torch.tensor([0], dtype=torch.int32))
    )

    with pytest.raises(ValueError, match="aux_hidden_states are required"):
        speculator.propose(
            input_batch=input_batch,
            attn_metadata={},
            slot_mappings={},
            last_hidden_states=torch.empty(0),
            aux_hidden_states=None,
            num_sampled=torch.empty(0),
            num_rejected=torch.empty(0),
            last_sampled=torch.tensor([[10]]),
            next_prefill_tokens=torch.empty(0),
            temperature=torch.empty(0),
            seeds=torch.empty(0),
        )
