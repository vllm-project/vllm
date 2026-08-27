# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def _make_speculator() -> tuple[DSparkSpeculator, Mock]:
    speculator = object.__new__(DSparkSpeculator)
    model = Mock()
    model.get_draft_kv_cache_layer_names.return_value = ["draft.0", "draft.1"]
    model.combine_hidden_states.side_effect = lambda states: states[:, :2] + 1
    speculator.model = model
    speculator._pcp_context_kv_precomputed = False
    return speculator, model


def test_precompute_pcp_context_kv_uses_only_local_rows():
    speculator, model = _make_speculator()
    input_batch = SimpleNamespace(
        num_tokens=2,
        positions=torch.tensor([10, 11, 99]),
    )
    aux_hidden_states = [
        torch.tensor([[1.0], [2.0], [90.0]]),
        torch.tensor([[3.0], [4.0], [91.0]]),
    ]
    slot_mappings = {
        "draft.0": torch.tensor([4, 5, 40]),
        "draft.1": torch.tensor([8, 9, 80]),
    }

    speculator.precompute_pcp_context_kv(input_batch, aux_hidden_states, slot_mappings)

    model.combine_hidden_states.assert_called_once()
    args, kwargs = model.precompute_and_store_context_kv.call_args
    assert torch.equal(args[0], torch.tensor([[2.0, 4.0], [3.0, 5.0]]))
    assert torch.equal(args[1], torch.tensor([10, 11]))
    assert len(args[2]) == 2
    assert torch.equal(args[2][0], torch.tensor([4, 5]))
    assert torch.equal(args[2][1], torch.tensor([8, 9]))
    assert kwargs == {"publish_to_pcp": True}
    assert speculator._pcp_context_kv_precomputed

    with pytest.raises(RuntimeError, match="already precomputed"):
        speculator.precompute_pcp_context_kv(
            input_batch, aux_hidden_states, slot_mappings
        )


def test_precompute_pcp_context_kv_rejects_missing_layer_mapping():
    speculator, _ = _make_speculator()
    input_batch = SimpleNamespace(num_tokens=1, positions=torch.tensor([0]))

    with pytest.raises(RuntimeError, match="draft.1"):
        speculator.precompute_pcp_context_kv(
            input_batch,
            [torch.ones(1, 1), torch.ones(1, 1)],
            {"draft.0": torch.tensor([0])},
        )
