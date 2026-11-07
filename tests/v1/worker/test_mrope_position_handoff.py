# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.mrope import MRoPEPositionState
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.mm_pruning import MultiModalPruner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test


def _position_state() -> MRoPEPositionState:
    return MRoPEPositionState(positions=[[0, 1, 7], [0, 2, 7], [0, 3, 7]], delta=-11)


def test_v2_model_state_restores_remote_prefill_positions() -> None:
    model_state = object.__new__(DefaultModelState)
    model_state.rope_state = Mock()
    model_state.prompt_embeds_state = None
    model_state.model = Mock()
    new_request = NewRequestData(
        req_id="req",
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=2,
        lora_request=None,
        prefill_token_ids=[1, 2, 3],
        mrope_position_state=_position_state(),
    )

    model_state.add_request(4, new_request)

    model_state.rope_state.stage_prefill_positions.assert_called_once_with(
        4, _position_state().positions, -11
    )
    model_state.rope_state.init_prefill_positions.assert_not_called()


def test_legacy_model_runner_restores_remote_prefill_positions() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.mrope_num_dims = 3
    request_state = SimpleNamespace(mrope_positions=None, mrope_position_delta=None)

    runner._init_mrope_positions(request_state, _position_state())

    assert torch.equal(
        request_state.mrope_positions,
        torch.tensor(_position_state().positions, dtype=torch.int64),
    )
    assert request_state.mrope_position_delta == -11


def test_pruner_serializes_latest_recomputed_position_state() -> None:
    pruner = object.__new__(MultiModalPruner)
    pruner._pending_position_states = {
        "req": (torch.tensor(_position_state().positions), -11)
    }

    assert pruner.take_position_states() == {"req": _position_state()}
    assert not pruner._pending_position_states
