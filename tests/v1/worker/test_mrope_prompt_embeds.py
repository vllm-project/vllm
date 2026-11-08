# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that M-RoPE position initialization handles prompt_embeds-only inputs.

Regression test for GHSA-33cg-gxv8-3p8g: sending /v1/completions with
prompt_embeds and no prompt_token_ids on M-RoPE models crashed the
EngineCore via an assertion failure.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.model_executor.models.interfaces import SupportsMRoPE
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.request import Request
from vllm.v1.worker.gpu.mm.rope import RopeState
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

NUM_DIMS = 3


class FakeMRoPEModel(SupportsMRoPE):
    """Minimal model that passes supports_mrope() check."""

    def get_mrope_input_positions(self, input_tokens, mm_features):
        seq_len = len(input_tokens)
        positions = torch.arange(seq_len).unsqueeze(0).expand(NUM_DIMS, -1)
        return positions.clone(), 0


def _add_request(
    prompt_token_ids: list[int] | None, prompt_embeds: torch.Tensor | None
) -> Mock:
    """Schedule a new request the way the scheduler does and add it to a
    model state with M-RoPE; return the staged prefill positions mock."""
    request = Request(
        request_id="req",
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        prompt_embeds=prompt_embeds,
    )
    new_req_data = NewRequestData.from_request(
        request, ([0],), request.prefill_token_ids, uses_mrope=True
    )

    rope_state = object.__new__(RopeState)
    rope_state.num_dims = NUM_DIMS
    rope_state.prefill_positions = Mock()
    rope_state.prefill_delta = SimpleNamespace(np=np.zeros(1, dtype=np.int32))

    model_state = object.__new__(DefaultModelState)
    model_state.model = FakeMRoPEModel()
    model_state.rope_state = rope_state
    model_state.prompt_embeds_state = None
    model_state.add_request(0, new_req_data)
    return rope_state.prefill_positions


@pytest.mark.parametrize(
    ("prompt_token_ids", "prompt_embeds", "num_tokens"),
    [
        pytest.param(None, torch.randn(15, 896), 15, id="prompt-embeds-only"),
        pytest.param([1, 2, 3, 4, 5], None, 5, id="token-ids"),
    ],
)
def test_mrope_positions_initialized(prompt_token_ids, prompt_embeds, num_tokens):
    prefill_positions = _add_request(prompt_token_ids, prompt_embeds)

    calls = prefill_positions.stage_write.call_args_list
    assert len(calls) == NUM_DIMS
    for dim, call in enumerate(calls):
        row, start, positions = call.args
        assert (row, start) == (dim, 0)
        np.testing.assert_array_equal(positions, np.arange(num_tokens))
