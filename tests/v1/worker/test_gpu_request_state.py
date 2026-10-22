# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.states import RequestState

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="requires CUDA",
)

MAX_NUM_REQS = 8
NUM_SPECULATIVE_STEPS = 3


def _make_request_state(num_speculative_steps: int) -> RequestState:
    return RequestState(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=32,
        max_num_batched_tokens=64,
        num_speculative_steps=num_speculative_steps,
        vocab_size=1024,
        device=torch.device("cuda"),
        use_dense_all_token_ids=True,
    )


def _add_request(req_states: RequestState, req_id: str) -> int:
    """Add `req_id` through the public API and return its slot index."""
    req_states.add_request(
        req_id=req_id,
        prompt_len=4,
        all_token_ids=[1, 2, 3, 4],
        num_computed_tokens=0,
        max_tokens=8,
    )
    return req_states.req_id_to_index[req_id]


def test_draft_tokens_reset_is_batched_on_apply_staged_writes():
    req_states = _make_request_state(NUM_SPECULATIVE_STEPS)
    req_states.draft_tokens.fill_(7)

    added_indices = [_add_request(req_states, f"req-{i}") for i in range(3)]

    # The reset is deferred until the staged writes are applied.
    assert torch.all(req_states.draft_tokens == 7)

    req_states.apply_staged_writes()

    expected = torch.full_like(req_states.draft_tokens, 7)
    expected[added_indices] = 0
    assert torch.equal(req_states.draft_tokens, expected)
    assert not req_states.new_req_indices


def test_draft_tokens_reset_for_slot_reused_in_same_step():
    req_states = _make_request_state(NUM_SPECULATIVE_STEPS)
    req_states.draft_tokens.fill_(7)

    req_idx = _add_request(req_states, "req-0")
    assert req_states.remove_request("req-0") == req_idx
    assert _add_request(req_states, "req-1") == req_idx

    req_states.apply_staged_writes()

    expected = torch.full_like(req_states.draft_tokens, 7)
    expected[req_idx] = 0
    assert torch.equal(req_states.draft_tokens, expected)
    assert not req_states.new_req_indices


def test_no_draft_tokens_reset_without_spec_decode():
    req_states = _make_request_state(0)
    before = req_states.draft_tokens.clone()

    req_idx = _add_request(req_states, "req-0")
    req_states.apply_staged_writes()

    assert req_states.req_id_to_index["req-0"] == req_idx
    assert torch.equal(req_states.draft_tokens, before)
