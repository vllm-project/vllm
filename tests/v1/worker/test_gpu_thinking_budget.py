# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for Model Runner V2 thinking budget tests",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
START = 90
END = 91
END_A = 92
END_B = 93
VOCAB_SIZE = 128


class MockReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END]


class MockMultiTokenEndReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END_A, END_B]


def _make_req_states(tokens: list[int], prompt_len: int = 1) -> RequestState:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    req_states.add_request(
        req_id="req",
        prompt_len=prompt_len,
        all_token_ids=tokens,
        num_computed_tokens=len(tokens),
        max_tokens=32,
    )
    req_states.apply_staged_writes()
    return req_states


def _apply(
    state: ThinkingBudgetState,
    logits: torch.Tensor,
    input_ids: list[int],
    local_pos: list[int],
) -> torch.Tensor:
    expanded_idx_mapping = torch.tensor(
        [3] * len(input_ids), dtype=torch.int32, device=DEVICE
    )
    idx_mapping_np = expanded_idx_mapping.cpu().numpy()
    state.apply(
        logits,
        expanded_idx_mapping,
        idx_mapping_np,
        torch.tensor(input_ids, dtype=torch.int32, device=DEVICE),
        torch.tensor(local_pos, dtype=torch.int32, device=DEVICE),
    )
    return logits.cpu()


def test_v2_thinking_budget_forces_end_after_budget_reached():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    assert out[0, END] == pytest.approx(1.0e9)
    assert torch.isneginf(out[0, :END]).all()
    assert torch.isneginf(out[0, END + 1 :]).all()


def test_v2_thinking_budget_allows_tokens_before_budget():
    req_states = _make_req_states([1, START, 10, 11], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[11], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_thinking_budget_continues_multi_token_end_marker():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockMultiTokenEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((2, VOCAB_SIZE), device=DEVICE)
    out = _apply(
        state,
        logits,
        input_ids=[12, END_A],
        local_pos=[0, 1],
    )

    assert out[0, END_A] == pytest.approx(1.0e9)
    assert out[1, END_B] == pytest.approx(1.0e9)


def test_v2_thinking_budget_ignores_plain_request():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams())
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    assert torch.all(out == 0)
