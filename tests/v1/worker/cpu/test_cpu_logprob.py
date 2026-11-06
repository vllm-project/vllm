# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np
import pytest
import torch

from vllm.platforms import CpuArchEnum, current_platform
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.cpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.cpu.logprob import (
    CPUPromptLogprobsWorker,
    LogprobTokenIdsState,
    compute_prompt_logprobs_with_chunking,
    compute_token_logprobs,
    compute_topk_scores,
    get_prompt_logprobs_token_ids,
)

if current_platform.is_cpu():
    import vllm._C  # noqa: F401

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)


@pytest.mark.parametrize("logits_mode", [False, True])
def test_cpu_output_logprobs(logits_mode: bool) -> None:
    logits = torch.tensor(
        [[4.0, 2.0, 1.0, 0.0], [0.0, 2.0, 1.0, 3.0]],
        dtype=torch.bfloat16,
    )
    sampled_token_ids = torch.tensor([2, 1], dtype=torch.int64)

    output = compute_topk_scores(
        logits,
        num_logprobs=2,
        sampled_token_ids=sampled_token_ids,
        cu_num_logits=[0, 1, 2],
        max_per_req_token_ids=0,
        logits_mode=logits_mode,
    )

    assert output.logprob_token_ids.tolist() == [
        [2, 0, 1],
        [1, 3, 1],
    ]
    assert output.logprob_token_ids.dtype == torch.int64
    expected_scores = torch.tensor([[1.0, 4.0, 2.0], [2.0, 3.0, 2.0]])
    if not logits_mode:
        expected_scores -= torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    torch.testing.assert_close(output.logprobs, expected_scores)
    assert output.selected_token_ranks.tolist() == [3, 2]
    assert output.cu_num_generated_tokens == [0, 1, 2]


def test_cpu_prompt_logprobs_scores_prompt_targets() -> None:
    logits = torch.tensor(
        [[4.0, 2.0, 1.0, 0.0], [0.0, 2.0, 1.0, 3.0]],
        dtype=torch.bfloat16,
    )

    prompt_token_ids = get_prompt_logprobs_token_ids(
        num_tokens=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        num_computed_tokens=torch.tensor([0], dtype=torch.int32),
        all_token_ids=torch.tensor([[7, 1, 2]], dtype=torch.int32),
    )
    token_ids, logprobs, token_ranks = compute_prompt_logprobs_with_chunking(
        prompt_token_ids=prompt_token_ids,
        prompt_hidden_states=logits,
        logits_fn=lambda hidden_states: hidden_states,
        num_prompt_logprobs=2,
    )

    assert token_ids.tolist() == [[1, 0, 1], [2, 3, 1]]
    expected = torch.tensor([[2.0, 4.0, 2.0], [1.0, 3.0, 2.0]])
    expected -= torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    torch.testing.assert_close(logprobs, expected)
    assert token_ranks.tolist() == [2, 3]


def _input_batch(
    req_id: str,
    num_computed_tokens: int,
    num_scheduled_tokens: int,
    prefill_len: int,
) -> InputBatch:
    buffers = InputBuffers(1, num_scheduled_tokens, torch.device("cpu"))
    batch = InputBatch.make_dummy(1, num_scheduled_tokens, buffers)
    batch.req_ids = [req_id]
    batch.num_computed_prefill_tokens_np[:] = num_computed_tokens
    batch.num_scheduled_tokens[:] = num_scheduled_tokens
    batch.prefill_len_np[:] = prefill_len
    return batch


def test_cpu_prompt_logprobs_accumulate_chunked_prefill() -> None:
    worker = CPUPromptLogprobsWorker(max_num_reqs=1, device=torch.device("cpu"))
    worker.add_request("request", 0, SamplingParams(prompt_logprobs=1))

    logits = torch.tensor(
        [
            [4.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0, 3.0],
            [1.0, 4.0, 3.0, 2.0],
            [3.0, 0.0, 2.0, 1.0],
        ]
    )
    all_token_ids = torch.tensor([[0, 1, 2, 3, 0]], dtype=torch.int32)
    prompt_lens = np.array([4], dtype=np.int32)

    first = worker.compute_prompt_logprobs(
        lambda hidden_states: hidden_states,
        logits[:2],
        _input_batch("request", 0, 2, 4),
        all_token_ids,
        torch.tensor([0], dtype=torch.int32),
        prompt_lens,
    )
    assert first == {}

    second = worker.compute_prompt_logprobs(
        lambda hidden_states: hidden_states,
        logits[2:],
        _input_batch("request", 2, 2, 4),
        all_token_ids,
        torch.tensor([2], dtype=torch.int32),
        prompt_lens,
    )

    result = second["request"]
    assert result.logprob_token_ids[:, 0].tolist() == [1, 2, 3]
    expected = torch.log_softmax(logits[:3], dim=-1)[
        torch.arange(3), torch.tensor([1, 2, 3])
    ]
    torch.testing.assert_close(result.logprobs[:, 0], expected)


def test_cpu_token_logprobs_accept_int32_token_ids() -> None:
    logits = torch.tensor(
        [[4.0, 2.0, 1.0, 0.0], [0.0, 2.0, 1.0, 3.0]],
        dtype=torch.bfloat16,
    )
    token_ids = torch.tensor([[1, 3], [2, 0]], dtype=torch.int32)

    actual = compute_token_logprobs(logits, token_ids)
    expected = logits.gather(-1, token_ids).float()
    expected -= torch.logsumexp(logits.float(), dim=-1, keepdim=True)

    torch.testing.assert_close(actual, expected)


def test_cpu_prompt_logprobs_honor_start_position() -> None:
    worker = CPUPromptLogprobsWorker(max_num_reqs=1, device=torch.device("cpu"))
    worker.add_request(
        "request",
        0,
        SamplingParams(prompt_logprobs=1, prompt_logprob_start=2),
    )
    logits = torch.tensor(
        [
            [4.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0, 3.0],
            [1.0, 4.0, 3.0, 2.0],
            [3.0, 0.0, 2.0, 1.0],
        ]
    )
    all_token_ids = torch.tensor([[0, 1, 2, 3, 0]], dtype=torch.int32)

    result = worker.compute_prompt_logprobs(
        lambda hidden_states: hidden_states,
        logits,
        _input_batch("request", 0, 4, 4),
        all_token_ids,
        torch.tensor([0], dtype=torch.int32),
        np.array([4], dtype=np.int32),
    )["request"]

    assert result.logprob_token_ids[:, 0].tolist() == [3]


def test_cpu_requested_prompt_token_scores_honor_start_position() -> None:
    worker = CPUPromptLogprobsWorker(max_num_reqs=1, device=torch.device("cpu"))
    token_ids = np.array([[0, 2], [1, 3]], dtype=np.int64)
    worker.add_request(
        "request",
        0,
        SamplingParams(
            prompt_logprob_token_ids=token_ids,
            prompt_logprob_start=1,
        ),
    )
    logits = torch.tensor(
        [
            [4.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0, 3.0],
            [1.0, 4.0, 3.0, 2.0],
            [3.0, 0.0, 2.0, 1.0],
        ]
    )

    result = worker.compute_prompt_token_id_logprobs(
        lambda hidden_states: hidden_states,
        logits,
        _input_batch("request", 0, 4, 4),
        np.array([4], dtype=np.int32),
    )["request"]
    expected = compute_token_logprobs(logits[1:3], torch.from_numpy(token_ids))

    torch.testing.assert_close(result, expected)


def _make_logprob_token_ids_state(
    num_token_ids: list[int], token_ids: list[list[int]]
) -> LogprobTokenIdsState:
    state = LogprobTokenIdsState(len(num_token_ids), torch.device("cpu"))
    for req_idx, num_ids in enumerate(num_token_ids):
        state.add_request(
            req_idx,
            SamplingParams(logprob_token_ids=token_ids[req_idx][:num_ids]),
        )
    return state


@pytest.mark.parametrize("logits_mode", [False, True])
def test_cpu_requested_output_logprobs(logits_mode: bool) -> None:
    logits = torch.tensor(
        [
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [0.0, 2.0, 1.0, 5.0, 4.0, 3.0],
            [1.0, 6.0, 2.0, 5.0, 3.0, 4.0],
            [4.0, 1.0, 6.0, 0.0, 5.0, 2.0],
        ],
        dtype=torch.bfloat16,
    )
    sampled_token_ids = torch.tensor([4, 1, 0, 5], dtype=torch.int64)
    state = _make_logprob_token_ids_state(
        num_token_ids=[0, 5, 2, 3],
        token_ids=[
            [0, 0, 0, 0, 0],
            [0, 1, 2, 3, 4],
            [5, 2, 0, 0, 0],
            [4, 1, 3, 0, 0],
        ],
    )

    output = compute_topk_scores(
        logits,
        num_logprobs=4,
        sampled_token_ids=sampled_token_ids,
        cu_num_logits=[0, 1, 2, 3, 4],
        num_per_req_token_ids=state.num_token_ids,
        per_req_token_ids=state.token_ids,
        expanded_idx_mapping=torch.tensor([2, 0, 3, 1], dtype=torch.int64),
        max_per_req_token_ids=5,
        logits_mode=logits_mode,
    )

    assert output.logprob_token_ids.tolist() == [
        [4, 5, 2, 0, 0, 0],
        [1, 3, 4, 5, 1, 0],
        [0, 4, 1, 3, 0, 0],
        [5, 0, 1, 2, 3, 4],
    ]
    expected_scores = torch.tensor(
        [
            [2.0, 1.0, 4.0, -math.inf, -math.inf, -math.inf],
            [2.0, 5.0, 4.0, 3.0, 2.0, -math.inf],
            [1.0, 3.0, 6.0, 5.0, -math.inf, -math.inf],
            [2.0, 4.0, 1.0, 6.0, 0.0, 5.0],
        ]
    )
    if not logits_mode:
        expected_scores -= torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    torch.testing.assert_close(output.logprobs, expected_scores)
    assert output.selected_token_ranks.tolist() == [5, 4, 6, 4]
    assert output.cu_num_generated_tokens == [0, 1, 2, 3, 4]


def test_cpu_requested_output_logprobs_without_topk() -> None:
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    state = _make_logprob_token_ids_state(
        num_token_ids=[0, 2],
        token_ids=[[0, 0], [3, 1]],
    )

    output = compute_topk_scores(
        logits,
        num_logprobs=0,
        sampled_token_ids=torch.tensor([2, 1]),
        cu_num_logits=None,
        num_per_req_token_ids=state.num_token_ids,
        per_req_token_ids=state.token_ids,
        expanded_idx_mapping=torch.tensor([1, 0], dtype=torch.int64),
        max_per_req_token_ids=2,
        logits_mode=True,
    )

    assert output.logprob_token_ids.tolist() == [[2, 3, 1], [1, 0, 0]]
    torch.testing.assert_close(
        output.logprobs,
        torch.tensor([[2.0, 1.0, 3.0], [2.0, -math.inf, -math.inf]]),
    )
    assert output.selected_token_ranks.tolist() == [3, 3]
