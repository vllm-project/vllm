# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch
import vllm._C  # noqa: F401

from vllm.platforms import current_platform
from vllm.v1.worker.cpu.logprob import CPUPromptLogprobsWorker
from vllm.v1.worker.cpu.sampler import CPUSampler
from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker

pytestmark = pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")

if current_platform.is_cpu():
    import vllm.v1.worker.cpu.shm  # noqa: F401


@pytest.mark.parametrize("logits_mode", [False, True])
def test_cpu_output_logprobs(logits_mode: bool) -> None:
    logits = torch.tensor(
        [[4.0, 2.0, 1.0, 0.0], [0.0, 2.0, 1.0, 3.0]],
        dtype=torch.bfloat16,
    )
    sampled_token_ids = torch.tensor([2, 1], dtype=torch.int64)
    sampler = object.__new__(CPUSampler)

    output = sampler._compute_topk_scores(
        logits,
        num_logprobs=2,
        sampled_token_ids=sampled_token_ids,
        cu_num_logits=[0, 1, 2],
        logprob_token_ids_state=object.__new__(LogprobTokenIdsState),
        expanded_idx_mapping=torch.tensor([0, 1], dtype=torch.int64),
        max_per_req_token_ids=0,
        logits_mode=logits_mode,
    )

    assert output.logprob_token_ids.tolist() == [
        [2, 0, 1],
        [1, 3, 1],
    ]
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

    worker = CPUPromptLogprobsWorker(max_num_reqs=1, device=torch.device("cpu"))
    prompt_token_ids = worker.get_prompt_logprobs_token_ids(
        num_tokens=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        num_computed_tokens=torch.tensor([0], dtype=torch.int32),
        all_token_ids=torch.tensor([[7, 1, 2]], dtype=torch.int32),
    )
    token_ids, logprobs, token_ranks = worker.compute_prompt_logprobs_with_chunking(
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


def test_cpu_prompt_logprobs_scores_requested_token_ids() -> None:
    logits = torch.tensor(
        [[4.0, 2.0, 1.0, 0.0], [0.0, 2.0, 1.0, 3.0]],
        dtype=torch.bfloat16,
    )
    token_ids = torch.tensor([[1, 3], [2, 0]])

    worker = CPUPromptLogprobsWorker(max_num_reqs=1, device=torch.device("cpu"))
    actual = worker._compute_token_logprobs(logits, token_ids)
    expected = logits.gather(-1, token_ids).float()
    expected -= torch.logsumexp(logits.float(), dim=-1, keepdim=True)

    torch.testing.assert_close(actual, expected)


def test_cpu_runner_selects_prompt_logprobs_worker_without_triton() -> None:
    from vllm.v1.worker.cpu import model_runner as cpu_model_runner

    triton_runner = object.__new__(cpu_model_runner.CPUModelRunner)
    no_triton_runner = object.__new__(cpu_model_runner.NoTritonCPUModelRunner)

    assert triton_runner.prompt_logprobs_worker_cls is PromptLogprobsWorker
    assert no_triton_runner.prompt_logprobs_worker_cls is CPUPromptLogprobsWorker


def _make_logprob_token_ids_state(
    num_token_ids: list[int], token_ids: list[list[int]]
) -> LogprobTokenIdsState:
    state = object.__new__(LogprobTokenIdsState)
    state.num_token_ids = SimpleNamespace(
        gpu=torch.tensor(num_token_ids, dtype=torch.int32)
    )
    state.token_ids = SimpleNamespace(gpu=torch.tensor(token_ids, dtype=torch.int32))
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

    output = object.__new__(CPUSampler)._compute_topk_scores(
        logits,
        num_logprobs=4,
        sampled_token_ids=sampled_token_ids,
        cu_num_logits=[0, 1, 2, 3, 4],
        logprob_token_ids_state=state,
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

    output = object.__new__(CPUSampler)._compute_topk_scores(
        logits,
        num_logprobs=0,
        sampled_token_ids=torch.tensor([2, 1]),
        cu_num_logits=None,
        logprob_token_ids_state=state,
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
