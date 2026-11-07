# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.prompt_logprob import (
    PromptLogprobsWorker,
    compute_prompt_logprobs_with_chunking,
)


def test_prompt_logprob_chunks_are_bounded_by_max_num_reqs():
    """profile_run computes one logits row per request, so prompt logprobs
    must not materialize more than max_num_reqs rows of logits at once."""
    max_num_reqs, prompt_len, vocab_size, hidden_size = 4, 10, 64, 8
    device = torch.device(current_platform.device_type)
    torch.manual_seed(0)
    # Integer-valued so the logits are exact for any chunking.
    weight = torch.randint(-3, 4, (hidden_size, vocab_size), device=device).float()
    hidden_states = torch.randint(-3, 4, (prompt_len, hidden_size), device=device)
    hidden_states = hidden_states.float()
    prompt = torch.randint(0, vocab_size, (prompt_len,), device=device)
    rows_seen: list[int] = []

    def compute_logits(h: torch.Tensor) -> torch.Tensor:
        rows_seen.append(h.shape[0])
        return h @ weight

    input_buffers = InputBuffers(max_num_reqs, prompt_len, device)
    input_batch = InputBatch.make_dummy(1, prompt_len, input_buffers)
    input_batch.prefill_len_np[0] = prompt_len
    req_id = input_batch.req_ids[0]
    worker = PromptLogprobsWorker(max_num_reqs, device)
    worker.add_request(req_id, 0, SamplingParams(prompt_logprobs=2))
    all_token_ids = torch.zeros(
        max_num_reqs, prompt_len + 1, dtype=torch.int32, device=device
    )
    all_token_ids[0, :prompt_len] = prompt

    out = worker.compute_prompt_logprobs(
        compute_logits,
        hidden_states,
        input_batch,
        all_token_ids,
        torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
        np.array([prompt_len, 0, 0, 0]),
    )

    assert max(rows_seen) <= max_num_reqs
    assert sum(rows_seen) == prompt_len
    _, reference, _ = compute_prompt_logprobs_with_chunking(
        prompt[1:], hidden_states[: prompt_len - 1], lambda h: h @ weight, 2
    )
    assert torch.equal(out[req_id].logprobs, reference)
