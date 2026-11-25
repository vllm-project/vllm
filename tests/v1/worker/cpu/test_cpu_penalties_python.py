# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.cpu.logits_processors import (
    CPUPenalties,
    LogitsContext,
    LogitsProcRequestState,
)


def _penalty_state(token_ids: list[list[int]], prompt_lens: list[int]):
    batch = len(token_ids)
    width = len(token_ids[0])
    prompt_len = torch.tensor(prompt_lens, dtype=torch.int32)
    req_states = LogitsProcRequestState(
        device=torch.device("cpu"),
        max_num_reqs=batch,
        vocab_size=16,
        all_token_ids=torch.tensor(token_ids, dtype=torch.int32),
        prompt_len=prompt_len,
        prompt_len_np=prompt_len.numpy(),
        prefill_len=prompt_len,
        prefill_len_np=prompt_len.numpy(),
        total_len=torch.full((batch,), width, dtype=torch.int32),
    )
    return CPUPenalties(None, req_states)


def _context(req_indices: list[int], input_ids: list[int]) -> LogitsContext:
    rows = len(req_indices)
    mapping = torch.tensor(req_indices, dtype=torch.int64)
    return LogitsContext(
        expanded_idx_mapping=mapping,
        idx_mapping=mapping,
        idx_mapping_np=mapping.numpy(),
        expanded_local_pos=torch.zeros(rows, dtype=torch.int32),
        input_ids=torch.tensor(input_ids, dtype=torch.int32),
        pos=torch.zeros(rows, dtype=torch.int32),
        seq_lens_upper_bound_np=np.ones(rows, dtype=np.int64),
    )


def test_frequency_presence_path_ignores_prompt_and_counts_output_repeats():
    # Invalid prompt IDs also verify this path does not need repetition's prompt scan.
    state = _penalty_state([[1, -1, 16, 2, 2]], [3])
    state.add_request(0, SamplingParams(frequency_penalty=0.5, presence_penalty=0.25))
    logits = torch.zeros((1, 16))
    logits[0, 1] = 2.0
    logits[0, 2] = 3.0
    logits[0, 3] = 4.0

    actual = state.apply(logits, _context([0], [3]))

    assert actual[0, 1].item() == 2.0
    assert actual[0, 2].item() == 1.75
    assert actual[0, 3].item() == 4.0


def test_mixed_penalty_rows_preserve_reference_behavior():
    state = _penalty_state([[1, 2, 2], [3, 4, 4]], [1, 1])
    state.add_request(0, SamplingParams(repetition_penalty=2.0))
    state.add_request(1, SamplingParams(frequency_penalty=0.5, presence_penalty=0.25))
    logits = torch.ones((2, 16))

    actual = state.apply(logits, _context([0, 1], [2, 4]))

    assert actual[0, 1].item() == 0.5
    assert actual[0, 2].item() == 0.5
    assert actual[1, 3].item() == 1.0
    assert actual[1, 4].item() == -0.25
