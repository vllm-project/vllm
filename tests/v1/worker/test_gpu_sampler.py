# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler


def _new_rapid_sampler(
    *,
    rapid_penalties: torch.Tensor,
    all_token_ids: torch.Tensor | None = None,
) -> Sampler:
    sampler = object.__new__(Sampler)
    sampler.rapid_penalties = rapid_penalties
    sampler.req_states = SimpleNamespace(
        max_num_reqs=rapid_penalties.shape[0],
        vocab_size=rapid_penalties.shape[1],
        device=rapid_penalties.device,
        all_token_ids=SimpleNamespace(gpu=all_token_ids),
    )
    return sampler


def test_worker_sampler_clears_rapid_penalty_row_for_reused_slot():
    rapid_penalties = torch.ones(2, 8, dtype=torch.float32)
    sampler = _new_rapid_sampler(rapid_penalties=rapid_penalties)

    sampler._reset_rapid_penalty_row(1)

    assert torch.count_nonzero(rapid_penalties[1]) == 0
    assert torch.all(rapid_penalties[0] == 1)


def test_worker_sampler_seeds_rapid_repetition_penalty_from_prompt():
    all_token_ids = torch.tensor(
        [
            [2, 5, 2, 99],
            [0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    rapid_penalties = torch.zeros(2, 8, dtype=torch.float32)
    sampler = _new_rapid_sampler(
        rapid_penalties=rapid_penalties,
        all_token_ids=all_token_ids,
    )

    sampler._seed_rapid_prompt_penalties(
        req_idx=0,
        prompt_len=4,
        sampling_params=SamplingParams(repetition_penalty=1.2),
    )

    assert torch.isclose(rapid_penalties[0, 2], torch.tensor(1.2))
    assert torch.isclose(rapid_penalties[0, 5], torch.tensor(1.2))
    assert torch.count_nonzero(rapid_penalties[0]) == 2
