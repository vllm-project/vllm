# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from tests.v1.core.utils import create_scheduler
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request


def _make_request(prompt_len: int) -> Request:
    prompt_token_ids = list(range(prompt_len))
    sampling_params = SamplingParams(max_tokens=1)
    return Request(
        request_id="req",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
    )


def test_mamba_split_zero_collapse_guard():
    scheduler = create_scheduler(block_size=128, skip_tokenizer_init=True)
    request = _make_request(prompt_len=256)
    request.num_computed_tokens = 0
    aligned = scheduler._mamba_block_aligned_split(request, 2)
    assert aligned > 0


def test_mamba_split_block_alignment():
    scheduler = create_scheduler(block_size=16, skip_tokenizer_init=True)
    request = _make_request(prompt_len=64)
    request.num_computed_tokens = 0
    aligned = scheduler._mamba_block_aligned_split(request, 40)
    assert aligned == 32
    assert aligned % 16 == 0


def test_mamba_split_cache_boundary_cut():
    scheduler = create_scheduler(block_size=16, skip_tokenizer_init=True)
    request = _make_request(prompt_len=34)
    request.num_computed_tokens = 20
    aligned = scheduler._mamba_block_aligned_split(request, 20)
    assert aligned == 12
