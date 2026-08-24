# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampler flag tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


def _make_sampler() -> Sampler:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        reasoning_config=MockReasoningConfig(),
    )


@pytest.mark.parametrize(
    ("sampling_params", "needs_processing", "needs_stored_processing"),
    [
        pytest.param(SamplingParams(), False, False, id="defaults"),
        pytest.param(SamplingParams(temperature=0.0), False, False, id="greedy"),
        pytest.param(
            SamplingParams(thinking_token_budget=3), True, True, id="thinking-budget"
        ),
        pytest.param(SamplingParams(logit_bias={1: 1.0}), True, True, id="logit-bias"),
        pytest.param(SamplingParams(frequency_penalty=0.1), True, True, id="penalty"),
        pytest.param(
            SamplingParams(_bad_words_token_ids=[[1]]), True, True, id="bad-words"
        ),
        pytest.param(SamplingParams(temperature=0.7), True, False, id="temperature"),
        pytest.param(SamplingParams(min_p=0.1), True, False, id="min-p"),
        pytest.param(SamplingParams(top_k=10), True, True, id="top-k"),
        pytest.param(SamplingParams(top_p=0.9), True, True, id="top-p"),
        pytest.param(
            SamplingParams.for_sampler_warmup(),
            True,
            True,
            id="all-logits-processors",
        ),
    ],
)
def test_logits_processing_cache_matches_request_features(
    sampling_params: SamplingParams,
    needs_processing: bool,
    needs_stored_processing: bool,
):
    sampler = _make_sampler()
    sampler.add_request(3, prompt_len=1, sampling_params=sampling_params)

    assert sampler.needs_logits_processing[3] == needs_processing
    assert sampler.needs_stored_logits_processing[3] == needs_stored_processing


def test_logits_processing_cache_is_overwritten_when_slot_is_reused():
    sampler = _make_sampler()
    sampler.add_request(3, 1, SamplingParams.for_sampler_warmup())
    sampler.add_request(3, 1, SamplingParams())

    assert not sampler.needs_logits_processing[3]
    assert not sampler.needs_stored_logits_processing[3]


def test_logits_processing_cache_only_checks_active_requests():
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0))
    sampler.add_request(2, 1, SamplingParams.for_sampler_warmup())

    sampling_only = np.array([0], dtype=np.int32)
    with_processing = np.array([0, 2], dtype=np.int32)

    assert not np.any(sampler.needs_logits_processing[sampling_only])
    assert np.any(sampler.needs_logits_processing[with_processing])


@pytest.mark.parametrize(
    "sampling_params",
    [
        SamplingParams(temperature=0.7),
        SamplingParams(temperature=0.7, top_p=0.9),
        SamplingParams(frequency_penalty=0.1),
    ],
    ids=["temperature-only", "top-p", "penalty"],
)
def test_head_dtype_processing_reuses_input(sampling_params: SamplingParams):
    sampler = _make_sampler()
    sampler.add_request(0, 1, sampling_params)
    sampler.apply_staged_writes()
    logits = torch.randn(1, VOCAB_SIZE, device=DEVICE, dtype=torch.bfloat16)
    indices = torch.zeros(1, device=DEVICE, dtype=torch.int32)

    result = sampler.apply_sampling_params(
        logits,
        indices,
        indices,
        np.zeros(1, dtype=np.int32),
        indices,
        indices,
        indices,
        use_head_dtype=True,
    )

    assert result.data_ptr() == logits.data_ptr()
    assert result.dtype == torch.bfloat16
