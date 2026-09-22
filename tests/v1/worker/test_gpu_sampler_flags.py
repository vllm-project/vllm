# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampler flag tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)
from vllm.v1.worker.gpu.sample.no_repeat_ngram import NoRepeatNGramState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


def _make_sampler(
    custom_logits_processors: Sequence[LogitsProcessor] | None = None,
) -> Sampler:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    vllm_config = SimpleNamespace(
        reasoning_config=MockReasoningConfig(), speculative_config=None
    )
    if custom_logits_processors is None:
        lp_req_state = LogitsProcRequestState.from_request_state(req_states)
        custom_logits_processors = [NoRepeatNGramState(vllm_config, lp_req_state)]
    return Sampler(
        vllm_config=vllm_config,
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        custom_logits_processors=custom_logits_processors,
    )


@pytest.mark.parametrize(
    ("sampling_params", "expected"),
    [
        pytest.param(SamplingParams(), False, id="defaults"),
        pytest.param(SamplingParams(temperature=0.0), False, id="greedy"),
        pytest.param(
            SamplingParams(thinking_token_budget=3), True, id="thinking-budget"
        ),
        pytest.param(SamplingParams(logit_bias={1: 1.0}), True, id="logit-bias"),
        pytest.param(SamplingParams(frequency_penalty=0.1), True, id="penalty"),
        pytest.param(SamplingParams(_bad_words_token_ids=[[1]]), True, id="bad-words"),
        pytest.param(
            SamplingParams(extra_args={"no_repeat_ngram_size": 3}),
            True,
            id="no-repeat-ngram",
        ),
        pytest.param(
            SamplingParams(extra_args={"no_repeat_ngram_size": 1}),
            True,
            id="canonical-unigram",
        ),
        pytest.param(
            SamplingParams(extra_args={"ngram_size": 1}),
            False,
            id="legacy-unigram-noop",
        ),
        pytest.param(SamplingParams(temperature=0.7), True, id="temperature"),
        pytest.param(SamplingParams(min_p=0.1), True, id="min-p"),
        pytest.param(SamplingParams(top_k=10), True, id="top-k"),
        pytest.param(SamplingParams(top_p=0.9), True, id="top-p"),
        pytest.param(
            SamplingParams.for_sampler_warmup(), True, id="all-logits-processors"
        ),
    ],
)
def test_logits_processing_cache_matches_request_features(
    sampling_params: SamplingParams, expected: bool
):
    sampler = _make_sampler()
    sampler.add_request(3, sampling_params=sampling_params)

    assert sampler.needs_logits_processing[3] == expected


def test_logits_processing_cache_is_overwritten_when_slot_is_reused():
    sampler = _make_sampler()
    sampler.add_request(3, SamplingParams.for_sampler_warmup())
    sampler.add_request(3, SamplingParams())

    assert not sampler.needs_logits_processing[3]


def test_no_repeat_ngram_state_is_cleared_when_slot_is_reused():
    sampler = _make_sampler()
    sampler.add_request(3, SamplingParams(extra_args={"no_repeat_ngram_size": 3}))
    sampler.add_request(3, SamplingParams())

    assert not sampler.needs_logits_processing[3]


def test_no_repeat_ngram_rejects_speculative_decoding_at_initialization():
    req_states = RequestState(
        max_num_reqs=1,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    vllm_config = SimpleNamespace(speculative_config=object())
    lp_req_state = LogitsProcRequestState.from_request_state(req_states)

    with pytest.raises(ValueError, match="does not support speculative decoding"):
        NoRepeatNGramState(vllm_config, lp_req_state)


class _GateProcessor(LogitsProcessor):
    """Reports a fixed admission decision from add_request()."""

    def __init__(self, admitted: bool):
        self.admitted = admitted

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        return self.admitted

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        raise AssertionError("the pipeline never runs in this test")


@pytest.mark.parametrize("admitted", [True, False])
def test_custom_processor_add_request_gates_pipeline_flag(admitted: bool):
    """A custom processor's add_request() return value is OR-ed into the
    per-request needs_logits_processing flag."""
    sampler = _make_sampler(custom_logits_processors=[_GateProcessor(admitted)])
    sampler.add_request(3, SamplingParams())

    assert sampler.needs_logits_processing[3] == admitted


def test_logits_processing_cache_only_checks_active_requests():
    sampler = _make_sampler()
    sampler.add_request(0, SamplingParams(temperature=0.0))
    sampler.add_request(2, SamplingParams.for_sampler_warmup())

    sampling_only = np.array([0], dtype=np.int32)
    with_processing = np.array([0, 2], dtype=np.int32)

    assert not np.any(sampler.needs_logits_processing[sampling_only])
    assert np.any(sampler.needs_logits_processing[with_processing])
