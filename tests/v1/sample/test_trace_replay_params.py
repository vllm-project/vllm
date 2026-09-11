# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import StructuredOutputsParams

pytestmark = pytest.mark.skip_global_cleanup

# ---------------------------------------------------------------------------
# SamplingParams field
# ---------------------------------------------------------------------------


def test_sampling_params_trace_field_defaults_to_none():
    params = SamplingParams(max_tokens=10)
    assert params.trace_decode_token_ids is None


def test_sampling_params_trace_field_accepts_list():
    ids = [100, 200, 300]
    params = SamplingParams(trace_decode_token_ids=ids)
    assert params.trace_decode_token_ids == ids


def test_sampling_params_trace_field_preserved_by_clone():
    ids = [1, 2, 3]
    params = SamplingParams(trace_decode_token_ids=ids)
    cloned = params.clone()
    assert cloned.trace_decode_token_ids == ids
    assert cloned.trace_decode_token_ids is not params.trace_decode_token_ids


def test_sampling_params_trace_field_rejects_empty_list():
    params = SamplingParams(trace_decode_token_ids=[])
    with pytest.raises(ValueError, match="non-empty"):
        params._validate_trace_replay(
            _make_model_config(vocab_size=100), speculative_config=None
        )


def test_sampling_params_trace_field_requires_single_output():
    params = SamplingParams(n=2, trace_decode_token_ids=[1])
    with pytest.raises(ValueError, match="requires n=1"):
        params._validate_trace_replay(
            _make_model_config(vocab_size=100), speculative_config=None
        )


@pytest.mark.parametrize("invalid_ids", [[-1, 5], [1, "2"]])
def test_sampling_params_trace_field_rejects_invalid_token_ids(invalid_ids):
    params = SamplingParams(trace_decode_token_ids=invalid_ids)
    with pytest.raises(ValueError, match="non-negative integers"):
        params._validate_trace_replay(
            _make_model_config(vocab_size=100), speculative_config=None
        )


def _make_model_config(vocab_size: int):
    from unittest.mock import Mock

    model_config = Mock()
    model_config.get_vocab_size = lambda: vocab_size
    return model_config


def test_validate_trace_replay_accepts_in_vocab():
    params = SamplingParams(trace_decode_token_ids=[0, 50, 99])
    # Should not raise.
    params._validate_trace_replay(
        _make_model_config(vocab_size=100), speculative_config=None
    )


def test_validate_trace_replay_rejects_out_of_vocab():
    # The non-negative check passes at construction, but the token id exceeds
    # the vocabulary; verify() must reject it before it reaches the sampler.
    params = SamplingParams(trace_decode_token_ids=[0, 100])
    with pytest.raises(VLLMValidationError, match="out-of-vocab"):
        params._validate_trace_replay(
            _make_model_config(vocab_size=100), speculative_config=None
        )


def test_validate_trace_replay_noop_when_unset():
    params = SamplingParams(max_tokens=4)
    # Should not raise when the field is unset.
    params._validate_trace_replay(
        _make_model_config(vocab_size=100), speculative_config=None
    )


def test_trace_decode_token_ids_rejects_speculative_decoding():
    params = SamplingParams(trace_decode_token_ids=[1])
    with pytest.raises(ValueError, match="not supported with speculative decoding"):
        params._validate_trace_replay(
            _make_model_config(vocab_size=100), speculative_config=object()
        )


def test_trace_decode_token_ids_rejects_structured_outputs():
    params = SamplingParams(
        trace_decode_token_ids=[1],
        structured_outputs=StructuredOutputsParams(json_object=True),
    )
    with pytest.raises(ValueError, match="not supported with structured outputs"):
        params._validate_trace_replay(
            _make_model_config(vocab_size=100), speculative_config=None
        )
