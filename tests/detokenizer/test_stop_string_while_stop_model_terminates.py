# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer


@pytest.fixture(params=[True, False])
def include_stop_str_in_output(request):
    return request.param


class _DummyDetokenizer(BaseIncrementalDetokenizer):

    def __init__(self, request: EngineCoreRequest):
        super().__init__(request)

    def decode_next(self, next_token_id: int) -> str:
        # Map token id to single ASCII character for deterministic testing.
        return chr(next_token_id)


def _make_request(stop, include_stop_str_in_output: bool, min_tokens: int = 0):
    params = SamplingParams(
        stop=stop,
        include_stop_str_in_output=include_stop_str_in_output,
        min_tokens=min_tokens)
    # Keep other fields minimal for unit test purposes.
    req = EngineCoreRequest(
        request_id="test",
        prompt_token_ids=[],
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        eos_token_id=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    return req


def test_stop_string_while_stop_token_terminates(
        include_stop_str_in_output: bool):
    """
    This test verifies that the detokenizer correctly handles the case where
    the generated token sequence contains both:
    - a stop token
    - an <eos> token
    
    The detokenizer should respect the stop string and truncate the output
    accordingly.
    
    Imagine the following sequence:
    - "abcdeZ" is generated, where "Z" is the <eos> token.
    - "cd" is the stop string.
    
    If include_stop_str_in_output=False, the detokenizer should truncate the
    output to "ab" because the stop string "cd" is excluded.
    If include_stop_str_in_output=True, the detokenizer should include the stop
    string "cd" in the output, resulting in "abcd".
    

    This verifies the behavioral change introduced in BaseIncrementalDetokenizer
    where stop-string evaluation occurs before the early-return on
    stop_terminated.
    """

    # Generate text "abcdeZ" and tokenize it.
    generated_text = "abcde"
    eos_token = "Z"
    stop_string = "cd"
    generated_text = generated_text + eos_token
    token_ids = [ord(c) for c in generated_text]

    # Create a request with the stop string and initialize the detokenizer.
    req = _make_request(stop=[stop_string],
                        include_stop_str_in_output=include_stop_str_in_output)
    detok = _DummyDetokenizer(req)

    # Simulate that the last token ('Z') is a stop token (stop_terminated=True).
    result = detok.update(new_token_ids=token_ids, stop_terminated=True)

    # The update should not report a stop string
    assert result == stop_string

    # Output text should reflect stop-string handling:
    # - include_stop_str_in_output=False => exclude "cd" => "ab"
    # - include_stop_str_in_output=True  => include "cd" => "abcd"
    expected_text = "abcd" if include_stop_str_in_output else "ab"
    assert detok.output_text == expected_text

    # The skipped final token should still be recorded in token_ids.
    assert detok.output_token_ids == token_ids

    # get_next_output_text should return the full text when finished=True.
    # (Buffering only applies during streaming when finished=False.)
    assert detok.get_next_output_text(finished=True,
                                      delta=False) == expected_text
