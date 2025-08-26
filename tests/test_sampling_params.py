# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SamplingParams class.
"""

import pytest

from vllm import SamplingParams
from vllm.config import ModelConfig
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

MODEL_NAME = "Qwen/Qwen1.5-7B"


def test_max_tokens_none():
    """max_tokens=None should be allowed"""
    SamplingParams(temperature=0.01, top_p=0.1, max_tokens=None)


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(
        MODEL_NAME,
        seed=0,
        dtype="float16",
    )


@pytest.fixture(scope="module")
def default_max_tokens():
    return 4096


def test_sampling_params_from_request_with_no_guided_decoding_backend(
        model_config, default_max_tokens):
    # guided_decoding_backend is not present at request level
    request = ChatCompletionRequest.model_validate({
        'messages': [{
            'role': 'user',
            'content': 'Hello'
        }],
        'model':
        MODEL_NAME,
        'response_format': {
            'type': 'json_object',
        },
    })

    sampling_params = request.to_sampling_params(
        default_max_tokens,
        model_config.logits_processor_pattern,
    )
    # we do not expect any backend to be present and the default
    # guided_decoding_backend at engine level will be used.
    assert sampling_params.guided_decoding.backend is None


@pytest.mark.parametrize("request_level_guided_decoding_backend,expected",
                         [("xgrammar", "xgrammar"), ("guidance", "guidance"),
                          ("outlines", "outlines")])
def test_sampling_params_from_request_with_guided_decoding_backend(
        request_level_guided_decoding_backend: str, expected: str,
        model_config, default_max_tokens):

    request = ChatCompletionRequest.model_validate({
        'messages': [{
            'role': 'user',
            'content': 'Hello'
        }],
        'model':
        MODEL_NAME,
        'response_format': {
            'type': 'json_object',
        },
        'guided_decoding_backend':
        request_level_guided_decoding_backend,
    })

    sampling_params = request.to_sampling_params(
        default_max_tokens,
        model_config.logits_processor_pattern,
    )
    # backend correctly identified in resulting sampling_params
    assert sampling_params.guided_decoding.backend == expected
