# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression: echo must still prepend prompt text when return_token_ids is set.

See https://github.com/vllm-project/vllm/issues/57996 — previously
`needs_detokenization=bool(echo and not return_token_ids)` skipped prompt
detokenization, and completion serving wiped `prompt_text` whenever
`return_token_ids` was set (#24405), so `echo` + `return_token_ids` returned
completion-only text for string prompts.
"""

from unittest.mock import Mock

import pytest

from vllm.config import ModelConfig
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.outputs import CompletionOutput, RequestOutput

pytestmark = pytest.mark.skip_global_cleanup


def _model_config() -> Mock:
    model_config = Mock(spec=ModelConfig)
    model_config.max_model_len = 128
    return model_config


def _minimal_serving() -> OpenAIServingCompletion:
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.enable_prompt_tokens_details = False
    serving.system_fingerprint = None
    serving.enable_per_request_metrics = False
    return serving


def test_completion_echo_still_detokenizes_with_return_token_ids():
    cfg = _model_config()

    both = CompletionRequest(model="m", prompt="hi", echo=True, return_token_ids=True)
    assert both.build_tok_params(cfg).needs_detokenization is True

    echo_only = CompletionRequest(
        model="m", prompt="hi", echo=True, return_token_ids=False
    )
    assert echo_only.build_tok_params(cfg).needs_detokenization is True

    ids_only = CompletionRequest(
        model="m", prompt="hi", echo=False, return_token_ids=True
    )
    assert ids_only.build_tok_params(cfg).needs_detokenization is False

    neither = CompletionRequest(
        model="m", prompt="hi", echo=False, return_token_ids=False
    )
    assert neither.build_tok_params(cfg).needs_detokenization is False


def test_chat_echo_still_detokenizes_with_return_token_ids():
    cfg = _model_config()
    messages = [{"role": "user", "content": "hi"}]

    both = ChatCompletionRequest(
        model="m", messages=messages, echo=True, return_token_ids=True
    )
    assert both.build_tok_params(cfg).needs_detokenization is True

    echo_only = ChatCompletionRequest(
        model="m", messages=messages, echo=True, return_token_ids=False
    )
    assert echo_only.build_tok_params(cfg).needs_detokenization is True


def test_echo_with_return_token_ids_response_text_starts_with_prompt():
    """String-prompt repro: response text must start with the echoed prompt."""
    prompt = "The capital of France is"
    completion = " Paris"
    serving = _minimal_serving()
    request = CompletionRequest(
        model="m",
        prompt=prompt,
        max_tokens=6,
        echo=True,
        return_token_ids=True,
    )
    request_output = RequestOutput(
        request_id="test-id",
        prompt=prompt,
        prompt_token_ids=[1, 2, 3, 4, 5],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=completion,
                token_ids=[100],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="length",
            )
        ],
        finished=True,
    )

    response = serving.request_output_to_completion_response(
        [request_output],
        request,
        "cmpl-test-id",
        0,
        "m",
        None,
        RequestResponseMetadata(request_id="cmpl-test-id"),
    )

    choice = response.choices[0]
    assert choice.text.startswith(prompt)
    assert choice.text == prompt + completion
    assert choice.prompt_token_ids == [1, 2, 3, 4, 5]
    assert choice.token_ids == [100]
