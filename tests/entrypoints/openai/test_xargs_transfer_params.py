# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject reserved transfer keys planted through vllm_xargs."""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError


@pytest.mark.parametrize("bad_value", ["x", 1, 1.5, ["x"], True])
@pytest.mark.parametrize("reserved", ["kv_transfer_params", "ec_transfer_params"])
def test_completion_rejects_reserved_xargs_key(reserved, bad_value):
    request = CompletionRequest(
        model="test-model",
        prompt="hi",
        max_tokens=1,
        vllm_xargs={reserved: bad_value},
    )
    with pytest.raises(VLLMValidationError, match="top-level field"):
        request.to_sampling_params(max_tokens=1, default_sampling_params={})


@pytest.mark.parametrize("bad_value", ["x", 1, ["x"]])
def test_chat_completion_rejects_reserved_xargs_key(bad_value):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
        vllm_xargs={"kv_transfer_params": bad_value},
    )
    with pytest.raises(VLLMValidationError, match="top-level field"):
        request.to_sampling_params(max_tokens=1, default_sampling_params={})


def test_responses_rejects_reserved_xargs_key():
    request = ResponsesRequest(
        model="test-model",
        input="hi",
        vllm_xargs={"kv_transfer_params": "x"},
    )
    with pytest.raises(VLLMValidationError, match="top-level field"):
        request.to_sampling_params(default_max_tokens=16)


def test_completion_overlays_typed_kv_transfer_params():
    kv = {"do_remote_prefill": True}
    request = CompletionRequest(
        model="test-model",
        prompt="hi",
        max_tokens=1,
        vllm_xargs={"custom": 7, "kv_cache_report_mode": "full"},
        kv_transfer_params=kv,
    )
    params = request.to_sampling_params(max_tokens=1, default_sampling_params={})
    assert params.extra_args == {
        "custom": 7,
        "kv_cache_report_mode": "full",
        "kv_transfer_params": kv,
    }


def test_completion_does_not_mutate_request_xargs():
    xargs = {"custom": 7}
    request = CompletionRequest(
        model="test-model",
        prompt="hi",
        max_tokens=1,
        vllm_xargs=xargs,
        kv_transfer_params={"do_remote_decode": True},
    )
    request.to_sampling_params(max_tokens=1, default_sampling_params={})
    assert request.vllm_xargs == {"custom": 7}
    assert "kv_transfer_params" not in xargs
