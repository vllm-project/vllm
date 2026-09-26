# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

KV_TRANSFER_PARAMS = {"request_id": "kv-request"}
EC_TRANSFER_PARAMS = {"request_id": "ec-request"}


@pytest.mark.parametrize(
    ("request_cls", "request_kwargs", "sampling_kwargs"),
    [
        (
            ChatCompletionRequest,
            {"messages": [{"role": "user", "content": "hello"}]},
            {"max_tokens": 8, "default_sampling_params": {}},
        ),
        (
            CompletionRequest,
            {"prompt": "hello"},
            {"max_tokens": 8},
        ),
        (
            ResponsesRequest,
            {"input": "hello"},
            {"default_max_tokens": 8},
        ),
    ],
)
def test_to_sampling_params_does_not_mutate_request_extra_args(
    request_cls: type[ChatCompletionRequest | CompletionRequest | ResponsesRequest],
    request_kwargs: dict[str, Any],
    sampling_kwargs: dict[str, Any],
) -> None:
    original_xargs = {"custom_key": "custom_value"}
    request = request_cls(
        model="test-model",
        vllm_xargs=original_xargs.copy(),
        kv_transfer_params=KV_TRANSFER_PARAMS,
        ec_transfer_params=EC_TRANSFER_PARAMS,
        **request_kwargs,
    )
    expected_extra_args = {
        **original_xargs,
        "kv_transfer_params": KV_TRANSFER_PARAMS,
        "ec_transfer_params": EC_TRANSFER_PARAMS,
    }

    assert request.vllm_xargs == original_xargs
    first_params = request.to_sampling_params(**sampling_kwargs)
    second_params = request.to_sampling_params(**sampling_kwargs)

    assert request.vllm_xargs == original_xargs
    assert first_params.extra_args == expected_extra_args
    assert second_params.extra_args == expected_extra_args
    assert first_params.extra_args is not request.vllm_xargs
    assert second_params.extra_args is not request.vllm_xargs
    assert first_params.extra_args is not second_params.extra_args
    request_cls.model_validate(request.model_dump())
