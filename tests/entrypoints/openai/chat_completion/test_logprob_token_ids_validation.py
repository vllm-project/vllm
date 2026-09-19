# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError

MODELS = [ChatCompletionRequest, CompletionRequest, BatchChatCompletionRequest]
BASE = {
    ChatCompletionRequest: {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    },
    CompletionRequest: {"model": "m", "prompt": "hi"},
    BatchChatCompletionRequest: {
        "messages": [[{"role": "user", "content": "hi"}]],
    },
}


@pytest.mark.parametrize("cls", MODELS)
@pytest.mark.parametrize("logprobs", [None, False, True])
def test_rejects_empty_token_ids(cls, logprobs):
    data = {**BASE[cls], "logprob_token_ids": []}
    if logprobs is not None:
        data["logprobs"] = logprobs
    with pytest.raises(VLLMValidationError, match="must not be an empty list") as exc:
        cls.model_validate(data)
    assert exc.value.parameter == "logprob_token_ids"


@pytest.mark.parametrize("cls", MODELS)
def test_rejects_token_ids_without_logprobs(cls):
    with pytest.raises(
        VLLMValidationError, match="when using `logprob_token_ids`"
    ) as exc:
        cls.model_validate({**BASE[cls], "logprob_token_ids": [100]})
    assert exc.value.parameter == "logprob_token_ids"


@pytest.mark.parametrize("cls", MODELS)
@pytest.mark.parametrize("val", [123, "123"])
def test_defers_non_list_to_pydantic(cls, val):
    with pytest.raises(ValidationError):
        cls.model_validate({**BASE[cls], "logprob_token_ids": val})
