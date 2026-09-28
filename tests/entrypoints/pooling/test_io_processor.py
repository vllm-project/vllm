# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm import PoolingParams
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.typing import OfflineEncodeInputsContext
from vllm.exceptions import VLLMValidationError
from vllm.renderers import TokenizeParams


@pytest.fixture
def processor() -> PoolingIOProcessor:
    return object.__new__(PoolingIOProcessor)


def test_rejects_untrusted_request_chat_template(processor: PoolingIOProcessor):
    with pytest.raises(VLLMValidationError) as exc_info:
        processor._validate_chat_template("template", None, False)

    assert str(exc_info.value) == (
        "Chat template is passed with request, but "
        "--trust-request-chat-template is not set. "
        "Refused request with untrusted chat template."
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_rejects_mismatched_pooling_params(processor: PoolingIOProcessor):
    with pytest.raises(VLLMValidationError) as exc_info:
        processor._params_to_seq([PoolingParams()], num_requests=2)

    assert str(exc_info.value) == (
        "The lengths of prompts (2) and params (1) must be the same."
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_rejects_mismatched_lora_requests(processor: PoolingIOProcessor):
    with pytest.raises(VLLMValidationError) as exc_info:
        processor._lora_request_to_seq([None], num_requests=2)

    assert str(exc_info.value) == (
        "The lengths of prompts (2) and lora_request (1) must be the same."
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_rejects_conflicting_pooling_task(processor: PoolingIOProcessor):
    processor.model_config = SimpleNamespace(is_encoder_decoder=False)
    processor.renderer = SimpleNamespace(
        default_cmpl_tok_params=TokenizeParams(max_total_tokens=None)
    )
    ctx = OfflineEncodeInputsContext(
        pooling_task="embed",
        tokenization_kwargs=None,
        lora_request=None,
        priorities=None,
        prompts=[[1]],
        pooling_params=PoolingParams(task="classify"),
    )

    with pytest.raises(VLLMValidationError) as exc_info:
        processor.get_request_factory_offline(ctx)

    assert str(exc_info.value) == (
        "You cannot overwrite param.task='classify' with pooling_task='embed'!"
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None
