# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.batches.jsonl_protocol import (
    BatchRequestInput,
    BatchRequestOutput,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.pooling.scoring.protocol import RerankRequest
from vllm.entrypoints.serve import create_error_response


def _line(url: str, body: dict) -> dict:
    return {"custom_id": "r1", "method": "POST", "url": url, "body": body}


def test_body_is_parsed_by_url():
    chat = BatchRequestInput.model_validate(
        _line("/v1/chat/completions", {"model": "m", "messages": []})
    )
    rerank = BatchRequestInput.model_validate(
        _line("/v1/rerank", {"model": "m", "query": "q", "documents": ["d"]})
    )

    assert isinstance(chat.body, ChatCompletionRequest)
    assert isinstance(rerank.body, RerankRequest)


def test_unsupported_url_is_rejected():
    with pytest.raises(ValidationError, match="Unsupported url"):
        BatchRequestInput.model_validate(_line("/v1/completions", {}))


def test_output_success_and_failure():
    success = BatchRequestOutput.success("r1", {"ok": True})
    failure = BatchRequestOutput.failure("r2", create_error_response("bad"))

    assert success.error is None
    assert success.response is not None
    assert success.response.body == {"ok": True}
    assert failure.error is not None
    assert failure.response is not None
    assert failure.response.status_code == 400
