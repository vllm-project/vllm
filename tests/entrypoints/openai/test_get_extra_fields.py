# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OpenAIBaseModel.get_extra_fields().

The OpenAI API accepts arbitrary extra fields by spec. vLLM preserves them
via Pydantic's ``extra="allow"`` configuration on
:class:`OpenAIBaseModel`, but until now downstream consumers had to read
them from ``model.__pydantic_extra__`` which is a private Pydantic API
that has changed shape between Pydantic releases.

``get_extra_fields()`` exposes that information as a public, documented
:class:`dict` so middleware, external safety classifiers, audit pipelines
and similar consumers can read it without dipping into Pydantic
internals.
"""

from __future__ import annotations

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel


def _minimal_chat_body(**extras):
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(extras)
    return body


def test_get_extra_fields_returns_unknown_fields_for_chat():
    req = ChatCompletionRequest.model_validate(
        _minimal_chat_body(
            audit_session_id="abc-123",
            experimental_safety=True,
            custom_meta={"tenant": "geodesia"},
        )
    )
    extras = req.get_extra_fields()
    assert extras == {
        "audit_session_id": "abc-123",
        "experimental_safety": True,
        "custom_meta": {"tenant": "geodesia"},
    }


def test_get_extra_fields_empty_when_only_known_fields():
    req = ChatCompletionRequest.model_validate(_minimal_chat_body())
    assert req.get_extra_fields() == {}


def test_get_extra_fields_returns_copy_not_alias():
    """Mutating the returned dict must not change the model state.

    Consumers can therefore safely mutate the returned dict in place
    (for example to redact sensitive values before logging) without
    accidentally corrupting the original request.
    """
    req = ChatCompletionRequest.model_validate(
        _minimal_chat_body(audit_session_id="abc-123")
    )
    extras = req.get_extra_fields()
    extras["audit_session_id"] = "REDACTED"
    assert req.get_extra_fields() == {"audit_session_id": "abc-123"}


def test_get_extra_fields_works_on_completion_request():
    req = CompletionRequest.model_validate(
        {
            "model": "test-model",
            "prompt": "Hello",
            "audit_session_id": "xyz",
        }
    )
    assert req.get_extra_fields() == {"audit_session_id": "xyz"}


def test_get_extra_fields_on_arbitrary_openaibasemodel_subclass():
    """The method is defined on ``OpenAIBaseModel`` so it works on
    every subclass, not just request models."""

    class DummyModel(OpenAIBaseModel):
        known: int = 0

    obj = DummyModel.model_validate({"known": 1, "unknown_a": "alpha", "unknown_b": 2})
    assert obj.get_extra_fields() == {"unknown_a": "alpha", "unknown_b": 2}


def test_get_extra_fields_returns_dict_type():
    req = ChatCompletionRequest.model_validate(_minimal_chat_body(custom="x"))
    extras = req.get_extra_fields()
    assert isinstance(extras, dict)
