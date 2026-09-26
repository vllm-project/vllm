# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the token_in_token_out request/response protocol.

These tests intentionally avoid spinning up a server — they exercise the
pydantic validators on ``GenerateRequest`` directly so they run fast and
fail loudly if the validator semantics ever drift.
"""

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    DerenderChatRequest,
    DerenderChatStreamRequest,
    GenerateRequest,
    GenerateResponse,
    GenerateStreamResponse,
    GenerateTextResponse,
    GenerateTextStreamResponse,
    GenerateTokensChoice,
    GenerateTokensResponse,
    GenerateTokensStreamResponse,
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.sampling_params import SamplingParams


def _base_payload() -> dict:
    return {"token_ids": [1, 2, 3], "sampling_params": {}}


def test_omitted_max_tokens_is_not_provided():
    """Body without ``max_tokens`` must surface as 'not provided' so the
    server can apply its own default instead of the dataclass 16."""
    req = GenerateRequest.model_validate(_base_payload())
    # SamplingParams' dataclass default leaks through the parsed instance —
    # this is exactly the bug the server-side defaulting works around.
    assert req.sampling_params.max_tokens == 16
    assert not req.is_sampling_param_provided("max_tokens")


def test_explicit_max_tokens_is_provided():
    """Even when the client picks the same value as the dataclass default,
    it must register as explicitly set so the server won't override it."""
    payload = _base_payload()
    payload["sampling_params"] = {"max_tokens": 16}
    req = GenerateRequest.model_validate(payload)
    assert req.sampling_params.max_tokens == 16
    assert req.is_sampling_param_provided("max_tokens")

    payload["sampling_params"] = {"max_tokens": 256}
    req = GenerateRequest.model_validate(payload)
    assert req.sampling_params.max_tokens == 256
    assert req.is_sampling_param_provided("max_tokens")


def test_other_fields_tracked_independently():
    payload = _base_payload()
    payload["sampling_params"] = {"temperature": 0.0}
    req = GenerateRequest.model_validate(payload)
    assert not req.is_sampling_param_provided("max_tokens")
    assert req.is_sampling_param_provided("temperature")


def test_json_roundtrip_preserves_provided_keys():
    payload = _base_payload()
    payload["sampling_params"] = {"temperature": 0.5}
    req = GenerateRequest.model_validate_json(json.dumps(payload))
    assert not req.is_sampling_param_provided("max_tokens")
    assert req.is_sampling_param_provided("temperature")


def test_internal_instance_construction_treats_all_as_provided():
    """When internal callers build ``GenerateRequest`` from a pre-resolved
    ``SamplingParams`` instance, every field is considered explicitly set
    so server-side defaulting can't clobber values resolved upstream."""
    sp = SamplingParams(max_tokens=500, temperature=0.0)
    req = GenerateRequest(token_ids=[1, 2, 3], sampling_params=sp)
    assert req.is_sampling_param_provided("max_tokens")
    assert req.is_sampling_param_provided("temperature")
    # And keys we never touched should also count as provided in this path.
    assert req.is_sampling_param_provided("top_p")


def test_multimodal_features_reject_mismatched_parallel_fields():
    with pytest.raises(ValueError, match="same length"):
        MultiModalFeatures(
            mm_hashes={"image": ["a", "b"]},
            mm_placeholders={"image": [PlaceholderRangeInfo(offset=0, length=1)]},
            kwargs_data={"image": ["encoded"]},
        )


def test_multimodal_features_reject_overlapping_placeholders():
    with pytest.raises(ValueError, match="non-overlapping"):
        MultiModalFeatures(
            mm_hashes={"image": ["a", "b"]},
            mm_placeholders={
                "image": [
                    PlaceholderRangeInfo(offset=1, length=2),
                    PlaceholderRangeInfo(offset=2, length=2),
                ]
            },
            kwargs_data={"image": ["a", "b"]},
        )


def test_generate_request_rejects_placeholder_outside_prompt():
    with pytest.raises(ValueError, match="within the token_ids sequence"):
        GenerateRequest(
            token_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            features=MultiModalFeatures(
                mm_hashes={"image": ["a"]},
                mm_placeholders={"image": [PlaceholderRangeInfo(offset=2, length=2)]},
                kwargs_data={"image": ["encoded"]},
            ),
        )


def test_output_mode_defaults_to_tokens():
    assert GenerateRequest.model_validate(_base_payload()).output_mode == "tokens"


def test_output_mode_text_is_accepted():
    payload = {**_base_payload(), "output_mode": "text"}
    assert GenerateRequest.model_validate(payload).output_mode == "text"


@pytest.mark.parametrize("output_mode", ["derender", "bogus", None])
def test_output_mode_rejects_unsupported_levels(output_mode):
    """Levels the server does not implement fail instead of falling back to
    tokens which would answer 200 with the wrong response shape."""
    payload = {**_base_payload(), "output_mode": output_mode}
    with pytest.raises(ValidationError, match="output_mode"):
        GenerateRequest.model_validate(payload)


def test_text_mode_rejects_detokenize_false():
    payload = {
        "token_ids": [1, 2, 3],
        "sampling_params": {"detokenize": False},
        "output_mode": "text",
    }
    with pytest.raises(ValidationError, match="detokenize"):
        GenerateRequest.model_validate(payload)


def test_tokens_mode_allows_detokenize_false():
    payload = {
        "token_ids": [1, 2, 3],
        "sampling_params": {"detokenize": False},
        "output_mode": "tokens",
    }
    assert GenerateRequest.model_validate(payload).output_mode == "tokens"


def test_response_without_output_mode_parses_as_tokens():
    """Existing clients and older servers never send output_mode, and
    /derender parses GenerateResponse as input."""
    parsed = TypeAdapter(GenerateResponse).validate_python(
        {"choices": [{"index": 0, "token_ids": [1]}]}
    )
    assert isinstance(parsed, GenerateTokensResponse)
    assert parsed.output_mode == "tokens"


def test_stream_chunk_without_output_mode_parses_as_tokens():
    parsed = TypeAdapter(GenerateStreamResponse).validate_python(
        {"choices": [{"index": 0, "token_ids": [1]}]}
    )
    assert isinstance(parsed, GenerateTokensStreamResponse)


def test_text_response_and_chunk_dispatch_on_output_mode():
    response = TypeAdapter(GenerateResponse).validate_python(
        {"output_mode": "text", "choices": [{"index": 0, "text": "hi"}]}
    )
    chunk = TypeAdapter(GenerateStreamResponse).validate_python(
        {"output_mode": "text", "choices": [{"index": 0, "text": "hi"}]}
    )
    assert isinstance(response, GenerateTextResponse)
    assert isinstance(chunk, GenerateTextStreamResponse)


def test_text_response_requires_text_on_every_choice():
    with pytest.raises(ValidationError, match="text"):
        TypeAdapter(GenerateResponse).validate_python(
            {"output_mode": "text", "choices": [{"index": 0}]}
        )


@pytest.mark.parametrize("output_mode", ["derender", None, 5])
def test_response_rejects_unknown_output_mode(output_mode):
    with pytest.raises(ValidationError):
        TypeAdapter(GenerateResponse).validate_python(
            {"output_mode": output_mode, "choices": []}
        )


def test_derender_requests_accept_responses_without_output_mode():
    DerenderChatRequest.model_validate(
        {"generate_response": {"choices": [{"index": 0, "token_ids": [1]}]}}
    )
    DerenderChatStreamRequest.model_validate(
        {
            "stream": True,
            "generate_chunk": {"choices": [{"index": 0, "token_ids": [1]}]},
        }
    )


def test_tokens_response_echoes_output_mode_without_text():
    dumped = GenerateTokensResponse(
        choices=[GenerateTokensChoice(index=0, token_ids=[1])]
    ).model_dump()
    assert dumped["output_mode"] == "tokens"
    assert "text" not in dumped["choices"][0]
