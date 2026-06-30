# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the token-offsets request/response protocol wiring:
the request flag flowing into ``TokenizeParams`` and the ``GenerateRequest``
serialization boundary. End-to-end behavior is covered by
``tests/entrypoints/scale_out/render/test_render.py``; plain Pydantic field
storage is not retested here.
"""

from unittest.mock import Mock

from vllm.config import ModelConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.sampling_params import SamplingParams


def _model_config() -> Mock:
    model_config = Mock(spec=ModelConfig)
    model_config.max_model_len = 128
    return model_config


def test_completion_flag_forwarded_to_tok_params():
    """build_tok_params must forward return_token_offsets, defaulting to
    False (zero behavioral change for existing callers) and coercing JSON
    null to False via the bool() guard."""
    cfg = _model_config()

    default = CompletionRequest(model="m", prompt="hi")
    assert default.build_tok_params(cfg).return_token_offsets is False

    on = CompletionRequest(model="m", prompt="hi", return_token_offsets=True)
    assert on.build_tok_params(cfg).return_token_offsets is True

    null = CompletionRequest(model="m", prompt="hi", return_token_offsets=None)
    assert null.build_tok_params(cfg).return_token_offsets is False


def test_chat_flag_forwarded_to_tok_params():
    """Chat build_tok_params has its own (max_completion_tokens) branch, so
    its return_token_offsets forwarding is verified independently."""
    cfg = _model_config()
    messages = [{"role": "user", "content": "hi"}]

    default = ChatCompletionRequest(model="m", messages=messages)
    assert default.build_tok_params(cfg).return_token_offsets is False

    on = ChatCompletionRequest(model="m", messages=messages, return_token_offsets=True)
    assert on.build_tok_params(cfg).return_token_offsets is True

    null = ChatCompletionRequest(
        model="m", messages=messages, return_token_offsets=None
    )
    assert null.build_tok_params(cfg).return_token_offsets is False


def test_generate_request_token_offsets_default_none():
    """Defaults to None so existing /v1/.../render responses are unchanged."""
    req = GenerateRequest(token_ids=[1, 2, 3], sampling_params=SamplingParams())
    assert req.token_offsets is None


def test_generate_request_token_offsets_survive_json_round_trip():
    """GenerateRequest crosses the disagg serialization boundary; the
    tuple[int, int] offsets must survive model_dump and re-validate."""
    req = GenerateRequest(
        token_ids=[10, 20],
        sampling_params=SamplingParams(),
        token_offsets=[(0, 1), (1, 3)],
    )
    dumped = req.model_dump()
    assert dumped["token_offsets"] == [(0, 1), (1, 3)]
    # Re-validate from the dumped dict (sampling_params doesn't round-trip
    # cleanly via dump, so re-inject a fresh instance).
    again = GenerateRequest.model_validate(
        {**dumped, "sampling_params": SamplingParams()}
    )
    assert again.token_offsets == [(0, 1), (1, 3)]
