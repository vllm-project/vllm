# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounds on request-controlled inputs that would otherwise amplify work.

Consolidates the regression tests for the request-input amplification fixes:
stop-string caps, bad-words dedup/tokenization limit, stop-token-id dedup,
beam-width/sequence caps, and the DeepSeek history-scan bound.
"""

import os
import subprocess
import sys
from collections.abc import Callable
from typing import Protocol

import pytest
from pydantic import ValidationError

import vllm.envs as envs
from vllm import SamplingParams
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import BeamSearchParams
from vllm.tokenizers import deepseek_v4_encoding, deepseek_v32_encoding

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


# --- Stop strings: public requests cap the number of stop strings ---------


class _StopRequest(Protocol):
    stop: str | list[str] | None


def _completion_request(stop: list[str]) -> _StopRequest:
    return CompletionRequest(model="test-model", prompt="hello", stop=stop)


def _chat_request(stop: list[str]) -> _StopRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        stop=stop,
    )


def _batch_chat_request(stop: list[str]) -> _StopRequest:
    return BatchChatCompletionRequest(
        model="test-model",
        messages=[[{"role": "user", "content": "hello"}]],
        stop=stop,
    )


def _responses_request(stop: list[str]) -> _StopRequest:
    return ResponsesRequest(model="test-model", input="hello", stop=stop)


REQUEST_BUILDERS: list[Callable[[list[str]], _StopRequest]] = [
    _completion_request,
    _chat_request,
    _batch_chat_request,
    _responses_request,
]


@pytest.mark.parametrize("build_request", REQUEST_BUILDERS)
def test_public_requests_accept_four_stop_strings(
    build_request: Callable[[list[str]], _StopRequest],
):
    stop = ["one", "two", "three", "four"]

    request = build_request(stop)

    assert request.stop == stop


@pytest.mark.parametrize("build_request", REQUEST_BUILDERS)
def test_public_requests_reject_more_than_four_stop_strings(
    build_request: Callable[[list[str]], _StopRequest],
):
    with pytest.raises(ValidationError, match="at most 4"):
        build_request(["one", "two", "three", "four", "five"])


def test_stop_string_limit_can_be_overridden():
    env = os.environ.copy()
    env["VLLM_MAX_STOP_STRINGS"] = "1"
    code = """
from pydantic import ValidationError
from vllm.entrypoints.openai.completion.protocol import CompletionRequest

try:
    CompletionRequest(
        model="test-model",
        prompt="hello",
        stop=["one", "two"],
    )
except ValidationError as error:
    assert "at most 1" in str(error)
else:
    raise AssertionError("configured stop-string limit was not enforced")
"""

    subprocess.run([sys.executable, "-c", code], check=True, env=env)


# --- Stop token ids: duplicates are deduplicated in order ------------------


def test_duplicate_stop_token_ids_are_deduplicated_in_order():
    params = SamplingParams(stop_token_ids=[42, 7, 42, 9, 7])

    assert params.stop_token_ids == [42, 7, 9]
    assert params.all_stop_token_ids == {7, 9, 42}


# --- Bad words: dedup, and the tokenization pass is bounded ----------------


class MockTokenizer:
    max_token_id = 1024

    def __init__(self):
        self.calls = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.calls += 1
        return [2] if text.startswith(" ") else [1]


def test_duplicate_bad_words_are_deduplicated_in_order():
    params = SamplingParams(bad_words=["bad", "worse", "bad", "worst"])

    assert params.bad_words == ["bad", "worse", "worst"]


def test_bad_word_tokenization_stops_at_worker_limit():
    params = SamplingParams(bad_words=[f"word-{i}" for i in range(65)])
    tokenizer = MockTokenizer()

    with pytest.raises(VLLMValidationError, match="Too many bad words"):
        params.update_from_tokenizer(tokenizer)

    assert tokenizer.calls == 129


def test_bad_word_tokenization_limit_can_be_overridden(monkeypatch):
    monkeypatch.setenv("VLLM_MAX_NUM_BAD_WORDS", "2")
    params = SamplingParams(bad_words=["bad", "worse"])
    tokenizer = MockTokenizer()

    with pytest.raises(VLLMValidationError, match="The max number is 2"):
        params.update_from_tokenizer(tokenizer)

    assert tokenizer.calls == 3


class EmptyBaseEncodingTokenizer:
    max_token_id = 1024

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [216] if text.startswith(" ") else []


def test_bad_word_rejects_empty_base_tokenization():
    params = SamplingParams(bad_words=["\x16"])

    with pytest.raises(
        VLLMValidationError,
        match="must tokenize to at least one token",
    ) as exc_info:
        params.update_from_tokenizer(EmptyBaseEncodingTokenizer())

    assert exc_info.value.parameter == "bad_words"


class EmptyPrefixedEncodingTokenizer:
    max_token_id = 1024

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [] if text.startswith(" ") else [321]


def test_bad_word_skips_empty_optional_prefixed_tokenization():
    params = SamplingParams(bad_words=["word"])

    params.update_from_tokenizer(EmptyPrefixedEncodingTokenizer())

    assert params.bad_words_token_ids == [[321]]


# --- Beam search: beam width / n honor the sequence cap --------------------


def _set_max_n(monkeypatch: pytest.MonkeyPatch, value: int) -> None:
    monkeypatch.setenv("VLLM_MAX_N_SEQUENCES", str(value))
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()


def test_direct_beam_width_rejects_values_over_sequence_cap(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_max_n(monkeypatch, 4)

    with pytest.raises(VLLMValidationError, match="beam_width must be at most 4"):
        BeamSearchParams(beam_width=5, max_tokens=1)


def test_chat_beam_conversion_rejects_n_before_stream_state_allocation(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_max_n(monkeypatch, 4)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        n=5,
        stream=True,
        use_beam_search=True,
        max_tokens=1,
    )

    with pytest.raises(VLLMValidationError, match="beam_width must be at most 4"):
        request.to_beam_search_params(max_tokens=1, default_sampling_params={})


def test_chat_beam_conversion_accepts_n_at_sequence_cap(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_max_n(monkeypatch, 4)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        n=4,
        stream=True,
        use_beam_search=True,
        max_tokens=1,
    )

    params = request.to_beam_search_params(max_tokens=1, default_sampling_params={})

    assert params.beam_width == 4


# --- DeepSeek encoders: the last-user scan runs once per conversation ------


ENCODING_MODULES = [deepseek_v32_encoding, deepseek_v4_encoding]


@pytest.mark.parametrize(
    "encoding_module",
    ENCODING_MODULES,
    ids=["deepseek_v32", "deepseek_v4"],
)
def test_encode_messages_scans_last_user_once_per_conversation(
    monkeypatch: pytest.MonkeyPatch,
    encoding_module,
):
    calls = 0
    original_find_last_user_index = encoding_module.find_last_user_index

    def counted_find_last_user_index(messages):
        nonlocal calls
        calls += 1
        return original_find_last_user_index(messages)

    monkeypatch.setattr(
        encoding_module,
        "find_last_user_index",
        counted_find_last_user_index,
    )

    messages = [{"role": "user", "content": "Hello"}]
    messages.extend({"role": "assistant", "content": "Hi"} for _ in range(8))

    encoding_module.encode_messages(messages, thinking_mode="chat")

    assert calls == 1


@pytest.mark.parametrize(
    "encoding_module",
    ENCODING_MODULES,
    ids=["deepseek_v32", "deepseek_v4"],
)
def test_encode_messages_preserves_small_chat_prompt(encoding_module):
    prompt = encoding_module.encode_messages(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "assistant", "content": "Again"},
        ],
        thinking_mode="chat",
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
        "Hi<｜end▁of▁sentence｜>Again<｜end▁of▁sentence｜>"
    )


@pytest.mark.parametrize(
    "encoding_module",
    ENCODING_MODULES,
    ids=["deepseek_v32", "deepseek_v4"],
)
def test_encode_messages_unknown_role_raises_value_error(encoding_module):
    # An invalid role (e.g. uppercase "SYSTEM") is a client error and must be
    # raised as ValueError so the OpenAI serving layer maps it to HTTP 400
    # instead of NotImplementedError, which would map to HTTP 501.
    with pytest.raises(ValueError, match="Invalid role: SYSTEM"):
        encoding_module.encode_messages(
            [{"role": "SYSTEM", "content": "Hello"}],
            thinking_mode="chat",
        )
