# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.entrypoints.utils as entry_utils
from vllm import envs
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.utils import MAX_CHARS_PER_TOKEN


class DummyAsyncTokenizer:
    def __init__(self, return_len: int = 3, respect_truncation: bool = False):
        self.return_len = return_len
        self.respect_truncation = respect_truncation
        self.calls: list[dict] = []

    async def __call__(self, *_args, **kwargs):
        self.calls.append(kwargs)
        length = self.return_len
        if (
            self.respect_truncation
            and kwargs.get("truncation")
            and kwargs.get("max_length") is not None
        ):
            length = min(length, kwargs["max_length"])
        return SimpleNamespace(input_ids=list(range(length)))

    async def decode(self, input_ids):
        return " ".join(map(str, input_ids))


@pytest.fixture()
def serving() -> OpenAIServing:
    model_config = SimpleNamespace(
        max_model_len=8,
        encoder_config=None,
    )
    models = SimpleNamespace(
        model_config=model_config,
        input_processor=Mock(),
        io_processor=Mock(),
    )
    return OpenAIServing(engine_client=Mock(), models=models, request_logger=None)


@pytest.mark.asyncio
async def test_default_truncation_uses_guardrail(serving: OpenAIServing):
    tokenizer = DummyAsyncTokenizer(return_len=3)
    serving._get_async_tokenizer = Mock(return_value=tokenizer)

    request = SimpleNamespace(truncate_prompt_tokens=None, max_tokens=None)
    result = await serving._normalize_prompt_text_to_input(
        request, prompt="hello", tokenizer=object(), add_special_tokens=True
    )

    assert isinstance(result, dict)
    assert "prompt_token_ids" in result
    assert tokenizer.calls[0]["truncation"] is True
    assert tokenizer.calls[0]["max_length"] == serving.max_model_len + 1
    assert len(result["prompt_token_ids"]) == 3


@pytest.mark.asyncio
async def test_disable_truncation_passes_through(serving: OpenAIServing):
    tokenizer = DummyAsyncTokenizer(return_len=3)
    serving._get_async_tokenizer = Mock(return_value=tokenizer)

    request = SimpleNamespace(truncate_prompt_tokens=0, max_tokens=None)
    result = await serving._normalize_prompt_text_to_input(
        request, prompt="hello", tokenizer=object(), add_special_tokens=True
    )

    assert isinstance(result, dict)
    assert "prompt_token_ids" in result
    assert tokenizer.calls[0]["truncation"] is True
    assert tokenizer.calls[0]["max_length"] == 0
    assert len(result["prompt_token_ids"]) == 3


@pytest.mark.asyncio
async def test_custom_truncation_size_used(serving: OpenAIServing):
    tokenizer = DummyAsyncTokenizer(return_len=3)
    serving._get_async_tokenizer = Mock(return_value=tokenizer)

    request = SimpleNamespace(truncate_prompt_tokens=5, max_tokens=None)
    result = await serving._normalize_prompt_text_to_input(
        request, prompt="hello", tokenizer=object(), add_special_tokens=True
    )

    assert isinstance(result, dict)
    assert "prompt_token_ids" in result
    assert tokenizer.calls[0]["truncation"] is True
    assert tokenizer.calls[0]["max_length"] == 5
    assert len(result["prompt_token_ids"]) == 3


@pytest.mark.asyncio
async def test_prompt_too_long_raises_value_error(serving: OpenAIServing):
    # Tokenizer produces more tokens than model context allows; default path
    # should raise a validation error (surfaced as HTTP 400 in serving layer).
    tokenizer = DummyAsyncTokenizer(
        return_len=serving.max_model_len + 2, respect_truncation=True
    )
    serving._get_async_tokenizer = Mock(return_value=tokenizer)

    request = SimpleNamespace(truncate_prompt_tokens=None, max_tokens=None)
    with pytest.raises(ValueError):
        await serving._normalize_prompt_text_to_input(
            request, prompt="too long", tokenizer=object(), add_special_tokens=True
        )


@pytest.mark.asyncio
async def test_prompt_char_length_guard_early_fails(serving: OpenAIServing):
    tokenizer = DummyAsyncTokenizer(return_len=3)
    serving._get_async_tokenizer = Mock(return_value=tokenizer)

    request = SimpleNamespace(truncate_prompt_tokens=None, max_tokens=None)
    too_long = "x" * (serving.max_model_len * MAX_CHARS_PER_TOKEN + 1)
    with pytest.raises(ValueError):
        await serving._normalize_prompt_text_to_input(
            request, prompt=too_long, tokenizer=object(), add_special_tokens=True
        )


@pytest.mark.asyncio
async def test_prompt_char_length_guard_skipped_for_minus_one(serving: OpenAIServing):
    tokenizer = DummyAsyncTokenizer(return_len=3)
    serving._get_async_tokenizer = Mock(return_value=tokenizer)

    request = SimpleNamespace(truncate_prompt_tokens=-1, max_tokens=None)
    too_long = "x" * (serving.max_model_len * MAX_CHARS_PER_TOKEN + 1)
    result = await serving._normalize_prompt_text_to_input(
        request, prompt=too_long, tokenizer=object(), add_special_tokens=True
    )
    assert "prompt_token_ids" in result


@pytest.mark.asyncio
async def test_prompt_char_length_guard_respects_env_override(
    serving: OpenAIServing, monkeypatch
):
    # Reset env cache between tests (safe even if cache isn't enabled).
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_MAX_CHARS_PER_TOKEN", "1")
    # The module-level constant is read at import time, so reload is needed.
    importlib.reload(entry_utils)

    tokenizer = DummyAsyncTokenizer(return_len=3)
    serving._get_async_tokenizer = Mock(return_value=tokenizer)

    request = SimpleNamespace(truncate_prompt_tokens=None, max_tokens=None)
    too_long = "x" * (serving.max_model_len * 1 + 1)
    with pytest.raises(ValueError):
        await serving._normalize_prompt_text_to_input(
            request, prompt=too_long, tokenizer=object(), add_special_tokens=True
        )
