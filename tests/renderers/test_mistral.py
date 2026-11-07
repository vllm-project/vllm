# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import pytest
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from vllm.entrypoints.generate.base.protocol import (
    JsonSchemaResponseFormat as OpenAIJsonSchemaResponseFormat,
)
from vllm.entrypoints.generate.base.protocol import (
    ResponseFormat as OpenAIResponseFormat,
)
from vllm.renderers import ChatParams, TokenizeParams
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.renderers.mistral import MistralRenderer, safe_apply_chat_template
from vllm.tokenizers.mistral import MistralTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    runner_type = "generate"
    model: str = MODEL_NAME
    tokenizer: str = MODEL_NAME
    trust_remote_code: bool = False
    max_model_len: int = 100
    tokenizer_revision = None
    tokenizer_mode = "mistral"
    hf_config = MockHFConfig()
    encoder_config: dict[str, Any] | None = None
    enable_prompt_embeds: bool = True
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


@pytest.mark.asyncio
async def test_async_mistral_tokenizer_does_not_block_event_loop():
    expected_tokens = [1, 2, 3]

    # Mock the blocking version to sleep
    def mocked_apply_chat_template(*_args, **_kwargs):
        time.sleep(2)
        return expected_tokens

    mock_model_config = MockModelConfig(skip_tokenizer_init=True)
    mock_tokenizer = Mock(spec=MistralTokenizer)
    mock_tokenizer.apply_chat_template = mocked_apply_chat_template
    mock_renderer = MistralRenderer(
        MockVllmConfig(mock_model_config, parallel_config=MockParallelConfig()),
        tokenizer=mock_tokenizer,
    )

    task = mock_renderer.render_messages_async([], ChatParams())

    # Ensure the event loop is not blocked
    blocked_count = 0
    for _i in range(20):  # Check over ~2 seconds
        start = time.perf_counter()
        await asyncio.sleep(0)
        elapsed = time.perf_counter() - start

        # an overly generous elapsed time for slow machines
        if elapsed >= 0.5:
            blocked_count += 1

        await asyncio.sleep(0.1)

    # Ensure task completes
    _, prompt = await task
    # ``.get`` because ``prompt`` is typed as any ``DictPrompt``, and only the
    # token-prompt members of that union declare ``prompt_token_ids``.
    assert prompt.get("prompt_token_ids") == expected_tokens, (
        "Mocked blocking tokenizer was not called"
    )
    assert blocked_count == 0, "Event loop blocked during tokenization"


@pytest.mark.parametrize(
    ("response_format", "expected"),
    [
        (
            OpenAIResponseFormat(
                type="json_schema",
                json_schema=OpenAIJsonSchemaResponseFormat(
                    name="answer", schema={"type": "object"}, strict=True
                ),
            ),
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
        ),
        (OpenAIResponseFormat(type="text"), None),
    ],
    ids=["json_schema", "text"],
)
def test_renderer_passes_response_format_to_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    response_format: OpenAIResponseFormat,
    expected: dict[str, Any] | None,
) -> None:
    mock_tokenizer = Mock(spec=MistralTokenizer)
    mock_tokenizer.apply_chat_template = Mock(return_value=[1])
    renderer = MistralRenderer(
        MockVllmConfig(
            MockModelConfig(skip_tokenizer_init=True),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=mock_tokenizer,
    )
    monkeypatch.setattr(
        "vllm.renderers.mistral.parse_chat_messages",
        lambda *_args, **_kwargs: ([], None, None),
    )

    renderer.render_messages([], ChatParams(response_format=response_format))

    kwargs = mock_tokenizer.apply_chat_template.call_args.kwargs
    assert kwargs.get("response_format") == expected


def test_apply_mistral_chat_template_thinking_chunk():
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
                {
                    "type": "thinking",
                    "closed": True,
                    "thinking": "Only return the answer when you are confident.",
                },
            ],
        },
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me think about it."},
                {"type": "thinking", "closed": True, "thinking": "2+2 = 4"},
                {
                    "type": "text",
                    "text": "The answer is 4.",
                },
            ],
        },
        {"role": "user", "content": "Thanks, what is 3+3?"},
    ]
    mistral_tokenizer = MistralTokenizer.from_pretrained(
        "mistralai/Magistral-Small-2509"
    )

    tokens_ids = safe_apply_chat_template(
        mistral_tokenizer, messages, chat_template=None, tools=None
    )

    string_tokens = mistral_tokenizer.mistral.decode(
        tokens_ids, special_token_policy=SpecialTokenPolicy.KEEP
    )

    expected_tokens = (
        r"<s>[SYSTEM_PROMPT]You are a helpful assistant.[THINK]Only return the"
        r" answer when you are confident.[/THINK][/SYSTEM_PROMPT]"
        r"[INST]What is 2+2?[/INST]"
        r"Let me think about it.[THINK]2+2 = 4[/THINK]The answer is 4.</s>"
        r"[INST]Thanks, what is 3+3?[/INST]"
    )

    assert string_tokens == expected_tokens


@pytest.mark.parametrize("truncation_side", ["left", "right"])
def test_explicit_truncation_side_keeps_bos(truncation_side):
    """Truncating from an explicit side must keep BOS, as the tokenizer's own
    truncation does, although `MistralTokenizer` has no `special_tokens_mask`."""
    renderer = MistralRenderer(
        MockVllmConfig(MockModelConfig(), parallel_config=MockParallelConfig()),
        tokenizer=MistralTokenizer.from_pretrained(MODEL_NAME),
    )
    text = "The quick brown fox jumps over the lazy dog."

    prompts = renderer.render_prompts([parse_model_prompt(renderer.model_config, text)])
    (result,) = renderer.tokenize_prompts(
        prompts,
        TokenizeParams(
            max_total_tokens=100,
            truncate_prompt_tokens=4,
            truncation_side=truncation_side,
        ),
    )

    expected = MistralTokenizer.from_pretrained(
        MODEL_NAME, truncation_side=truncation_side
    )(text, truncation=True, max_length=4)
    # `.get`, as the result is typed as any `TokPrompt` and not every member of
    # that union has this key.
    assert result.get("prompt_token_ids") == expected["input_ids"]
