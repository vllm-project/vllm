# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import pytest
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from vllm.renderers import ChatParams
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
    mock_renderer = MistralRenderer(mock_model_config, tokenizer_kwargs={})
    mock_renderer._tokenizer = mock_tokenizer

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
    assert prompt["prompt_token_ids"] == expected_tokens, (
        "Mocked blocking tokenizer was not called"
    )
    assert blocked_count == 0, "Event loop blocked during tokenization"


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
