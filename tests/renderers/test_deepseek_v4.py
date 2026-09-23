# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Any

import pytest

from vllm.renderers import ChatParams
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer, _join_tool_text_parts


class StubTokenizer:
    """Records the conversation the renderer hands to ``apply_chat_template``."""

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.conversations: list[list[dict[str, Any]]] = []

    def apply_chat_template(self, conversation, **kwargs) -> list[int]:
        self.conversations.append(conversation)
        return list(self.token_ids)


@dataclass
class MockHFConfig:
    model_type: str = "deepseek_v4"


@dataclass
class MockModelConfig:
    runner_type: str = "generate"
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    multimodal_config: Any = None
    hf_config: MockHFConfig = field(default_factory=MockHFConfig)
    allowed_local_media_path: str = ""
    allowed_media_domains: Any = None
    enable_prompt_embeds: bool = False
    renderer_num_workers: int = 1


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


def _make_renderer(tokenizer: StubTokenizer) -> DeepseekV4Renderer:
    config = MockVllmConfig(MockModelConfig(), MockParallelConfig())
    return DeepseekV4Renderer(config, tokenizer)


def _messages(tool_content):
    return [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": tool_content},
    ]


class TestJoinToolTextParts:
    def test_text_parts_join_with_blank_line(self):
        parts = [{"type": "text", "text": "line 1"}, {"type": "text", "text": "line 2"}]
        out = _join_tool_text_parts(_messages(parts))
        assert out[2]["content"] == "line 1\n\nline 2"
        # other messages and the original list are untouched
        assert out[0] == {"role": "user", "content": "Weather?"}
        assert isinstance(_messages(parts)[2]["content"], list)

    def test_string_content_untouched(self):
        out = _join_tool_text_parts(_messages("Sunny"))
        assert out[2]["content"] == "Sunny"

    def test_non_text_parts_kept_structured(self):
        parts = [{"type": "text", "text": "a"}, {"type": "tool_reference", "id": "x"}]
        out = _join_tool_text_parts(_messages(parts))
        assert out[2]["content"] == parts


def test_render_messages_joins_tool_text_parts_with_blank_line():
    tokenizer = StubTokenizer([1, 2, 3])
    renderer = _make_renderer(tokenizer)
    parts = [{"type": "text", "text": "line 1"}, {"type": "text", "text": "line 2"}]
    conversation, prompt = renderer.render_messages(_messages(parts), ChatParams())
    assert prompt["prompt_token_ids"] == [1, 2, 3]
    assert conversation[-1]["role"] == "tool"
    assert conversation[-1]["content"] == "line 1\n\nline 2"
    assert tokenizer.conversations[0][-1]["content"] == "line 1\n\nline 2"


@pytest.mark.asyncio
async def test_render_messages_async_joins_tool_text_parts_with_blank_line():
    tokenizer = StubTokenizer([4, 5])
    renderer = _make_renderer(tokenizer)
    parts = [{"type": "text", "text": "line 1"}, {"type": "text", "text": "line 2"}]
    conversation, _ = await renderer.render_messages_async(
        _messages(parts), ChatParams()
    )
    assert conversation[-1]["content"] == "line 1\n\nline 2"
