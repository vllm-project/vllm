# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Any

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.renderers import ChatParams
from vllm.renderers.kimi_k3 import KimiK3Renderer, _merge_k3_media_io_kwargs
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import TokenizerRegistry


class StubTokenizer:
    """Stands in for the model's TikTokenTokenizer.

    Records the kwargs it is called with and returns fixed token ids, so tests
    can assert how the renderer drives ``apply_chat_template`` without the real
    (git-LFS) tiktoken vocabulary.
    """

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.calls: list[dict[str, Any]] = []
        self.conversations: list[list[dict[str, Any]]] = []

    def apply_chat_template(self, conversation, **kwargs) -> list[int]:
        self.conversations.append(conversation)
        self.calls.append(kwargs)
        return list(self.token_ids)


@dataclass
class MockHFConfig:
    model_type: str = "kimi_k3"


@dataclass
class MockModelConfig:
    runner_type: str = "generate"
    is_multimodal_model: bool = False
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


def _make_renderer(tokenizer: StubTokenizer) -> KimiK3Renderer:
    config = MockVllmConfig(MockModelConfig(), MockParallelConfig())
    return KimiK3Renderer(config, tokenizer)


def test_kimi_k3_registered():
    assert RENDERER_REGISTRY.load_renderer_cls("kimi_k3").__name__ == "KimiK3Renderer"
    assert (
        TokenizerRegistry.load_tokenizer_cls("kimi_k3").__name__ == "CachedHfTokenizer"
    )


def test_k3_media_io_defaults_preserve_original_mode():
    # Default: K3 keeps the original image mode (no background flattening).
    assert _merge_k3_media_io_kwargs(None) == {"image": {"image_mode": None}}

    # Server-/request-level values take precedence over the K3 default.
    assert _merge_k3_media_io_kwargs({"image": {"image_mode": "RGB"}}) == {
        "image": {"image_mode": "RGB"}
    }

    # Unrelated image kwargs are merged with the default.
    assert _merge_k3_media_io_kwargs(
        {"image": {"rgba_background_color": (0, 0, 0)}}
    ) == {"image": {"image_mode": None, "rgba_background_color": (0, 0, 0)}}


def test_apply_chat_template_forces_tokenize_and_pins_return_dict():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    tools = [{"type": "function", "function": {"name": "search"}}]
    params = ChatParams(
        chat_template_kwargs={"tools": tools, "tokenize": False, "thinking": True}
    )

    token_ids = renderer._apply_chat_template(
        [{"role": "user", "content": "hi"}], params
    )

    assert token_ids == [7, 8, 9]
    kwargs = tokenizer.calls[-1]
    # tokenize is forced on even though the request asked for False, so K3 keeps
    # the special-vs-ordinary token distinction instead of re-tokenizing a string.
    assert kwargs["tokenize"] is True
    # return_dict is pinned False so we always get a flat list of ids.
    assert kwargs["return_dict"] is False
    assert kwargs["tools"] == tools
    assert kwargs["thinking"] is True


def test_apply_chat_template_translates_standard_thinking_kwargs():
    # Standard enable_thinking/reasoning_effort kwargs must be translated
    # to K3's native thinking/thinking_effort.
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(
        chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "none"}
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    kwargs = tokenizer.calls[-1]
    assert kwargs["thinking"] is False
    assert "thinking_effort" not in kwargs
    assert "enable_thinking" not in kwargs
    assert "reasoning_effort" not in kwargs


@pytest.mark.parametrize("reasoning_effort", ["low", "high", "max"])
def test_apply_chat_template_translates_supported_reasoning_effort(
    reasoning_effort: str,
):
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(chat_template_kwargs={"reasoning_effort": reasoning_effort})

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    kwargs = tokenizer.calls[-1]
    assert kwargs["thinking_effort"] == reasoning_effort
    assert "reasoning_effort" not in kwargs


@pytest.mark.parametrize("reasoning_effort", ["minimal", "medium", "xhigh"])
def test_apply_chat_template_rejects_unsupported_reasoning_effort(
    reasoning_effort: str,
):
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(chat_template_kwargs={"reasoning_effort": reasoning_effort})

    with pytest.raises(VLLMValidationError, match="thinking_effort") as exc_info:
        renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert exc_info.value.parameter == "thinking_effort"
    assert exc_info.value.value == reasoning_effort
    assert tokenizer.calls == []


def test_apply_chat_template_validates_canonical_native_thinking_effort():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(
        chat_template_kwargs={
            "thinking_effort": "low",
            "reasoning_effort": "medium",
        }
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert tokenizer.calls[-1]["thinking_effort"] == "low"


@pytest.mark.parametrize("thinking_effort", ["none", "minimal", "medium", "xhigh"])
def test_apply_chat_template_rejects_unsupported_native_thinking_effort(
    thinking_effort: str,
):
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(chat_template_kwargs={"thinking_effort": thinking_effort})

    with pytest.raises(VLLMValidationError, match="thinking_effort") as exc_info:
        renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert exc_info.value.parameter == "thinking_effort"
    assert exc_info.value.value == thinking_effort
    assert tokenizer.calls == []


def test_apply_chat_template_native_k3_kwargs_take_precedence():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(
        chat_template_kwargs={
            "thinking": True,
            "enable_thinking": False,
            "thinking_effort": "low",
            "reasoning_effort": "high",
        }
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    kwargs = tokenizer.calls[-1]
    assert kwargs["thinking"] is True
    assert kwargs["thinking_effort"] == "low"


def test_apply_chat_template_adds_k3_api_metadata():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    response_format = {"type": "json_object"}
    params = ChatParams(
        tool_choice="required",
        response_format=response_format,
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    kwargs = tokenizer.calls[-1]
    assert kwargs["tool_choice"] == "required"
    assert kwargs["response_format"] == response_format


def test_apply_chat_template_auto_tool_choice_keeps_template_kwarg():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(
        chat_template_kwargs={"tool_choice": "required"},
        tool_choice="auto",
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert tokenizer.calls[-1]["tool_choice"] == "required"


def test_apply_chat_template_omits_tool_choice_without_tools():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)

    renderer._apply_chat_template(
        [{"role": "user", "content": "hi"}], ChatParams(tool_choice=None)
    )

    assert "tool_choice" not in tokenizer.calls[-1]


def test_render_messages_returns_token_prompt():
    renderer = _make_renderer(StubTokenizer([1, 2, 3]))

    conversation, prompt = renderer.render_messages(
        [{"role": "user", "content": "hi"}], ChatParams()
    )

    assert prompt == {"prompt_token_ids": [1, 2, 3]}
    assert "multi_modal_data" not in prompt
    assert conversation[0]["role"] == "user"


def test_render_messages_derives_private_xtml_tool_attrs():
    tokenizer = StubTokenizer([1, 2, 3])
    renderer = _make_renderer(tokenizer)

    conversation, _ = renderer.render_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "lookup:0",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    },
                    {
                        "id": "lookup:1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "lookup:1",
                "tool": "client-supplied-name",
                "index": 99,
                "content": "second",
            },
            {
                "role": "tool",
                "tool_call_id": "lookup:0",
                "content": "first",
            },
        ],
        ChatParams(),
    )

    assert [message["content"] for message in conversation[1:]] == [
        "first",
        "second",
    ]
    assert conversation[1]["tool"] == "lookup"
    assert conversation[1]["index"] == 1
    assert conversation[2]["tool"] == "lookup"
    assert conversation[2]["index"] == 2
    assert tokenizer.conversations[-1] == conversation


def test_render_messages_ignores_client_supplied_xtml_tool_attrs():
    tokenizer = StubTokenizer([1, 2, 3])
    renderer = _make_renderer(tokenizer)

    conversation, _ = renderer.render_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "lookup:0",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "unknown",
                "tool": "lookup",
                "index": 1,
                "content": "result",
            },
        ],
        ChatParams(),
    )

    assert "tool" not in conversation[1]
    assert "index" not in conversation[1]


@pytest.mark.asyncio
async def test_render_messages_async_returns_token_prompt():
    renderer = _make_renderer(StubTokenizer([4, 5]))

    conversation, prompt = await renderer.render_messages_async(
        [{"role": "user", "content": "hi"}], ChatParams()
    )

    assert prompt == {"prompt_token_ids": [4, 5]}
    assert conversation[0]["role"] == "user"


# ---------------------------------------------------------------------------
# generation_prefix_len — issue #51465
# ---------------------------------------------------------------------------

class _PrefixAwareStubTokenizer(StubTokenizer):
    """Returns full_ids when add_generation_prompt=True, base_ids otherwise.

    Simulates the K3 chat template appending a channel-open stub of
    ``len(full_ids) - len(base_ids)`` tokens when add_generation_prompt is on.
    """

    def __init__(self, base_ids: list[int], stub_ids: list[int]) -> None:
        super().__init__(base_ids + stub_ids)
        self._base_ids = base_ids
        self._stub_ids = stub_ids

    def apply_chat_template(self, conversation, **kwargs) -> list[int]:
        self.conversations.append(conversation)
        self.calls.append(kwargs)
        if kwargs.get("add_generation_prompt", True):
            return list(self._base_ids + self._stub_ids)
        return list(self._base_ids)


def test_generation_prefix_len_set_on_prompt():
    """render_messages sets generation_prefix_len equal to the stub length."""
    stub = _PrefixAwareStubTokenizer(
        base_ids=list(range(99)),
        stub_ids=[200, 201, 202],  # 3-token channel-open stub
    )
    renderer = _make_renderer(stub)

    _, prompt = renderer.render_messages(
        [{"role": "user", "content": "Hello"}], ChatParams()
    )

    assert prompt["prompt_token_ids"] == list(range(99)) + [200, 201, 202]
    assert prompt["generation_prefix_len"] == 3


def test_generation_prefix_len_absent_when_no_prefix():
    """When the stub length is 0 the field is absent (not set to 0)."""
    stub = _PrefixAwareStubTokenizer(
        base_ids=list(range(10)),
        stub_ids=[],  # no stub
    )
    renderer = _make_renderer(stub)

    _, prompt = renderer.render_messages(
        [{"role": "user", "content": "hi"}], ChatParams()
    )

    assert "generation_prefix_len" not in prompt


def test_generation_prefix_len_absent_when_add_generation_prompt_false():
    """With add_generation_prompt=False no stub is appended → field absent."""
    stub = _PrefixAwareStubTokenizer(
        base_ids=list(range(10)),
        stub_ids=[200, 201, 202],
    )
    renderer = _make_renderer(stub)
    params = ChatParams(chat_template_kwargs={"add_generation_prompt": False})

    _, prompt = renderer.render_messages(
        [{"role": "user", "content": "hi"}], params
    )

    assert "generation_prefix_len" not in prompt


@pytest.mark.asyncio
async def test_generation_prefix_len_set_async():
    """render_messages_async also sets generation_prefix_len."""
    stub = _PrefixAwareStubTokenizer(
        base_ids=list(range(5)),
        stub_ids=[100, 101, 102],
    )
    renderer = _make_renderer(stub)

    _, prompt = await renderer.render_messages_async(
        [{"role": "user", "content": "hi"}], ChatParams()
    )

    assert prompt["generation_prefix_len"] == 3


def test_generation_prefix_len_propagated_to_engine_input():
    """generation_prefix_len on the DictPrompt flows into the TokensInput."""
    from vllm.renderers.inputs.preprocess import parse_dec_only_prompt

    stub = _PrefixAwareStubTokenizer(
        base_ids=list(range(99)),
        stub_ids=[200, 201, 202],
    )
    renderer = _make_renderer(stub)
    _, prompt = renderer.render_messages(
        [{"role": "user", "content": "Hello"}], ChatParams()
    )

    # Simulate what BaseRenderer._process_tokens does: propagate into engine input.
    engine_input = parse_dec_only_prompt(prompt["prompt_token_ids"])
    engine_input["generation_prefix_len"] = prompt.get("generation_prefix_len", 0)

    assert engine_input.get("generation_prefix_len") == 3
