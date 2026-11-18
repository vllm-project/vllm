# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.tokenizers_.test_deepseek_v4 import FakeHfTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.chat_utils import BaseMultiModalItemTracker
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
from vllm.renderers.params import ChatParams
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.deepseek_v41 import get_deepseek_v41_tokenizer
from vllm.tokenizers.deepseek_v41_encoding import IMAGE_PLACEHOLDER, encode_messages
from vllm.utils.async_utils import make_async

FIXTURES = Path(__file__).parent / "fixtures"


def render(messages, **kwargs):
    return get_deepseek_v41_tokenizer(FakeHfTokenizer()).apply_chat_template(
        messages, tokenize=False, **kwargs
    )


def tool_call(name):
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


@pytest.mark.parametrize("case_id", [1, 2, 3, 4, 5])
def test_reference_encoder_fixtures(case_id):
    # Copied from DeepSeek-V4.1-Flash-1017 encoding/tests.
    data = json.loads(
        (FIXTURES / "deepseek_v41" / f"test_input_{case_id}.json").read_text()
    )
    case = data if isinstance(data, dict) else {"messages": data}
    messages = case["messages"]
    thinking_mode = case.get("thinking_mode") or "chat"
    expected = (FIXTURES / "deepseek_v41" / f"test_output_{case_id}.txt").read_text()

    # The reference encodes a conversation that ends with an assistant message as
    # closed history, which is add_generation_prompt=False here.
    final_assistant = messages[-1]["role"] == "assistant"
    if any(m["role"] == "latest_reminder" for m in messages):
        # latest_reminder is not an API role.
        assert case.get("tools") is None
        prompt = encode_messages(
            messages,
            thinking_mode=thinking_mode,
            reasoning_effort=case.get("reasoning_effort"),
            add_generation_prompt=not final_assistant,
        )
    else:
        kwargs = dict(
            tools=case.get("tools"),
            thinking=thinking_mode == "thinking",
            reasoning_effort=case.get("reasoning_effort"),
        )
        prompt = render(messages, add_generation_prompt=not final_assistant, **kwargs)
        if final_assistant:
            # By default a new assistant turn follows the final assistant message.
            opener = "<｜Assistant｜>" + (
                "<think>" if thinking_mode == "thinking" else "</think>"
            )
            assert render(messages, **kwargs) == expected + opener
    assert prompt == expected


def test_drop_thinking_keeps_current_turn_across_tool_results():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {
            "role": "assistant",
            "reasoning_content": "r2",
            "tool_calls": [tool_call("c1")],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "value"},
    ]
    prompt = render(messages)
    assert "<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>" in prompt
    assert "r1" not in prompt
    assert "<think>r2</think>" in prompt
    assert prompt.endswith("<tool_result>value</tool_result><｜Assistant｜><think>")

    next_turn = render(messages + [{"role": "user", "content": "q3"}])
    assert "r2" not in next_turn
    assert next_turn.endswith("q3<｜Assistant｜><think>")


@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize("as_part", [False, True])
def test_image_placeholder_in_text_is_plain_text(role, as_part):
    text = f"hi {IMAGE_PLACEHOLDER}"
    content = [{"type": "text", "text": text}] if as_part else text
    prompt = render([{"role": role, "tool_call_id": "c1", "content": content}])
    assert text in prompt


def test_image_placeholder_in_reasoning_is_plain_text():
    prompt = render(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "reasoning": IMAGE_PLACEHOLDER},
        ]
    )
    assert IMAGE_PLACEHOLDER in prompt


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        (None, 75),
        ("low", 50),
        ("high", 75),
        ("xhigh", 75),
        ("max", 100),
        (1, 1),
        (42, 42),
        (100, 100),
    ],
)
def test_numeric_reasoning_effort(effort, budget):
    assert render(
        [{"role": "user", "content": "question"}], reasoning_effort=effort
    ) == (
        "<｜begin▁of▁sentence｜><｜System｜>"
        f"Reasoning Effort: {budget} (range 1-100, the higher the value, "
        "the more thorough the reasoning)\n\n"
        "<｜User｜>question<｜Assistant｜><think>"
    )


@pytest.mark.parametrize("effort", ["medium", "minimal", -1, 0, 101, True, 1.5, []])
def test_invalid_effort_is_a_request_error(effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        render([{"role": "user", "content": "question"}], reasoning_effort=effort)


@pytest.mark.parametrize(
    "controls",
    [
        {"thinking": False},
        {"enable_thinking": False},
        {"thinking": True, "reasoning_effort": "none"},
        {"enable_thinking": True, "reasoning_effort": "none"},
    ],
)
def test_chat_mode_has_closed_thinking_prefix(controls):
    assert render([{"role": "user", "content": "question"}], **controls) == (
        "<｜begin▁of▁sentence｜><｜User｜>question<｜Assistant｜></think>"
    )


def test_raw_text_parts_preserve_reference_separator():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }
    ]
    original = copy.deepcopy(messages)
    assert (
        render(
            messages,
            conversation=[{"role": "user", "content": "first\nsecond"}],
            thinking=False,
        )
        == "<｜begin▁of▁sentence｜><｜User｜>first\n\nsecond<｜Assistant｜></think>"
    )
    assert messages == original


@pytest.mark.parametrize(
    ("content", "stream"),
    [
        ([], False),
        ([], True),
        ([{"type": "text", "text": "hello"}], False),
    ],
)
def test_request_content_iterator_can_be_rendered_repeatedly(content, stream):
    request = ChatCompletionRequest.model_validate(
        {
            "model": "deepseek-ai/DeepSeek-V4.1-Flash",
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
        }
    )

    first = render(request.messages, thinking=False)
    assert first == render(request.messages, thinking=False)
    assert request.messages[0]["content"] == content
    request.model_dump_json()
    assert first == render(request.messages, thinking=False)


@pytest.mark.parametrize(
    ("role", "responses_type"),
    [("user", "input_text"), ("assistant", "output_text")],
)
def test_responses_text_parts_match_chat_text_parts(role, responses_type):
    responses_message = {
        "role": role,
        "content": [{"type": responses_type, "text": "hello"}],
    }
    chat_message = {
        "role": role,
        "content": [{"type": "text", "text": "hello"}],
    }
    assert render([responses_message], thinking=False) == render(
        [chat_message], thinking=False
    )


def test_mid_system_gets_its_own_marker_and_generation_header():
    assert render(
        [
            {"role": "user", "content": "question"},
            {"role": "system", "content": "answer briefly"},
        ],
        thinking=False,
    ) == (
        "<｜begin▁of▁sentence｜><｜User｜>question"
        "<｜System｜>answer briefly<｜Assistant｜></think>"
    )


def test_top_level_effort_overrides_template_effort():
    request = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        messages=[{"role": "user", "content": "question"}],
        reasoning_effort="low",
        chat_template_kwargs={"reasoning_effort": 100},
    )
    kwargs = request.build_chat_params(None, "auto").get_apply_chat_template_kwargs()
    assert "Reasoning Effort: 50 " in render(request.messages, **kwargs)


def test_encode_uses_one_bos_and_forwards_truncation():
    tokenizer = get_deepseek_v41_tokenizer(FakeHfTokenizer())
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "question"}],
        thinking=False,
        truncation=True,
        max_length=16,
    )
    text, add_special_tokens, kwargs = tokenizer.last_encode
    assert text.startswith("<｜begin▁of▁sentence｜>")
    assert add_special_tokens is False
    assert kwargs == {"truncation": True, "max_length": 16}


@pytest.mark.parametrize(
    "image_type", ["image_url", "input_image", "image_pil", "image_embeds"]
)
def test_images_preserve_content_order_and_reference_separator(image_type):
    from PIL import Image

    def image_part(color):
        if image_type == "image_pil":
            return {"type": image_type, "image_pil": Image.new("RGB", (2, 2), color)}
        if image_type == "image_embeds":
            return {"type": image_type, "image_embeds": {"llm_grid": [2, 2]}}
        url = f"https://example.com/{color}.png"
        return {
            "type": image_type,
            "image_url": {"url": url} if image_type == "image_url" else url,
        }

    parts = [
        {"type": "text", "text": "first"},
        image_part("red"),
        {"type": "text", "text": "second"},
        image_part("blue"),
        {"type": "text", "text": "compare"},
    ]
    assert render([{"role": "user", "content": parts}], thinking=False) == (
        "<｜begin▁of▁sentence｜><｜User｜>first\n\n<｜deepseek_image｜>\n\n"
        "second\n\n<｜deepseek_image｜>\n\ncompare<｜Assistant｜></think>"
    )
    assert parts[1]["type"] == parts[3]["type"] == image_type


def test_unsupported_media_is_a_request_error():
    with pytest.raises(ValueError, match="text and image content only"):
        render([{"role": "user", "content": [{"type": "input_audio"}]}])


def test_tool_references_render_as_tool_names():
    messages = [
        {"role": "assistant", "tool_calls": [tool_call("tool_search")]},
        {
            "role": "tool",
            "tool_call_id": "tool_search",
            "content": [
                {"type": "tool_reference", "name": "get_weather"},
                {"type": "tool_reference", "name": "get_time"},
            ],
        },
    ]
    assert render(messages, thinking=False).endswith(
        "<｜User｜><tool_result>get_weather\n\nget_time</tool_result>"
        "<｜Assistant｜></think>"
    )


def function_tool(name, description, parameters, deferred):
    function = {"name": name, "description": description, "parameters": parameters}
    return {"type": "function", "function": function, "defer_loading": deferred}


TOOL_SEARCH_TOOLS = [
    function_tool(
        "tool_search",
        "Search tools",
        {"type": "object", "properties": {"query": {"type": "string"}}},
        False,
    ),
    function_tool(
        "get_weather",
        "Weather",
        {"type": "object", "properties": {"city": {"type": "string"}}},
        True,
    ),
    function_tool("get_time", "Time", {"type": "object"}, True),
]


def tool_search_messages(*references):
    search = tool_call("tool_search")
    search["function"]["arguments"] = '{"query": "weather"}'
    return [
        {"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "tool_calls": [search]},
        {
            "role": "tool",
            "tool_call_id": "tool_search",
            "content": [{"type": "tool_reference", "name": n} for n in references],
        },
        {"role": "user", "content": "go on"},
    ]


def test_tool_references_list_deferred_tools_like_deepseek_recipe():
    # deepseek-recipe (935817c) leaves deferred tools out of the tool prompt
    # and lists referenced ones after the tool search result's user turn.
    # Same fixture as the Rust renderer's test of the same name.
    expected = (FIXTURES / "deepseek_v41" / "tool_search_references.txt").read_text()
    messages = tool_search_messages("get_weather", "get_time")
    assert render(messages, tools=TOOL_SEARCH_TOOLS, thinking=False) == expected


@pytest.mark.parametrize(
    ("messages", "tools", "match"),
    [
        (tool_search_messages(), TOOL_SEARCH_TOOLS[1:], "cannot be deferred"),
        (tool_search_messages("nope"), TOOL_SEARCH_TOOLS, "'nope' not found"),
    ],
    ids=["all_deferred", "unknown_reference"],
)
def test_tool_search_request_errors(messages, tools, match):
    with pytest.raises(ValueError, match=match):
        render(messages, tools=tools)


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def render_with_media(request, monkeypatch):
    # Keep real message parsing and image tracking; no model download is needed.
    processor = SimpleNamespace(
        info=SimpleNamespace(validate_num_items=lambda *args: None)
    )
    model_cls = SimpleNamespace(
        get_placeholder_str=lambda *args: "<｜deepseek_image｜>"
    )
    monkeypatch.setattr(
        BaseMultiModalItemTracker, "mm_processor", property(lambda self: processor)
    )
    monkeypatch.setattr(
        BaseMultiModalItemTracker, "model_cls", property(lambda self: model_cls)
    )
    cls = RENDERER_REGISTRY.load_renderer_cls("deepseek_v41")
    renderer = cast(DeepseekV4Renderer, cls.__new__(cls))
    renderer.model_config = SimpleNamespace(
        multimodal_config=None,
        allowed_local_media_path="",
        allowed_media_domains=None,
        enable_prompt_embeds=False,
        is_multimodal_model=True,
        hf_config=SimpleNamespace(),
    )
    renderer.tokenizer = get_deepseek_v41_tokenizer(FakeHfTokenizer())
    renderer._apply_chat_template_async = make_async(renderer._apply_chat_template)

    def run(messages, **kwargs):
        expected = render(messages, **kwargs)
        params = ChatParams(chat_template_kwargs={"tokenize": False, **kwargs})
        if request.param:
            _, prompt = asyncio.run(renderer.render_messages_async(messages, params))
        else:
            _, prompt = renderer.render_messages(messages, params)
        # The renderer returns token ids: the engine would otherwise tokenize
        # the text again and match reserved tokens in client text.
        assert prompt.get("prompt_token_ids") == [len(expected)]
        # Expose the rendered text for ordering assertions.
        cast(dict[str, Any], prompt)["prompt"] = expected
        return prompt

    return run


def _image(color):
    from PIL import Image

    return {
        "type": "image_pil",
        "image_pil": Image.new("RGB", (2, 2), color),
        "uuid": color,
    }


def test_tool_images_follow_rendered_order(render_with_media):
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": color,
                    "type": "function",
                    "function": {"name": color, "arguments": "{}"},
                }
                for color in ("red", "blue")
            ],
        },
        *[
            {
                "role": "tool",
                "tool_call_id": color,
                "content": [{"type": "text", "text": color}, _image(color)],
            }
            for color in ("blue", "red")
        ],
    ]
    prompt = render_with_media(messages)
    assert prompt["prompt"].index("<tool_result>red") < prompt["prompt"].index(
        "<tool_result>blue"
    )
    assert prompt["multi_modal_uuids"]["image"] == ["red", "blue"]
    assert [im.getpixel((0, 0)) for im in prompt["multi_modal_data"]["image"]] == [
        (255, 0, 0),
        (0, 0, 255),
    ]


def test_dropped_developer_only_image_leaves_text_prompt(render_with_media):
    prompt = render_with_media(
        [
            {"role": "developer", "content": [_image("red")]},
            {"role": "user", "content": "question"},
        ]
    )
    assert "<｜deepseek_image｜>" not in prompt["prompt"]
    assert "multi_modal_data" not in prompt
    assert "multi_modal_uuids" not in prompt


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("final_role", ["user", "developer", "system", "assistant"])
def test_generation_prompt_only_controls_final_turn(thinking, final_role):
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": final_role, "content": "final message"},
    ]
    original = copy.deepcopy(messages)
    closed = render(messages, thinking=thinking, add_generation_prompt=False)
    opened = render(messages, thinking=thinking, add_generation_prompt=True)
    cue = "<｜Assistant｜>" + ("<think>" if thinking else "</think>")

    assert opened == closed + cue
    assert "first answer<｜end▁of▁sentence｜>" in closed
    assert "<｜User｜>first question<｜Assistant｜>" in closed
    assert messages == original


@pytest.mark.parametrize("thinking", [False, True])
def test_continue_final_message_preserves_historical_eos(thinking):
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "The answer is "},
    ]
    original = copy.deepcopy(messages)
    closed = render(messages, thinking=thinking, add_generation_prompt=False)
    continued = render(
        messages,
        thinking=thinking,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    assert closed == continued + "<｜end▁of▁sentence｜>"
    assert continued.endswith("The answer is ")
    assert "first answer<｜end▁of▁sentence｜>" in continued
    assert messages == original


def test_continue_final_message_through_chat_request():
    request = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        messages=[
            {"role": "user", "content": "Finish this sentence."},
            {"role": "assistant", "content": "The answer is "},
        ],
        continue_final_message=True,
    )
    kwargs = request.build_chat_params(None, "auto").get_apply_chat_template_kwargs()
    assert render(request.messages, **kwargs).endswith("The answer is ")


def test_continue_final_reasoning_keeps_thinking_open():
    messages = [
        {"role": "user", "content": "Explain your answer."},
        {"role": "assistant", "content": None, "reasoning": "Let me consider "},
    ]
    assert render(messages, continue_final_message=True).endswith(
        "<｜Assistant｜><think>Let me consider "
    )


@pytest.mark.parametrize("thinking_mode", ["chat", "thinking"])
@pytest.mark.parametrize(
    "final",
    [
        {"content": "The answer is "},
        {"content": "", "reasoning_content": "Let me consider "},
        {"content": "", "tool_calls": [tool_call("search")]},
    ],
)
def test_open_final_message_gets_no_generation_prompt(thinking_mode, final):
    # A final assistant message left open with wo_eos is never followed by a
    # new assistant turn, whatever add_generation_prompt says.
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "wo_eos": True, **final},
    ]

    def encode(**kwargs):
        return encode_messages(
            copy.deepcopy(messages), thinking_mode=thinking_mode, **kwargs
        )

    closed = encode(add_generation_prompt=False)
    assert not closed.endswith("<｜end▁of▁sentence｜>")
    assert encode() == closed


@pytest.mark.parametrize("messages", [[], [{"role": "user", "content": "hello"}]])
def test_continue_final_message_requires_assistant(messages):
    with pytest.raises(ValueError, match="final assistant"):
        render(messages, continue_final_message=True)


@pytest.mark.skip_global_cleanup
def test_deepseek_v41_unknown_role_raises_value_error():
    # Invalid roles are client errors: they must surface as ValueError
    # (mapped to HTTP 400 by the OpenAI serving layer), not
    # NotImplementedError (mapped to HTTP 501).
    with pytest.raises(ValueError, match="Invalid role: SYSTEM"):
        encode_messages(
            [{"role": "SYSTEM", "content": "Hello"}],
            thinking_mode="chat",
        )


class _AddedTokenTokenizer:
    """Fake tokenizer that matches added tokens in text, like the real one."""

    vocab_size = 100
    IMAGE_ID = 1_000_000
    THINK_ID = 1_000_001
    END_THINK_ID = 1_000_002
    USER_ID = 1_000_003
    EOS_ID = 1_000_004
    REF_ID = 1_000_005

    def get_added_vocab(self) -> dict[str, int]:
        return {
            IMAGE_PLACEHOLDER: self.IMAGE_ID,
            "<think>": self.THINK_ID,
            "</think>": self.END_THINK_ID,
            "<｜User｜>": self.USER_ID,
            "<｜end▁of▁sentence｜>": self.EOS_ID,
            "<｜ref｜>": self.REF_ID,
        }

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        added = self.get_added_vocab()
        ids: list[int] = []
        while text:
            hits = [(text.find(t), t) for t in added if t in text]
            if not hits:
                ids += [ord(c) for c in text]
                break
            idx, tok = min(hits)
            ids += [ord(c) for c in text[:idx]] + [added[tok]]
            text = text[idx + len(tok) :]
        return ids


def _tokenize(messages):
    tok = get_deepseek_v41_tokenizer(_AddedTokenTokenizer())
    return tok.apply_chat_template(messages, tokenize=True)


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_image_placeholder_in_text_is_not_encoded_as_special_token(role):
    messages = [{"role": "user", "content": "q"}]
    if role == "assistant":
        messages.append({"role": "assistant", "content": f"a {IMAGE_PLACEHOLDER}"})
        messages.append({"role": "user", "content": "q2"})
    else:
        messages[0]["content"] = f"q {IMAGE_PLACEHOLDER}"
    assert _AddedTokenTokenizer.IMAGE_ID not in _tokenize(messages)


def test_image_part_still_encodes_special_token():
    ids = _tokenize(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "x"}},
                    {"type": "text", "text": f"hi {IMAGE_PLACEHOLDER}"},
                ],
            }
        ]
    )
    assert ids.count(_AddedTokenTokenizer.IMAGE_ID) == 1


@pytest.mark.parametrize(
    "text",
    [
        f"x {IMAGE_PLACEHOLDER}",
        "x <think>",
        "x </think>",
        "x <｜User｜>",
        "x <｜end▁of▁sentence｜>",
    ],
)
def test_renderer_keeps_reserved_tokens_as_text(text):
    # The online renderer passes tokenize=False; the V4.1 renderer must still
    # return token ids, otherwise the engine tokenizes the text again and
    # matches the reserved tokens as special tokens.
    cls = RENDERER_REGISTRY.load_renderer_cls("deepseek_v41")
    renderer = cast(DeepseekV4Renderer, cls.__new__(cls))
    renderer.model_config = SimpleNamespace(
        multimodal_config=None,
        allowed_local_media_path="",
        allowed_media_domains=None,
        enable_prompt_embeds=False,
        is_multimodal_model=False,
        hf_config=SimpleNamespace(),
    )
    tokenizer = get_deepseek_v41_tokenizer(_AddedTokenTokenizer())
    renderer.tokenizer = tokenizer
    messages = [{"role": "user", "content": text}]
    params = ChatParams(chat_template_kwargs={"tokenize": False})
    _, prompt = renderer.render_messages(messages, params)
    _, baseline = renderer.render_messages([{"role": "user", "content": "x"}], params)
    ids = cast(dict[str, Any], prompt)["prompt_token_ids"]
    base = cast(dict[str, Any], baseline)["prompt_token_ids"]
    # The chat template adds its own '<think>'; the text must add nothing.
    for special in (
        _AddedTokenTokenizer.IMAGE_ID,
        _AddedTokenTokenizer.THINK_ID,
        _AddedTokenTokenizer.END_THINK_ID,
        _AddedTokenTokenizer.USER_ID,
        _AddedTokenTokenizer.EOS_ID,
    ):
        assert ids.count(special) == base.count(special)


REF = "<｜ref｜>"
RESERVED = [
    IMAGE_PLACEHOLDER,
    "<think>",
    "</think>",
    "<｜User｜>",
    "<｜end▁of▁sentence｜>",
]
RESERVED_IDS = [
    _AddedTokenTokenizer.IMAGE_ID,
    _AddedTokenTokenizer.THINK_ID,
    _AddedTokenTokenizer.END_THINK_ID,
    _AddedTokenTokenizer.USER_ID,
    _AddedTokenTokenizer.EOS_ID,
]


def _call(name="f", arguments="{}"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _extra_special_ids(build, text, special_id):
    """How many more times `special_id` is in the prompt of `text` than of 'x'."""
    tok = get_deepseek_v41_tokenizer(_AddedTokenTokenizer())

    def count(value):
        messages, tools = build(value)
        kwargs = {"tools": tools} if tools else {}
        return tok.apply_chat_template(messages, tokenize=True, **kwargs).count(
            special_id
        )

    return count(f"a {text} b") - count("x")


@pytest.mark.parametrize("field", ["content", "reasoning_content"])
def test_assistant_keeps_grounding_tokens_only(field):
    # As in deepseek-recipe: grounding tokens stay special tokens in assistant
    # content and reasoning content, every other reserved token is plain text.
    def build(value):
        if field == "content":
            return [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": value},
                {"role": "user", "content": "ok"},
            ], None
        return [
            {"role": "user", "content": "go"},
            {**_call(), "reasoning_content": value},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ], None

    assert _extra_special_ids(build, REF, _AddedTokenTokenizer.REF_ID) == 1
    for text, special_id in zip(RESERVED, RESERVED_IDS):
        assert _extra_special_ids(build, text, special_id) == 0


def test_user_and_tool_text_do_not_keep_grounding_tokens():
    def user(value):
        return [{"role": "user", "content": value}], None

    def tool(value):
        return [
            {"role": "user", "content": "go"},
            _call(),
            {"role": "tool", "tool_call_id": "c1", "content": value},
            {"role": "user", "content": "ok"},
        ], None

    for build in (user, tool):
        assert _extra_special_ids(build, REF, _AddedTokenTokenizer.REF_ID) == 0


@pytest.mark.parametrize(
    "field", ["arguments", "arguments_key", "double_encoded", "name", "description"]
)
@pytest.mark.parametrize("text", RESERVED + [REF])
def test_tool_fields_are_plain_text(field, text):
    # As in deepseek-recipe: tool call names and arguments and tool definitions
    # are plain text, grounding tokens included.
    def build(value):
        tools = None
        if field == "arguments":
            call = _call(arguments=json.dumps({"x": value}))
        elif field == "arguments_key":
            call = _call(arguments=json.dumps({value: 1}))
        elif field == "double_encoded":
            call = _call(arguments=json.dumps(json.dumps({"x": value})))
        elif field == "name":
            call = _call(name="f" + value)
        else:
            call = _call()
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "description": value,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        messages = [
            {"role": "user", "content": "go"},
            call,
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        return messages, tools

    ids = dict(zip(RESERVED + [REF], RESERVED_IDS + [_AddedTokenTokenizer.REF_ID]))
    assert _extra_special_ids(build, text, ids[text]) == 0
