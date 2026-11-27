# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import copy
import json
import string
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from tokenizers import AddedToken, Tokenizer, models
from transformers import TokenizersBackend

from tests.tokenizers_.test_deepseek_v4 import FakeHfTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.chat_utils import BaseMultiModalItemTracker
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
from vllm.renderers.params import ChatParams
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.deepseek_v41 import (
    _compile_added_tokens_pattern,
    get_deepseek_v41_tokenizer,
)
from vllm.tokenizers.deepseek_v41_encoding import IMAGE_PLACEHOLDER, encode_messages
from vllm.utils.async_utils import make_async

FIXTURES = Path(__file__).parent / "fixtures"


def render(messages, **kwargs):
    return get_deepseek_v41_tokenizer(FakeHfTokenizer()).apply_chat_template(
        messages, tokenize=False, **kwargs
    )


def tool_call(name, arguments="{}"):
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
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
    renderer.tokenizer = _TOK
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
        ids = cast(dict[str, Any], prompt)["prompt_token_ids"]
        # The ids spell the text of the tokenizer's own text rendering, so the
        # assertions below are made on what the renderer returned.
        text = "".join(_TOK.convert_ids_to_tokens(ids))
        assert text == expected
        return prompt, text

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
    prompt, text = render_with_media(messages)
    assert text.index("<tool_result>red") < text.index("<tool_result>blue")
    assert prompt["multi_modal_uuids"]["image"] == ["red", "blue"]
    assert [im.getpixel((0, 0)) for im in prompt["multi_modal_data"]["image"]] == [
        (255, 0, 0),
        (0, 0, 255),
    ]


def test_dropped_developer_only_image_leaves_no_image(render_with_media):
    prompt, text = render_with_media(
        [
            {"role": "developer", "content": [_image("red")]},
            {"role": "user", "content": "question"},
        ]
    )
    assert "<｜deepseek_image｜>" not in text
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


_TINY_CHARS = sorted(
    set(string.ascii_letters + string.digits + " \n\"'`\\(){}[]:,.-_=*#!?/<>|｜▁")
)
REF = "<｜ref｜>"
# The reserved tokens under test.
RESERVED = [
    IMAGE_PLACEHOLDER,
    "<think>",
    "</think>",
    "<｜User｜>",
    "<｜end▁of▁sentence｜>",
]
# The added tokens of the tiny tokenizer: the ones under test and the ones the
# chat template adds.
_ADDED_TOKENS = [
    *RESERVED,
    REF,
    "<｜Assistant｜>",
    "<｜begin▁of▁sentence｜>",
    "｜DSML｜",
]


def _tiny_tokenizer(with_added: bool = True) -> TokenizersBackend:
    """A tiny real BPE tokenizer with the reserved tokens as added tokens.

    Its merges cross the characters next to a reserved token ('a<', '>b', ...),
    like a real BPE, so the ids of a text depend on the text around the token.
    """
    merges = [("a", "<"), (">", "b"), (" ", "<"), ("\n", "<")]
    vocab = {"<unk>": 0, **{c: i + 1 for i, c in enumerate(_TINY_CHARS)}}
    for left, right in merges:
        vocab[left + right] = len(vocab)
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    if with_added:
        tokenizer.add_tokens(
            [AddedToken(t, special=False, normalized=False) for t in _ADDED_TOKENS]
        )
    return TokenizersBackend(tokenizer_object=tokenizer)


_ID = _tiny_tokenizer().get_added_vocab()
_TOK = get_deepseek_v41_tokenizer(_tiny_tokenizer())


def _tokenize(messages):
    return _TOK.apply_chat_template(messages, tokenize=True)


def test_image_part_still_encodes_special_token():
    """A real image content part still produces the image placeholder token."""
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
    # Exactly one: the placeholder of the image part, not the text spelling it.
    assert ids.count(_ID[IMAGE_PLACEHOLDER]) == 1


def _tool_turn(call=None, result="ok", **assistant):
    call = call or tool_call("f")
    return [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [call], **assistant},
        {"role": "tool", "tool_call_id": call["id"], "content": result},
    ]


_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "f",
        "description": "d",
        "parameters": {"type": "object", "properties": {}},
    },
}

_USER = {"role": "user", "content": "go"}
_DEFERRED_TOOL_DESCRIPTION = "Schema for the referenced deferred tool"


def _tool_reference_case(value):
    name = "f" + value
    tool = function_tool(name, _DEFERRED_TOOL_DESCRIPTION, {"type": "object"}, True)
    return tool_search_messages(name), [TOOL_SEARCH_TOOLS[0], tool]


# Every place a client text enters a request: value -> (messages, tools).
_FIELDS = {
    "system": lambda v: ([{"role": "system", "content": v}, _USER], None),
    # A developer message before the last user message is dropped.
    "developer": lambda v: ([_USER, {"role": "developer", "content": v}], None),
    "user": lambda v: ([{"role": "user", "content": v}], None),
    "user_part": lambda v: (
        [{"role": "user", "content": [{"type": "text", "text": v}]}],
        None,
    ),
    # Encoder-internal keys from the client are ignored.
    "content_blocks": lambda v: (
        [
            {
                "role": "user",
                "content": v,
                "content_blocks": [{"type": "text", "text": v}],
            }
        ],
        None,
    ),
    "tool_result": lambda v: (_tool_turn(result=v), None),
    "tool_result_part": lambda v: (
        _tool_turn(result=[{"type": "text", "text": v}]),
        None,
    ),
    "tool_reference": _tool_reference_case,
    "assistant": lambda v: (
        [_USER, {"role": "assistant", "content": v}, {"role": "user", "content": "ok"}],
        None,
    ),
    "reasoning_content": lambda v: (
        _tool_turn(reasoning_content=v),
        None,
    ),
    "arguments": lambda v: (_tool_turn(tool_call("f", json.dumps({"x": v}))), None),
    "arguments_key": lambda v: (_tool_turn(tool_call("f", json.dumps({v: 1}))), None),
    "double_encoded": lambda v: (
        _tool_turn(tool_call("f", json.dumps(json.dumps({"x": v})))),
        None,
    ),
    "name": lambda v: (_tool_turn(tool_call("f" + v)), None),
    "description": lambda v: (
        [_USER],
        [{**_TOOL, "function": {**_TOOL["function"], "description": v}}],
    ),
    "response_format": lambda v: (
        [
            {"role": "system", "content": "s", "response_format": {"description": v}},
            _USER,
        ],
        None,
    ),
}


@pytest.mark.parametrize("field", _FIELDS)
@pytest.mark.parametrize("token", [IMAGE_PLACEHOLDER, "</think>", REF])
def test_reserved_tokens_in_text_are_plain_text(field, token):
    """As in deepseek-recipe, a reserved token in any client text is plain text;
    only the grounding tokens in assistant content and reasoning stay special."""
    special = token == REF and field in ("assistant", "reasoning_content")

    def render(value, tokenize):
        messages, tools = _FIELDS[field](value)
        kwargs = {"tools": tools} if tools else {}
        return _TOK.apply_chat_template(messages, tokenize=tokenize, **kwargs)

    text = f"a {token} b"
    extra = render(text, True).count(_ID[token]) - render("x", True).count(_ID[token])
    # The client text adds the token's id only where it is meant to stay special.
    assert extra == special
    # With tokenize=False the text comes back as sent...
    rendered = render(text, False)
    assert text in rendered
    # ...without a leftover escape marker.
    assert "\ue000" not in rendered
    if field == "tool_reference":
        # Check the rendered name itself, not just its copy in the schema, and
        # verify that the original name still resolves the deferred schema.
        assert f"<tool_result>f{text}</tool_result>" in rendered
        assert _DEFERRED_TOOL_DESCRIPTION in rendered


def test_non_text_reasoning_is_a_request_error():
    """As in deepseek-recipe, reasoning must be a string: any other value would
    reach the prompt unescaped."""
    messages = [_USER, {"role": "assistant", "reasoning_content": ["</think>"]}]
    with pytest.raises(ValueError, match="expects text"):
        _tokenize(messages)


@pytest.mark.parametrize(
    "context",
    [
        "a</think>b",
        "a<think>b",
        " <think>",
        "\n</think>",
        '"<｜User｜>"',
        "x<｜User｜><｜end▁of▁sentence｜>y",
        "foo(<think>)",
        '{"a": "a<think>b"}',
    ],
)
def test_plain_text_ids_are_the_ordinary_encoding(context):
    """As in deepseek-recipe, a stretch of plain text is encoded as a whole with
    the added vocabulary disabled, so a merge across the characters next to a
    reserved token ('a<', '>b', ...) is kept."""
    ordinary = _tiny_tokenizer(with_added=False)

    def ids(text):
        return _tokenize([{"role": "user", "content": text}])

    base = ids("Q")
    q = ordinary.encode("Q", add_special_tokens=False)
    # 'Q' is one id that appears once in the prompt: the place of the context.
    assert base.count(q[0]) == 1
    index = base.index(q[0])
    expected = (
        base[:index]
        + ordinary.encode(context, add_special_tokens=False)
        + base[index + 1 :]
    )
    # The template ids around the ordinary encoding of the whole context.
    assert ids(context) == expected


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("content", ["a <think> b", "a b"])
def test_truncation_follows_truncation_side_with_and_without_escaped_tokens(
    side, content
):
    """A prompt is truncated on the tokenizer's truncation side, whether or not
    it holds an escaped token."""
    tokenizer = get_deepseek_v41_tokenizer(_tiny_tokenizer())
    tokenizer.truncation_side = side
    messages = [{"role": "user", "content": content}]
    full = tokenizer.apply_chat_template(messages, tokenize=True)
    cut = tokenizer.apply_chat_template(
        messages, tokenize=True, truncation=True, max_length=5
    )
    # The last 5 ids when truncating on the left, the first 5 on the right.
    assert cut == (full[-5:] if side == "left" else full[:5])


def test_truncation_does_not_carry_over_to_escaped_prompts():
    """A truncated prompt leaves its truncation on the HF backend, which the
    plain text tokenizer must not inherit."""
    tokenizer = get_deepseek_v41_tokenizer(_tiny_tokenizer())
    tokenizer.apply_chat_template([_USER], tokenize=True, truncation=True, max_length=3)
    messages = [{"role": "user", "content": "a <think> b c d e f"}]
    assert tokenizer.apply_chat_template(messages, tokenize=True) == _tokenize(messages)


def test_added_tokens_pattern_prefers_the_longest_token():
    pattern = _compile_added_tokens_pattern(["<a>", "<a>b", "<ab>"])
    assert pattern is not None
    assert pattern.findall("<a>b <a>c <ab>") == ["<a>b", "<a>", "<ab>"]


def test_rendering_does_not_modify_the_request():
    """Escaping writes into nested fields (tool call names and arguments) of
    copies: the caller's messages and tools are unchanged."""
    messages = [
        {"role": "user", "content": "a <think> b"},
        {
            "role": "assistant",
            "tool_calls": [tool_call("c1", json.dumps({"x": "<｜User｜>"}))],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "</think>"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "description": "<｜User｜>",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    expected = copy.deepcopy((messages, tools))
    _TOK.apply_chat_template(messages, tools=tools, tokenize=True)
    # Same content as the deep copy taken before rendering.
    assert (messages, tools) == expected
