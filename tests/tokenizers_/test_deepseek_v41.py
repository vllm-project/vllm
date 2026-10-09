# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tests.tokenizers_.test_deepseek_v4 import FakeHfTokenizer
from vllm.entrypoints.chat_utils import BaseMultiModalItemTracker
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
from vllm.renderers.params import ChatParams
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.deepseek_v41 import get_deepseek_v41_tokenizer
from vllm.utils.async_utils import make_async

FIXTURES = Path(__file__).parent / "fixtures"


def render(messages, **kwargs):
    return get_deepseek_v41_tokenizer(FakeHfTokenizer()).apply_chat_template(
        messages, tokenize=False, **kwargs
    )


@pytest.mark.parametrize("case_id", [1, 2])
def test_reference_encoder_fixtures(case_id):
    # Expected prompts come from the DeepSeek V4.1 reference encoder.
    data = json.loads(
        (FIXTURES / "deepseek_v4" / f"test_input_{case_id}.json").read_text()
    )
    messages = data["messages"] if isinstance(data, dict) else data
    tools = data.get("tools") if isinstance(data, dict) else None
    expected = (FIXTURES / "deepseek_v41" / f"test_output_{case_id}.txt").read_text()
    assert render(messages, tools=tools) == expected


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


@pytest.mark.parametrize("image_type", ["image_url", "input_image", "image_pil"])
def test_images_preserve_content_order_and_reference_separator(image_type):
    from PIL import Image

    def image_part(color):
        if image_type == "image_pil":
            return {"type": image_type, "image_pil": Image.new("RGB", (2, 2), color)}
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
        assert prompt.get("prompt") == expected
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
