# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the TranslateGemma example chat template."""

from pathlib import Path

import jinja2
import jinja2.sandbox
import pytest

TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "tool_chat_template_translategemma.jinja"
)


def _raise_exception(message: str):
    raise jinja2.TemplateError(message)


@pytest.fixture(scope="module")
def translategemma_template():
    template_str = TEMPLATE_PATH.read_text(encoding="utf-8")
    env = jinja2.sandbox.ImmutableSandboxedEnvironment()
    return env.from_string(template_str)


def _render(template, messages, **kwargs):
    kwargs.setdefault("bos_token", "<bos>")
    kwargs.setdefault("add_generation_prompt", False)
    kwargs.setdefault("raise_exception", _raise_exception)
    return template.render(messages=messages, **kwargs)


class TestTranslateGemmaChatTemplate:
    def test_string_text_translation(self, translategemma_template):
        messages = [
            {
                "role": "user",
                "content": (
                    "<<<source>>>en<<<target>>>zh<<<text>>>"
                    "We distribute two models for language identification."
                ),
            }
        ]

        result = _render(
            translategemma_template,
            messages,
            add_generation_prompt=True,
        )

        assert result.startswith("<bos>")
        assert "English (en) to Chinese (zh) translator" in result
        assert "We distribute two models for language identification." in result
        assert result.endswith("<start_of_turn>model\n")

    def test_openai_text_array_translation(self, translategemma_template):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<<<source>>>en<<<target>>>hi<<<text>>>"
                            "Language identification"
                        ),
                    }
                ],
            }
        ]

        result = _render(translategemma_template, messages)

        assert "English (en) to Hindi (hi) translator" in result
        assert "Language identification" in result

    def test_openai_multiple_text_parts_are_concatenated(
        self, translategemma_template
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<<<source>>>en<<<target>>>hi<<<text>>>",
                    },
                    {
                        "type": "text",
                        "text": "Language identification",
                    },
                ],
            }
        ]

        result = _render(translategemma_template, messages)

        assert "English (en) to Hindi (hi) translator" in result
        assert "Language identification" in result

    def test_openai_image_array_translation(self, translategemma_template):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<<<source>>>cs<<<target>>>de-DE<<<image>>>",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/traffic-sign.jpg",
                        },
                    },
                ],
            }
        ]

        result = _render(translategemma_template, messages)

        assert "Czech (cs) to German (de-DE) translator" in result
        assert "Please translate the Czech text in the provided image" in result
        assert "<start_of_image>" in result
        assert "https://example.com/traffic-sign.jpg" not in result

    @pytest.mark.parametrize(
        "content",
        [
            "warmup",
            [{"type": "text", "text": "warmup"}],
        ],
    )
    def test_warmup_content(self, translategemma_template, content):
        result = _render(
            translategemma_template,
            [{"role": "user", "content": content}],
        )

        assert "<start_of_turn>user\nwarmup" in result

    def test_custom_prompt(self, translategemma_template):
        result = _render(
            translategemma_template,
            [{"role": "user", "content": "<<<custom>>>Describe this image."}],
        )

        assert "<start_of_turn>user\nDescribe this image." in result

    def test_custom_prompt_with_image_part(self, translategemma_template):
        result = _render(
            translategemma_template,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<<<custom>>>Describe this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/image.jpg",
                            },
                        },
                    ],
                }
            ],
        )

        assert "<start_of_turn>user\nDescribe this image." in result
        assert "<start_of_image>" in result

    def test_rejects_unknown_format(self, translategemma_template):
        with pytest.raises(jinja2.TemplateError, match="User role must provide content"):
            _render(translategemma_template, [{"role": "user", "content": "hello"}])
