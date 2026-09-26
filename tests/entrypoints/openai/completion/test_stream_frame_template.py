# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frames rendered from the per-stream template must be byte-identical to the
frames the response models would have produced."""

import json

import pytest

from vllm.entrypoints.openai.completion.protocol import (
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)
from vllm.entrypoints.openai.completion.serving import _delta_frame_template

CREATED = 1706000000


def model_frame(request_id: str, model_name: str, index: int, text: str) -> str:
    chunk = CompletionStreamResponse(
        id=request_id,
        object="text_completion",
        created=CREATED,
        model=model_name,
        choices=[
            CompletionResponseStreamChoice(
                index=index,
                text=text,
                logprobs=None,
                finish_reason=None,
                stop_reason=None,
                prompt_token_ids=None,
                token_ids=None,
            )
        ],
    )
    return f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello, world",
        ' "quoted" and \\backslashed\\ ',
        "line\nbreaks\r\ttabs",
        "\x00\x01\x1f\x7f",
        "unicode é 日本語 🚀 \U0001d549\U0010ffff",
    ],
)
@pytest.mark.parametrize(
    ("request_id", "model_name", "index"),
    [
        ("cmpl-abc123", "meta-llama/Llama-3.1-8B", 0),
        ('id "with" quotes\\', "модель 🎯", 7),
    ],
)
def test_template_frame_matches_model_frame(
    request_id: str, model_name: str, index: int, text: str
):
    prefix, infix, suffix = _delta_frame_template(request_id, CREATED, model_name)
    frame = f"{prefix}{index}{infix}{json.dumps(text, ensure_ascii=False)}{suffix}"
    assert frame == model_frame(request_id, model_name, index, text)


def test_template_is_none_when_sentinels_are_ambiguous():
    assert _delta_frame_template("cmpl-987654321", CREATED, "m") is None


def test_delta_chunk_fields_are_frozen():
    """The template tracks field order and formatting, but not a new field the
    generator starts populating per chunk; adding one must fail here."""
    assert set(CompletionResponseStreamChoice.model_fields) == {
        "index",
        "text",
        "logprobs",
        "finish_reason",
        "stop_reason",
        "prompt_token_ids",
        "token_ids",
    }
