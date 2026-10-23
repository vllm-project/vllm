# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.renderers.chat_utils import (
    can_conversation_defer_media_fetching,
    has_non_text_content,
)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("Describe", False, id="text"),
        pytest.param(
            [{"type": "image_url", "image_url": {"url": "url"}}], True, id="image"
        ),
    ],
)
def test_has_non_text_content(content, expected):
    assert has_non_text_content([{"role": "user", "content": content}]) is expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(
            [{"type": "image_url", "image_url": {"url": "url"}, "uuid": "image"}],
            True,
            id="image-url",
        ),
        pytest.param(
            [
                {"type": "image_url", "image_url": {"url": "url"}, "uuid": "image"},
                {"type": "video_url", "video_url": {"url": "url"}},
            ],
            False,
            id="missing-uuid",
        ),
        pytest.param(
            [{"type": "image_url", "image_url": None, "uuid": "image"}],
            False,
            id="uuid-without-data",
        ),
        pytest.param(
            [
                {
                    "type": "input_audio",
                    "input_audio": {"data": "", "format": "wav"},
                    "uuid": "audio",
                }
            ],
            False,
            id="input-audio-without-data",
        ),
    ],
)
def test_can_conversation_defer_media_fetching(content, expected):
    """Fetching is deferred only if all media has UUIDs and some has data."""
    messages = [{"role": "user", "content": content}]

    assert can_conversation_defer_media_fetching(messages) is expected


def test_can_conversation_defer_media_fetching_does_not_consume_iterators():
    part = {"type": "image_url", "image_url": {"url": "url"}, "uuid": "image"}
    content = iter([part])

    messages = [{"role": "user", "content": content}]
    assert not can_conversation_defer_media_fetching(messages)
    assert next(content) == part
