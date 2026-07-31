# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the TranslateGemma extra-fields-on-content-parts pathway.

TranslateGemma's bundled chat template reads ``source_lang_code`` and
``target_lang_code`` off each content part to assemble the translation prompt.
This PR makes sure those extra fields survive request parsing and reach the
template; these tests verify the full pathway with the real model on both
the text and image content branches of the template.
"""

import json

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer
from vllm.assets.image import ImageAsset
from vllm.multimodal.utils import encode_image_url

MODEL_NAME = "google/translategemma-4b-it"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "16",
        "--enforce-eager",
        "--chat-template-content-format",
        "openai",
        "--limit-mm-per-prompt",
        json.dumps({"image": 1}),
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.fixture(scope="module")
def stop_sign_image_url():
    return encode_image_url(ImageAsset("stop_sign").pil_image)


@pytest.mark.asyncio
async def test_translategemma_extra_lang_code_fields(client: openai.AsyncOpenAI):
    """en -> es translation through TranslateGemma's bundled chat template's
    ``text`` content branch, which depends on ``source_lang_code`` /
    ``target_lang_code`` extra fields being preserved on the content part."""
    completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The quick brown fox jumps over the lazy dog.",
                        "source_lang_code": "en",
                        "target_lang_code": "es",
                    }
                ],
            }
        ],
        max_tokens=64,
        temperature=0.0,
    )

    content = completion.choices[0].message.content
    assert content is not None and content.strip(), (
        "expected a non-empty Spanish translation"
    )
    # If extra fields had been stripped, the bundled template would not
    # produce a translation prompt and the output would not be Spanish.
    spanish_markers = (" el ", " la ", " los ", " las ", " perro", " zorro")
    lowered = f" {content.lower()} "
    assert any(m in lowered for m in spanish_markers), (
        f"output does not look like Spanish: {content!r}"
    )


@pytest.mark.asyncio
async def test_translategemma_image_extra_lang_code_fields(
    client: openai.AsyncOpenAI,
    stop_sign_image_url: str,
):
    """en -> es OCR-translation through TranslateGemma's bundled chat
    template's ``image`` content branch. Exercises the multimodal branch of
    ``_collect_extra_fields`` in ``_parse_chat_message_content_part``: the
    ``source_lang_code`` / ``target_lang_code`` extras must survive parsing
    and end up alongside the ``{"type": "image"}`` placeholder so the
    template's image branch (``content["type"] == 'image'``) renders the
    right OCR-translation prompt."""
    completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": stop_sign_image_url},
                        "source_lang_code": "en",
                        "target_lang_code": "es",
                    }
                ],
            }
        ],
        max_tokens=64,
        temperature=0.0,
    )

    content = completion.choices[0].message.content
    assert content is not None and content.strip(), (
        "expected a non-empty translation of the stop-sign image"
    )
    # Common Spanish renderings of "STOP" on a road sign. If extras were
    # stripped we would either fail the template or get back the original
    # English word, neither of which match this set.
    spanish_stop_terms = ("detén", "deteng", "deten", "alto", "pare", "para")
    lowered = content.lower()
    assert "stop" not in lowered, (
        f"output appears to echo the English source: {content!r}"
    )
    assert any(t in lowered for t in spanish_stop_terms), (
        f"output is not a recognizable Spanish translation of 'STOP': {content!r}"
    )
