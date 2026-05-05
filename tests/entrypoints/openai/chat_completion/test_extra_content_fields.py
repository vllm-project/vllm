# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for the TranslateGemma extra-fields-on-content-parts pathway.

TranslateGemma's bundled chat template reads ``source_lang_code`` and
``target_lang_code`` off each content part to assemble the translation prompt.
This PR makes sure those extra fields survive request parsing and reach the
template; this test verifies the full pathway with the real model.
"""

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

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
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_translategemma_extra_lang_code_fields(client: openai.AsyncOpenAI):
    """en -> es translation through TranslateGemma's bundled chat template,
    which depends on ``source_lang_code``/``target_lang_code`` extra fields
    being preserved on the content part."""
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
