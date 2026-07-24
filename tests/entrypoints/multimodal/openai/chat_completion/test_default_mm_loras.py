# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from tests.conftest import AudioTestAssets
from tests.utils import RemoteOpenAIServer

# NOTE - the tests in this module are currently analogous to test_chat, but are
# separated to avoid OOM killing due to module-scoped servers, since we
# need a multimodal model for these tests.

ACTIVE_MM_LORA_RESPONSE = "Spoken text: The first words I spoke in the original chronograph, a little piece of practical poetry. Mary had a little lamb, it slept with quite a snow, and everywhere that Mary went, the lamb was sure to go."  # noqa: E501


@pytest.fixture(scope="module")
def multimodal_server(
    phi4_multimodal_model_path: str,
    phi4_multimodal_audio_lora_path: str,
):
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "half",
        "--max-model-len",
        "4096",
        "--enforce-eager",
        # lora config below
        "--enable-lora",
        "--lora-modules",
        f"speech={phi4_multimodal_audio_lora_path}",
        "--max-lora-rank",
        "320",
        "--max-num-seqs",
        "2",
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.8",
        "--default-mm-loras",
        f'{{"audio": "{phi4_multimodal_audio_lora_path}"}}',
    ]

    with RemoteOpenAIServer(
        phi4_multimodal_model_path, args, max_wait_seconds=480
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def multi_modal_client(multimodal_server):
    async with multimodal_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # base model with default lora should give the same response as lora model
    "model_name",
    ["base", "speech"],
)
async def test_default_mm_lora_chat_completions(
    model_name: str,
    multi_modal_client: openai.AsyncOpenAI,
    audio_assets: AudioTestAssets,
    phi4_multimodal_model_path: str,
):
    served_model_name = (
        phi4_multimodal_model_path if model_name == "base" else model_name
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Can you transcribe this audio?",
                },
                {
                    "type": "audio_url",
                    "audio_url": {"url": audio_assets[0].url},
                },
            ],
        }
    ]

    chat_completion = await multi_modal_client.chat.completions.create(
        model=served_model_name,
        messages=messages,
        max_completion_tokens=128,
        temperature=0.0,
    )

    assert len(chat_completion.choices) > 0

    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0
    assert message.content == ACTIVE_MM_LORA_RESPONSE
