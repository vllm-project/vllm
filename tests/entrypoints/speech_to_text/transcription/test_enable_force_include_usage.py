# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer


@pytest.fixture(scope="module")
def transcription_server_with_force_include_usage():
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-num-seqs",
        "4",
        "--enforce-eager",
        "--enable-force-include-usage",
        "--gpu-memory-utilization",
        "0.2",
    ]

    with RemoteOpenAIServer("openai/whisper-large-v3-turbo", args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def transcription_client_with_force_include_usage(
    transcription_server_with_force_include_usage,
):
    async with (
        transcription_server_with_force_include_usage.get_async_client() as async_client
    ):
        yield async_client


@pytest.mark.asyncio
async def test_transcription_with_enable_force_include_usage(
    transcription_client_with_force_include_usage, winning_call
):
    res = (
        await transcription_client_with_force_include_usage.audio.transcriptions.create(
            model="openai/whisper-large-v3-turbo",
            file=winning_call,
            language="en",
            temperature=0.0,
            stream=True,
            timeout=30,
        )
    )

    async for chunk in res:
        if not len(chunk.choices):
            # final usage sent
            usage = chunk.usage
            assert isinstance(usage, dict)
            assert usage["prompt_tokens"] > 0
            assert usage["completion_tokens"] > 0
            assert usage["total_tokens"] > 0
        else:
            assert not hasattr(chunk, "usage")
