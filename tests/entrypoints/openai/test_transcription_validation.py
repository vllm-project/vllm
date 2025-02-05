# SPDX-License-Identifier: Apache-2.0

# imports for guided decoding tests
import re

import openai
import pytest

from ...utils import RemoteOpenAIServer
from vllm.assets.audio import AudioAsset
import json

@pytest.fixture
def mary_had_lamb():
    path = AudioAsset('mary_had_lamb').get_asset_path()
    with open(str(path), "rb") as f:
        yield f

@pytest.fixture
def winning_call():
    path = AudioAsset('winning_call').get_asset_path()
    with open(str(path), "rb") as f:
        yield f

@pytest.mark.asyncio
async def test_basic_audio(mary_had_lamb):
    model_name = "openai/whisper-large-v3-turbo"
    server_args = ["--enforce-eager"]
    # Based on https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb.
    prompt="THE FIRST WORDS I SPOKE"
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        transcription = await client.audio.transcriptions.create(
            model=model_name,
            file=mary_had_lamb,
            language="en",
            response_format="text",
            temperature=0.0)
        out = json.loads(transcription)['text']
        assert "Mary had a little lamb," in out
        # This should "force" whisper to continue prompt in all caps
        transcription_wprompt = await client.audio.transcriptions.create(
            model=model_name,
            file=mary_had_lamb,
            language="en",
            response_format="text",
            prompt=prompt,
            temperature=0.0)
        out_capital = json.loads(transcription_wprompt)['text']
        assert prompt not in out_capital
        print(out_capital.capitalize(), out_capital)
        

@pytest.mark.asyncio
async def test_bad_requests(mary_had_lamb):
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        # invalid language
        with pytest.raises(openai.BadRequestError):
            await client.audio.transcriptions.create(
                model=model_name,
                file=mary_had_lamb,
                language="hh",
                response_format="text",
                temperature=0.0)
        
        # TODO audio too long
