# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import pytest_asyncio
from mistral_common.audio import Audio
from mistral_common.protocol.instruct.chunk import AudioChunk, RawAudio, TextChunk
from mistral_common.protocol.instruct.messages import UserMessage

from vllm.tokenizers.mistral import MistralTokenizer

from ....conftest import AudioTestAssets
from ....utils import RemoteOpenAIServer
from .test_ultravox import MULTI_AUDIO_PROMPT, run_multi_audio_test

MODEL_NAME = "mistralai/Voxtral-Mini-3B-2507"
MISTRAL_FORMAT_ARGS = [
    "--tokenizer_mode",
    "mistral",
    "--config_format",
    "mistral",
    "--load_format",
    "mistral",
]


@pytest.fixture()
def server(request, audio_assets: AudioTestAssets):
    args = [
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"audio": len(audio_assets)}),
    ] + MISTRAL_FORMAT_ARGS

    with RemoteOpenAIServer(
        MODEL_NAME, args, env_dict={"VLLM_AUDIO_FETCH_TIMEOUT": "30"}
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


def _get_prompt(audio_assets, question):
    tokenizer = MistralTokenizer.from_pretrained(MODEL_NAME)

    audios = [
        Audio.from_file(str(audio_assets[i].get_local_path()), strict=False)
        for i in range(len(audio_assets))
    ]
    audio_chunks = [
        AudioChunk(input_audio=RawAudio.from_audio(audio)) for audio in audios
    ]

    text_chunk = TextChunk(text=question)
    messages = [UserMessage(content=[*audio_chunks, text_chunk]).to_openai()]

    return tokenizer.apply_chat_template(messages=messages)


@pytest.mark.core_model
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models_with_multiple_audios(
    vllm_runner,
    audio_assets: AudioTestAssets,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    vllm_prompt = _get_prompt(audio_assets, MULTI_AUDIO_PROMPT)
    run_multi_audio_test(
        vllm_runner,
        [(vllm_prompt, [audio.audio_and_sample_rate for audio in audio_assets])],
        MODEL_NAME,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tokenizer_mode="mistral",
    )


@pytest.mark.asyncio
async def test_online_serving(client, audio_assets: AudioTestAssets):
    """Exercises online serving with/without chunked prefill enabled."""

    def asset_to_chunk(asset):
        audio = Audio.from_file(str(asset.get_local_path()), strict=False)
        audio.format = "wav"
        audio_dict = AudioChunk.from_audio(audio).to_openai()
        return audio_dict

    audio_chunks = [asset_to_chunk(asset) for asset in audio_assets]
    text = f"What's happening in these {len(audio_assets)} audio clips?"
    messages = [
        {
            "role": "user",
            "content": [*audio_chunks, {"type": "text", "text": text}],
        }
    ]

    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME, messages=messages, max_tokens=10
    )

    assert len(chat_completion.choices) == 1
    choice = chat_completion.choices[0]
    assert choice.message.content == "In the first audio clip, you hear a brief"
    assert choice.finish_reason == "length"
