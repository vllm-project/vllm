# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import openai
import pytest

from tests.models.registry import HF_EXAMPLE_MODELS
from tests.utils import RemoteOpenAIServer
from vllm.assets.audio import AudioAsset
from vllm.multimodal.utils import encode_audio_base64, encode_audio_url

MODEL_NAME = "OpenMOSS-Team/MOSS-Audio-4B-Instruct"
TEST_AUDIO = AudioAsset("winning_call")
MARY_HAD_LAMB_AUDIO = AudioAsset("mary_had_lamb")
QUESTION = "What is happening in this audio?"
MARY_HAD_LAMB_QUESTION = "Transcribe this audio."


@pytest.fixture(scope="module")
def server():
    model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL_NAME)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    args = [
        "--dtype",
        "half",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "1",
        "--enforce-eager",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        json.dumps({"audio": 1}),
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture
def client(server):
    return server.get_client()


def _messages_from_audio_url(
    audio_url: str,
    question: str = QUESTION,
) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_url}},
                {"type": "text", "text": question},
            ],
        }
    ]


def _messages_from_input_audio(audio_b64: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                },
                {"type": "text", "text": QUESTION},
            ],
        }
    ]


def _get_test_audio_data():
    return TEST_AUDIO.audio_and_sample_rate


def test_single_chat_session_audio_url(
    client: openai.OpenAI,
):
    audio_url = encode_audio_url(*_get_test_audio_data())
    chat_completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=_messages_from_audio_url(audio_url),
        max_completion_tokens=16,
        temperature=0.0,
    )

    assert len(chat_completion.choices) == 1
    choice = chat_completion.choices[0]
    assert choice.message.content is not None
    assert choice.message.content.strip() != ""


def test_mary_had_lamb_audio_url(
    client: openai.OpenAI,
):
    audio_url = encode_audio_url(*MARY_HAD_LAMB_AUDIO.audio_and_sample_rate)
    chat_completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=_messages_from_audio_url(audio_url, MARY_HAD_LAMB_QUESTION),
        max_completion_tokens=16,
        temperature=0.0,
    )

    assert len(chat_completion.choices) == 1
    choice = chat_completion.choices[0]
    assert choice.message.content is not None
    assert choice.message.content.strip() != ""


def test_single_chat_session_input_audio(
    client: openai.OpenAI,
):
    audio_b64 = encode_audio_base64(*_get_test_audio_data())
    chat_completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=_messages_from_input_audio(audio_b64),
        max_completion_tokens=16,
        temperature=0.0,
    )

    assert len(chat_completion.choices) == 1
    choice = chat_completion.choices[0]
    assert choice.message.content is not None
    assert choice.message.content.strip() != ""
